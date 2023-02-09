"""Some helpers for training.

All credits for early stopping go to Bjarte Mehus Sunde
(https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
"""

import colorama
import csv
import json
import logging
import numpy as np
import pandas as pd
import sys
import torch
import wandb

from utils import utils

import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaModel


colorama.init()

GREEN = colorama.Fore.GREEN
MAGENTA = colorama.Fore.MAGENTA
RED = colorama.Fore.RED
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET

DATE = datetime.now().strftime("%d_%m_%y_%H_%M")


logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fine_tuned_model(model_id, model_name):
    """..."""
    model = prepare_model(model_name=model_name)

    logging.info(f"{GREEN} Loading fine-tuned weights.{RESET}")
    if device == "cpu":
        model.load_state_dict(torch.load(model_id, map_location=torch.device("cpu")))
    else:
        model.load_state_dict(torch.load(model_id, map_location=torch.device("cpu")))
    return model


def save_fine_tuned_model(model, model_id):
    """..."""
    logging.info(f"{GREEN} Saving fine-tuned model.{RESET}")

    torch.save(model.state_dict(), model_id)


def prepare_model(model_name):
    """Load the model."""
    # Load BertForSequenceClassification, the pretrained BERT model with a
    # single linear classification layer on top.
    if model_name == "xlmroberta":

        model = XLMRobertaForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            # "bert-base-uncased",
            "xlm-roberta-base",
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # You can increase this for multi-class tasks.
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
        )
    elif model_name == "multibert":

        model = BertForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            # "bert-base-uncased",
            "bert-base-multilingual-cased",
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # You can increase this for multi-class tasks.
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
        )
    elif model_name == "character-bert":
        # config = AutoConfig.from_pretrained("helboukkouri/character-bert-medical")
        # model = AutoModelForSequenceClassification.from_config(config)
        model = AutoModelForSequenceClassification.from_pretrained(
            "helboukkouri/character-bert"
        )
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     "helboukkouri/character-bert",
        #     num_labels=2,
        #     # You can increase this for multi-class tasks.
        #     # Whether the model returns attentions weights.
        #     output_attentions=False,
        #     # Whether the model returns all hidden-states.
        #     output_hidden_states=False,
        # )
        print(f"model: {model}")
    elif model_name == "bioreddit-bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            "cambridgeltl/BioRedditBERT-uncased"
        )

    else:
        logging.warning(f"{RED} Please specify a model name.{RESET}")
        sys.exit()

    model = torch.nn.DataParallel(model)

    return model


def prepare_augmentation_model(model_name):
    """Load model for augmentation.

    We only need the pre-trained model without classification layer.
    """
    # Load BertForSequenceClassification, the pretrained BERT model with a
    # single linear classification layer on top.
    if model_name == "xlmroberta":

        model = XLMRobertaModel.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            # "bert-base-uncased",
            "xlm-roberta-base",
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # You can increase this for multi-class tasks.
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=True,
        )
    elif model_name == "multibert":

        model = BertModel.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            # "bert-base-uncased",
            "bert-base-multilingual-uncased",
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # You can increase this for multi-class tasks.
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=True,
        )

    else:
        logging.warning(f"{RED} Please specify a model name.{RESET}")
        sys.exit(1)

    return model


def get_results(y_true, y_pred, print_results=True):
    """Calculate all scores."""
    results = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0
    )

    try:
        results["auc"] = roc_auc_score(y_true, y_pred)
    # might occur when there is only one label occurring in the given test set
    except ValueError:
        results["auc"] = "NaN"

    if print_results:
        print(json.dumps(results, indent=2))
    # wandb.log(results)

    return results


def calculate_mcc(true_labels, predictions):
    """..."""
    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    logging.info("Calculating Matthews Corr. Coef. for each batch...")

    # For each input batch...
    for i in range(len(true_labels)):

        # The predictions for this batch are a 2-column ndarray (one column
        # for "0" and one column for "1"). Pick the label with the highest
        # value and turn this in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    # Create a barplot showing the MCC score for each batch of test
    # samples.
    # ax = sns.barplot(x=list(range(len(matthews_set))),
    #                  y=matthews_set, ci=None)

    # plt.title('MCC Score per Batch')
    # plt.ylabel('MCC Score (-1 to +1)')
    # plt.xlabel('Batch #')

    # plt.show()

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print("Total MCC: %.3f" % mcc)


def calculate_f1(true_labels, predictions, model_name, decoded_sentences):
    """..."""
    scores = []

    # Evaluate each test batch using Matthew's correlation coefficient
    logging.info(f"{GREEN} Calculating F1 score ...{RESET}")

    # For each input batch...
    for i in range(len(true_labels)):

        # Pick the label with the highest value and turn this
        # in a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        (
            precision_macro,
            recall_macro,
            fscore_macro,
            support,
        ) = precision_recall_fscore_support(
            true_labels[i], pred_labels_i, average="macro"
        )
        scores.append(fscore_macro)

        # (precision_standard,
        #     recall_standard,
        #     fscore_standard,
        #     support) = precision_recall_fscore_support(true_labels[i],
        #                                                pred_labels_i,
        #                                                average=None)

        # (precision_weighted,
        #     recall_weighted,
        #     fscore_weighted,
        #     support) = precision_recall_fscore_support(true_labels[i],
        #                                                pred_labels_i,
        #                                                average="weighted")

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    with open(f"results_{model_name}_{DATE}.csv", "w") as write_handle:
        write_handle.write(f"text\tpredicted\tgold\n")
        for sent, pred, true in zip(
            decoded_sentences, flat_predictions, flat_true_labels
        ):
            write_handle.write(f"{sent}\t{pred}\t{true}\n")

    results = classification_report(
        y_true=flat_true_labels,
        y_pred=flat_predictions,
        output_dict=True,
        zero_division="warn",
    )

    logging.info(
        f"{YELLOW}\n{classification_report(y_true=flat_true_labels, y_pred=flat_predictions)}{RESET}"
    )
    logging.info(f"{GREEN} Calculating ROC & AUC\n{RESET}")

    auc = roc_auc_score(flat_true_labels, flat_predictions)

    results["auc"] = auc
    return results


def get_results_per_language(results_df):
    """Calculate scores per language."""
    # columns=["text", "language" "predicted", "gold"]
    # get all languages
    langs = set(results_df["language"])
    # get rows per language
    for lang in set(results_df["language"]):
        data = results_df.loc[results_df["language"] == lang]
        report = classification_report(
            y_true=data["gold"], y_pred=data["predicted"], output_dict=True
        )
        for key, value in report.items():
            if isinstance(value, dict):
                for s, res in value.items():
                    score_name = f"{lang}_{s}_{key}"
                    wandb.log({score_name: res})
            else:
                wandb.log({f"{lang}_{key}": value})

        df = pd.DataFrame(report)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "score"}, inplace=True)
        wandb.log({f"scores_{lang}": wandb.Table(dataframe=df)})


def write_predictions_to_file(sentences, y_pred, y_true, file_name):
    """Save predictions in a CSV."""
    with open(file_name, "w") as write_handle:

        write_handle.write(f"text\tpredicted\tgold\n")

        for sent, pred, true in zip(sentences, y_pred, y_true):

            write_handle.write(f"{sent}\t{pred}\t{true}\n")


def flat_accuracy(preds, labels):
    """..."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Take a time in seconds and returns a string hh:mm:ss."""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))


def plot_f1_curves(data_train, data_val):
    """..."""
    plt.figure()
    plt.plot(
        range(0, len(data_train)), data_train, color="red", linewidth=3, linestyle="--"
    )
    plt.plot(
        range(0, len(data_val)), data_val, color="blue", linewidth=3, linestyle="--"
    )

    plt.title("Training (red) and validation (blue) F1 scores", size=14)
    plt.xlabel("epochs", size=14)
    plt.ylabel("F1 score", size=14)
    plt.tick_params(axis="both", size=14)
    plt.grid(alpha=0.8, linestyle=":")
    plt.savefig("f1_curves.png")


def plot_losses(stats):
    """..."""
    # Use plot styling from seaborn.
    sns.set(style="darkgrid")

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(stats["Training Loss"], "b-o", label="Training")
    plt.plot(stats["Valid. Loss"], "g-o", label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()


def plot_training_stats(training_stats):
    """..."""
    # Display floats with two decimal places.
    pd.set_option("precision", 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    # df_stats = df_stats.set_index('fold')
    df_stats.reset_index(inplace=True)
    # df_stats.rename(columns={"index": "fold"}, inplace=True)

    # Display the table.
    # print(df_stats)

    wandb.log({"training_stats": wandb.Table(dataframe=df_stats)})

    return df_stats


def plot_curves_in_grid(data):
    """Plot training and validations losses.

    Expects per run (e.g. per fine-tuned layer) a list of tuples:
    data = {"run": [(train_loss, val_loss), (train_loss, val_loss)]
    """
    fig, axs = plt.subplots(
        int(len(data) / 2) + 1,
        int(len(data) / 2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    i = 0
    j = 0

    for key, scores in data.items():

        train = [x[0] for x in scores]
        val = [x[1] for x in scores]

        axs[i, j].plot(
            range(0, len(scores)), train, color="green", linewidth=3, linestyle="--"
        )

        axs[i, j].plot(
            range(0, len(scores)), val, color="blue", linewidth=3, linestyle="--"
        )
        axs[i, j].set_title(f"Fine-tuning layer {key}")

        if j < 2:
            j += 1

        if j == 2:
            i += 1
            j = 0

    for ax in axs.flat:
        ax.set(xlabel="epochs", ylabel="losses")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # plt.show()
    plt.savefig("train_val_losses.png")


def epoch_time(start_time, end_time):
    """Calculate processing time per epoch."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def categorical_accuracy(preds, y):
    """Return modified predictions and accuracy per batch."""
    # get the index of the max probability (in our case, index == class)
    max_preds = preds.argmax(dim=1, keepdim=True).squeeze(1).to(device)
    correct = max_preds.eq(y).sum().to(device)
    # divide the number of correct predictions by the number of all predictions
    acc = torch.true_divide(correct, y.shape[0])
    return acc


def binary_accuracy(preds, y, report=False):
    """Return modified predictions and accuracy per batch.

    For inference, we have to apply sigmoid ourselves
    output_prob = torch.sigmoid(output) # calculate probabilities
    pred = output_prob > 0.5
    """
    sigmoids = torch.sigmoid(preds)
    # print(f"sigmoids: {sigmoids}")
    # round predictions to the closest integer,
    # equal to preds = output_prob > 0.5
    rounded_preds = torch.round(sigmoids)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / y.shape[0]

    return acc


def prec_rec_f1_per_class(preds, y, scores, data_type="adr"):
    """Calculate metrics globally.

    Counts the total true positives, false negatives and false positives.
    The calculation depends on the type of data, that is, the classification
    output.
    """
    y = y.cpu()
    preds = preds.cpu()
    if data_type == "sentiment":
        # for sentiment, we take the argmax of every prediction, since the
        # index corresponds to the class "name"
        y_pred = preds.argmax(dim=1, keepdim=True).squeeze(1)

    if data_type == "adr":

        sigmoids = torch.sigmoid(preds)
        # equal to preds = output_prob > 0.5
        y_pred = [int(x) for x in torch.round(sigmoids).detach().cpu().numpy()]
        y = [int(x) for x in y]

    # y_pred = [int(x) for x in preds.detach().cpu().numpy()]

    report = classification_report(y_true=y, y_pred=y_pred, output_dict=True)

    for class_name, metrics in report.items():
        if class_name == "accuracy":
            scores[class_name].append(metrics)
        else:
            for metric, result in metrics.items():
                scores[class_name][metric].append(result)

    return scores


def prec_rec_f1_macro(preds, y):
    """Calculate metrics for each label, and find their unweighted mean.

    This does not take label imbalance into account.
    """
    # round predictions to the closest integer
    print(precision_recall_fscore_support(y_pred=preds, y_true=y, average="macro"))


def prec_rec_f1_micro(preds, y):
    """..."""
    # round predictions to the closest integer
    print(precision_recall_fscore_support(y_pred=preds, y_true=y, average="micro"))


def average_scores(scores, sentiment=False):
    """..."""
    averaged_scores = {
        "0": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
        "1": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
        "accuracy": 0,
        "macro avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
        "weighted avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
    }

    if sentiment:
        averaged_scores["2"] = {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0,
        }

    for class_name, metrics in scores.items():
        if class_name == "accuracy":

            averaged_scores[class_name] = np.mean(metrics)
        else:
            for metric, result in metrics.items():
                averaged_scores[class_name][metric] = np.mean(result)

    # for class_, v in averaged_scores.items():
    #     if class_ in ("1", "2", "3"):
    #         print(f"class {class_}: F1: {v['f1-score']}")
    # print(json.dumps(averaged_scores, indent=2))
    # print("\n")
    return averaged_scores


def flatten_dict(results, mode):
    """Flatten a results dictionary."""
    flattened = {}
    for key, val_1 in results.items():
        key_1 = "_".join(key.split())

        if isinstance(val_1, dict):
            for key_2, val_2 in val_1.items():
                new_key = f"{mode}_{key_1}_{key_2}"
                flattened[new_key] = val_2

        else:
            new_key = f"{mode}_{key_1}"
            flattened[new_key] = val_1

    return flattened


def write_results_to_file(results_file, results_senti, results_adr, config):
    """..."""
    flattened_sentiment = flatten_dict(results_senti, "sentiment")
    flattened_adr = flatten_dict(results_adr, "adr")

    config.update(flattened_sentiment)
    config.update(flattened_adr)

    dt = datetime.today()
    config["date"] = dt.strftime("%y_%m_%d")
    config["time"] = dt.strftime("%H:%M")

    with open(results_file, "r") as file_handle:
        dictreader = csv.DictReader(file_handle, delimiter=",")
        fieldnames = dictreader.fieldnames

    with open(results_file, "a", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle, fieldnames=fieldnames, restval="-", extrasaction="ignore"
        )
        writer.writerow(config)

    print(f"Wrote results to file '{results_file}'.\n")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """Dein the earl stopping procedure.

        Args:
            patience (int): How long to wait after last time validation
                            loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
                            improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify
                           as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """..."""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of " f"{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                "Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                "Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
