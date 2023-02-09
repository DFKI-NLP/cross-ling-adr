"""..."""
import colorama
import logging
import numpy as np
import pandas as pd
import time
import torch
import wandb

from collections import namedtuple
from datetime import datetime
from sklearn.metrics import precision_recall_curve

import data

from utils import training_utils as train_utils
from utils import visualizations


GREEN = colorama.Fore.GREEN
MAGENTA = colorama.Fore.MAGENTA
RED = colorama.Fore.RED
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET

DATE = datetime.now().strftime("%d_%m_%Y_%H_%M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Scores = namedtuple("scores", ["y_true", "y_score", "y_pred", "metrics"])


def plot_results(results_list, descriptions):
    """..."""
    visualizations.plot_precision_recall_curve(results_list, descriptions, "baselines")
    visualizations.plot_roc_curve(results_list, descriptions, "baselines")


def incorporate_mp(sentences, votes):
    """..."""
    keywords = set()
    with open("meno_pause_keywords.txt", "r") as read_handle:
        for line in read_handle:
            keywords.add(line.strip().lower())

    new_votes = []
    num_labels_switched = 0
    # if there is a positive prediction and the document includes a keyword,
    # switch the label
    for sentence, vote in zip(sentences, votes):
        tokens = sentence.lower().split()
        if vote == 0:
            new_label = vote
        if vote == 1:
            new_label = vote
            for kw in keywords:
                for token in tokens:
                    if kw == token:
                        new_label = 0
                        num_labels_switched += 1

        new_votes.append(new_label)

    assert len(new_votes) == len(votes) == len(sentences)

    logging.info(
        f"{GREEN} Switched {num_labels_switched} labels when checking for MP keywords.{RESET}"
    )

    return new_votes


def incorporate_drugs(sentences, votes):
    """..."""
    keywords = set()
    with open("german_medication_list.txt", "r") as read_handle:
        for line in read_handle:
            phrase = line.strip().split()
            keywords.add(phrase[0].lower())

    num_labels_kept = 0
    new_votes = []
    keywords_found = set()
    # if there is a positive prediction and the document does not include
    # any keyword, switch the label to negative
    for sentence, vote in zip(sentences, votes):
        tokens = sentence.lower().split()
        if vote == 0:
            new_label = vote
        if vote == 1:
            new_label = 0
            for kw in keywords:
                if kw in tokens:
                    keywords_found.add(kw)
                    new_label = 1
                    num_labels_kept += 1

        new_votes.append(new_label)

    assert len(new_votes) == len(votes) == len(sentences)

    wandb.log({"drugs_found": list(keywords_found)})

    logging.info(
        f"{GREEN} Kept {num_labels_kept} labels when checking for medication keywords.{RESET}"
    )

    return new_votes


def calculate_voting_winners(
    all_predictions, y_true, decoded_sentences, languages, post_processing=False):
    """Get majority vote of all 10 votes for every data sample."""
    final_votes = []
    for sample, predictions in all_predictions.items():

        counts = np.bincount(predictions)
        final_vote = np.argmax(counts)
        final_votes.append(final_vote)

    wandb.log({"true_labels": y_true})
    wandb.log({"final_votes": final_votes})

    results_df = pd.DataFrame(
        list(zip(decoded_sentences, languages, final_votes, y_true)),
        columns=["text", "language", "predicted", "gold"],
    )
    # plot_results(results_list=[scores], descriptions=[f"TODO"])

    if post_processing:
        logging.info(f"{GREEN} Post-processing votes.{RESET}")

        # get results when switching labels of positives that contain meno pause words
        predicted_mp = incorporate_mp(decoded_sentences, final_votes)
        predicted_drugs = incorporate_drugs(decoded_sentences, final_votes)

        # get results when switching labels of positives that contain drug names
        results_df = pd.DataFrame(
            list(
                zip(
                    decoded_sentences,
                    languages,
                    final_votes,
                    predicted_drugs,
                    predicted_mp,
                    y_true,
                )
            ),
            columns=[
                "text",
                "language",
                "predicted",
                "predicted_drugs",
                "predicted_mp",
                "gold",
            ],
        )

        results_mp = train_utils.get_results(
            y_true=y_true, y_pred=predicted_mp, print_results=False
        )

        results_drugs = train_utils.get_results(
            y_true=y_true, y_pred=predicted_drugs, print_results=False
        )

        wandb.log({"final_f1_class_1_mp": results_mp["1"]["f1-score"]})
        wandb.log({"including_mp": results_mp})
        wandb.log({"final_f1_class_1_drugs": results_drugs["1"]["f1-score"]})
        wandb.log({"including_drugs": results_drugs})

    results = train_utils.get_results(
        y_true=y_true, y_pred=final_votes, print_results=False
    )

    # add a wandb table containing scores per language
    train_utils.get_results_per_language(results_df)

    wandb.log({"final_f1_class_1": results["1"]["f1-score"]})

    wandb.log(results)
    wandb.log({"final_predictions": wandb.Table(dataframe=results_df)})


    # return only the standard results (without drug/mp lookup)
    return results, results_df


def evaluate_on_testset_for_votings(
    model, prediction_dataloader, num_sentences, model_name, model_number
):
    """..."""
    # Prediction on test set

    logging.info(
        f"{GREEN} Predicting labels for {num_sentences} test " f"sentences ...{RESET}"
    )

    # Put model in evaluation mode
    model.eval()
    model.to(device)

    # Tracking variables
    predictions = []
    true_labels = []
    y_true = []

    decoded_sentences = []
    probas = []
    collected_logits = []
    t0 = time.time()

    # Predict
    for step, batch in enumerate(prediction_dataloader):
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = train_utils.format_time(time.time() - t0)

            # Report progress.
            logging.info(
                f"{GREEN} Batch {step} of {len(prediction_dataloader)}" f".{RESET}"
            )
            logging.info(f" Elapsed: {elapsed}.{RESET}")

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory
        # and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs.logits

        # get probabilities from logits
        probas += torch.softmax(logits, dim=1).tolist()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        collected_logits += logits.tolist()
        label_ids = b_labels.to("cpu").numpy()
        label_ids_as_list = label_ids.tolist()

        # takes pretty long
        # decoded_sentences += data.decode_ids(
        #     model_name=model_name, id_sequences=b_input_ids
        # )
        # Store predictions and true labels
        predictions_per_batch = list(np.argmax(logits, axis=1))
        predictions += predictions_per_batch
        true_labels.append(label_ids)
        y_true += label_ids_as_list

    logging.info(f"{GREEN} DONE.{RESET}")

    wandb.log({f"predictions_model_{model_number}": predictions})
    wandb.log({"probabilities": probas})
    wandb.log({"logits": collected_logits})

    # get only the probabilities of the positive class
    probas_pos = [x[1] for x in probas]
    # determine the best threshold
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=probas_pos, pos_label=1
    )

    votes = {}
    for i, prediction in enumerate(predictions):
        votes[i] = prediction

    return votes, y_true


def evaluate_on_testset(
    model, prediction_dataloader, num_sentences, model_name, languages
):
    """..."""
    # Prediction on test set

    logging.info(
        f"{GREEN} Predicting labels for {num_sentences} test " f"sentences ...{RESET}"
    )

    # Put model in evaluation mode
    model.eval()
    model.to(device)

    # Tracking variables
    predictions = []
    true_labels = []
    y_true = []

    decoded_sentences = []
    probas = []
    t0 = time.time()

    # Predict
    for step, batch in enumerate(prediction_dataloader):
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = train_utils.format_time(time.time() - t0)

            # Report progress.
            logging.info(
                f"{GREEN} Batch {step} of {len(prediction_dataloader)}" f".{RESET}"
            )
            logging.info(f" Elapsed: {elapsed}.{RESET}")

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory
        # and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs.logits

        # get probabilities from logits
        probas += torch.softmax(logits, dim=1).tolist()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        label_ids_as_list = label_ids.tolist()

        decoded_sentences += data.decode_ids(
            model_name=model_name, id_sequences=b_input_ids
        )
        # Store predictions and true labels
        predictions_per_batch = list(np.argmax(logits, axis=1))  # .flatten()
        predictions += predictions_per_batch
        # TODO: fix this -- F1 score calculation and prec/rec curve need
        # different formats of the same thing
        true_labels.append(label_ids)
        y_true += label_ids_as_list

    logging.info(f"{GREEN} DONE.{RESET}")

    probas = np.asarray(probas)

    results = train_utils.get_results(y_true=y_true, y_pred=predictions)

    scores = Scores(y_true=y_true, y_score=probas, y_pred=predictions, metrics=results)
    # wandb results
    wandb.log({"model_name": model_name})
    wandb.log(results)

    plot_results(results_list=[scores], descriptions=[f"TODO"])

    results_df = pd.DataFrame(
        list(zip(decoded_sentences, languages, predictions, y_true)),
        columns=["text", "language", "predicted", "gold"],
    )
    wandb.log({"final_predictions": wandb.Table(dataframe=results_df)})

    # add a wandb table containing scores per language
    train_utils.get_results_per_language(results_df)

    # file_name = f"predictions_{model_name}_{DATE}.csv"
    # train_utils.write_predictions_to_file(sentences=decoded_sentences,
    #                                       y_pred=predictions,
    #                                       y_true=y_true,
    #                                       file_name=file_name)
