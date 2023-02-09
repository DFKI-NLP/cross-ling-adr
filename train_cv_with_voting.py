"""..."""
import argparse
import colorama
import data
import json
import logging
import numpy as np
import os
import random
import sys
import time
import torch
import wandb

from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AdamW

from utils import training_utils as train_utils

from utils.colors import GREEN
from utils.colors import RED
from utils.colors import RESET
from utils.evaluate import calculate_voting_winners
from utils.evaluate import evaluate_on_testset
from utils.evaluate import evaluate_on_testset_for_votings
from utils.few_shot_sampler import FewShotSampler
from utils.trainer import train_model


colorama.init()


wandb.init(project="final_binary_classification", entity="lraithel")

DATE = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
logging.basicConfig(level=logging.INFO)

local_rank = int(os.environ.get("LOCAL_RANK", -1))

if torch.cuda.is_available():
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")
    parser.add_argument(
        "-sp",
        "--save_probas",
        action="store_true",
        help="Save probas and true labels for visualization.",
    )
    parser.add_argument(
        "-debug", "--debug", action="store_true", help="Run with test data."
    )

    args = parser.parse_args()

    with open(args.config, "r") as read_handle:
        config = json.load(read_handle)

    # parameters loaded from config (does not change (anymore))
    epochs = config["epochs"]
    debug = config["debug"]
    min_length = config["min_length"]
    max_length = config["max_length"]
    model_name = config["model_name"]
    model_paths = config["model_paths"]
    patience = config["patience"]
    test_data = config["test_data"]
    train_dev_data = config["train_dev_data"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]
    post_processing = config.get("post_processing", 0)

    wandb.config.epochs = epochs
    wandb.config.min_length = min_length
    wandb.config.model_name = model_name
    wandb.config.patience = patience
    wandb.config.model_paths = model_paths
    wandb.config.test_data = test_data

    wandb.config.train_dev_data = train_dev_data
    wandb.config.batch_size = batch_size
    wandb.config.max_length = max_length

    sweep_config = wandb.config

    # in some modes we do not need negative samples (per_class) or
    # source samples (per_class, add_neg)
    train_sampler = sweep_config["train_sampler"]
    seed = sweep_config["seed"]

    try:
        learning_rate = sweep_config["learning_rate"]
    except KeyError:
        learning_rate = 0.00001056
        wandb.config.learning_rate = learning_rate
    try:
        freeze_all = sweep_config["freeze_all"]
    except KeyError:
        freeze_all = 1
        wandb.config.freeze_all = freeze_all

    # overwrite with test data in debug mode
    if config["debug"]:
        logging.info(f"{RED} Running in debug mode.{RESET}")

        epochs = 2
        batch_size = 4
        min_length = 3
        max_length = 50
        train_dev_data = "data/old/forum_data/combined/TEST_traindevset_combined.jsonl"
        test_data = "data/old/forum_data/combined/TEST_testset_combined.jsonl"
        model_paths = [
            "/home/lisa/projects/adr_classification/fine_tuned/model_weights_14_12_2021_12_01_28.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/model_weights_14_12_2021_12_01_28.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/model_weights_14_12_2021_12_01_28.pth",
        ]
        model_name = "xlmroberta"

    logging.info(f"{GREEN} Training with {model_name}\n{RESET}")

    # get test data (always stays the same)
    (
        test_input_ids,
        test_attention_masks,
        test_labels,
        langs_test,
        test_sentences,
    ) = data.prepare_test_data(
        model_name=model_name,
        test_data_file=test_data,
        min_num_tokens=min_length,
        max_num_tokens=max_length,
    )

    test_dataloader, num_test_sentences = data.get_test_data_loader(
        input_ids=test_input_ids,
        attention_masks=test_attention_masks,
        labels=test_labels,
        batch_size=batch_size,
    )

    # create the results dict
    all_predictions = {}

    for i in range(0, num_test_sentences):
        all_predictions[i] = []

    # prepare data for the train and dev data
    docs, int_labels, languages, trans_labels = data.prepare_data(
        train_dev_data, min_num_tokens=min_length, max_num_tokens=max_length
    )

    # take only the texts from the dictionary
    sentences = [doc["text"] for doc in docs]

    (input_ids, attention_masks, labels) = data.tokenize(
        model_name=model_name,
        sentences=sentences,
        labels=int_labels,
        max_length=max_length,
    )

    test_size = 0.1
    cross_val = 1
    # chosen via sweep: seed, mode, num_shots
    for model_number, model_path in enumerate(model_paths):

        print(f"model number {model_number}: {model_path}")

        global_val_loss = 0
        global_train_loss = 0
        macro_F1 = 0
        global_macro_F1 = 0
        lowest_loss = float("inf")
        best_model = None

        # keep the split fixed for all experiments, only vary the seed for the models
        sss = StratifiedShuffleSplit(
            n_splits=cross_val, test_size=test_size, random_state=seed
        )

        folds = sss.split(sentences, int_labels)

        for fold_num, (train_index, val_index) in enumerate(folds):

            logging.info(
                f"\n{GREEN}======== Training on fold {fold_num + 1} /"
                f" {cross_val} ========\n{RESET}"
            )
            wandb.log({"current_fold": fold_num + 1})

            input_ids_train = input_ids[train_index]
            attention_masks_train = attention_masks[train_index]
            labels_train = labels[train_index]

            input_ids_val = input_ids[val_index]
            attention_masks_val = attention_masks[val_index]
            labels_val = labels[val_index]

            wandb.log({f"fold_{fold_num}": {"train_data_size": len(input_ids_train)}})

            train_loader = data.get_data_loader(
                input_ids_train,
                attention_masks_train,
                labels_train,
                batch_size=batch_size,
                shuffle=True,
                sampler=train_sampler,
                model_name=model_name,
            )

            # we do not need sample weights in the validation data
            val_loader = data.get_data_loader(
                input_ids_val,
                attention_masks_val,
                labels_val,
                batch_size=batch_size,
                shuffle=True,
                sampler=False,
            )

            # Loading fine-tuned model
            logging.info(f"{GREEN} Loading fine-tuned model from {model_path}{RESET}")

            model = train_utils.load_fine_tuned_model(
                model_id=model_path, model_name=model_name
            )

            # freeze all (except the classifier), otherwise fine-tune everything
            if freeze_all:
                # freeze every layer except the classifier
                for name, param in model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False

            model.to(device)

            optimizer = AdamW(
                model.parameters(), lr=learning_rate, eps=1e-8  # lr=2e-5,
            )

            # fine tune model
            (newest_model, avg_val_loss, avg_train_loss, new_macro_F1) = train_model(
                model=model,
                train_dataloader=train_loader,
                validation_dataloader=val_loader,
                epochs=epochs,
                optimizer=optimizer,
                patience=patience,
                fold=fold_num + 1,
                seed=seed,
            )

            logging.info(f"{GREEN} new macro F1: {new_macro_F1}{RESET}")

            global_val_loss += avg_val_loss
            global_train_loss += avg_train_loss
            global_macro_F1 += new_macro_F1

            if avg_val_loss < lowest_loss:
                wandb.log({"fold_with_lowest_loss": fold_num})
                lowest_loss = avg_val_loss
                # best_model = newest_model
            # update the model if the F1 score increases
            if new_macro_F1 > macro_F1:
                wandb.log({"fold_with_highest_F1": fold_num})
                macro_F1 = new_macro_F1
                best_model = newest_model

        wandb.log(
            {
                "global_val_loss": global_val_loss / cross_val,
                "global_train_loss": global_train_loss / cross_val,
                "global_macro_F1": global_macro_F1 / cross_val,
            }
        )

        # ----------------------------------------------------------------------- #
        # save the best model with a model identifier containing the date
        new_model_path = "/".join(model_path.split("/")[:-1])
        new_model_id = os.path.join(
            new_model_path,
            f"model_full_de_data_{model_name}_{model_number}_seed_{seed}_freeze_{freeze_all}_lr_{learning_rate}_{train_sampler}.pth",
        )
        wandb.log({f"model_id_new_{model_name}_{model_number}": new_model_id})

        logging.info(f"{GREEN} Model ID: {new_model_id}{RESET}")
        train_utils.save_fine_tuned_model(best_model, model_id=new_model_id)

        # run model on test data; return a dict with a prediction per sample:
        # {sample_1: 1, smaple_2: 0, sample_1: 0, ...}
        results_per_doc, y_true, = evaluate_on_testset_for_votings(
            model=best_model,
            prediction_dataloader=test_dataloader,
            num_sentences=num_test_sentences,
            model_name=model_name,
            model_number=model_number,
        )

        # {sample_1: [0, 1, 1, 1, 0], sample_2: [0, 0, 0, 1, 0]}
        for sample, prediction in results_per_doc.items():
            all_predictions[sample].append(prediction)

    for sample, predictions in all_predictions.items():
        assert len(predictions) == len(model_paths)

    calculate_voting_winners(
        all_predictions=all_predictions,
        y_true=y_true,
        decoded_sentences=test_sentences,
        languages=langs_test,
        post_processing=post_processing,
    )
