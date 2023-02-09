"""..."""
import argparse
import colorama
import data
import gc
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import wandb

from collections import namedtuple

from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit

from transformers import AdamW
from transformers import BertForSequenceClassification
from transformers import XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from utils import training_utils as train_utils
from utils import utils

from evaluate import evaluate_on_testset
from trainer import train_model

colorama.init()


wandb.init(project="final_binary_classification", entity="lraithel")
# config = wandb.config


GREEN = colorama.Fore.GREEN
MAGENTA = colorama.Fore.MAGENTA
RED = colorama.Fore.RED
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET

DATE = datetime.now().strftime("%d_%m_%Y_%H_%M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Scores = namedtuple("scores", ["y_true", "y_score", "y_pred", "metrics"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("train_dev_data",
    #                     help=("Path to train/dev data (or to data that needs "
    #                           "to be split)."))

    # parser.add_argument("test_data", default=None, help="Path to test data.")
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

    cross_val = config["cross_val"]
    data_type = config["data_type"]
    epochs = config["epochs"]
    debug = config["debug"]
    min_length = config["min_length"]
    model_name = config["model_name"]
    model_path = config["model_path"]
    patience = config["patience"]
    test_data = config["test_data"]
    train_dev_data = config["train_dev_data"]

    if config["debug"]:
        logging.info(f"{RED} Running in debug mode.{RESET}")

        epochs = 2
        batch_size = 4
        min_length = 3
        max_length = 50
        train_sampler = "weighted"
        # train_dev_data = "data/cadec_segura_smm4h20_traindev_balanced.jsonl"
        # test_data = "data/cadec_segura_smm4h20_testset_balanced.jsonl"
        learning_rate = 2e-5
        model_path = "fine_tuned/model_weights_08_11_2021_16_35.pth"
        model_name = "xlmroberta"
        up_down_sample_data = True
        threshold = 0.5
        t5_model = "t5"
        seed = 42
        test_size = 0.2
        cross_val = 2
        train_dev_data = "data/old/forum_data/combined/TEST_traindevset_combined.jsonl"
        test_data = "data/old/forum_data/combined/TEST_testset_combined.jsonl"

    else:
        wandb.config.cross_val = cross_val
        wandb.config.epochs = epochs
        wandb.config.min_length = min_length
        wandb.config.model_name = model_name
        wandb.config.model_path = model_path
        wandb.config.patience = patience
        wandb.config.test_data = test_data
        wandb.config.train_dev_data = train_dev_data

        sweep_config = wandb.config

        batch_size = sweep_config["batch_size"]
        max_length = sweep_config["max_length"]
        learning_rate = sweep_config["learning_rate"]
        train_sampler = sweep_config["train_sampler"]
        threshold = sweep_config["sampling_threshold"]
        t5_model = sweep_config["t5_model"]
        seed = sweep_config["seed"]
        up_down_sample_data = sweep_config["augment_data"]
        logging.info(f"{GREEN} Up/Downsampling method {up_down_sample_data}\n{RESET}")

        test_size = 0.1

    logging.info(f"{GREEN} Training with {model_name}\n{RESET}")

    # get training data, labels, and transformed labels for stratification in
    # CV
    docs, labels, _, trans_labels = data.prepare_data(
        train_dev_data, min_num_tokens=min_length, max_num_tokens=max_length
    )

    # take only the texts from the dictionary
    sentences = [doc["text"] for doc in docs]
    languages = [doc["language"] for doc in docs]
    original_sentences = [doc["original_sentence"] for doc in docs]

    if len(sentences) > 1000:
        logging.debug(
            f"Example sentence:\n{sentences[723]}\nLabel: " f"{labels[723]}\n"
        )

    input_ids, attention_masks, labels = data.tokenize(
        model_name=model_name, sentences=sentences, labels=labels, max_length=max_length
    )

    global_val_loss = 0
    global_train_loss = 0
    macro_F1 = 0
    global_macro_F1 = 0

    if cross_val:
        lowest_loss = float("inf")
        best_model = None

        # keep the split fixed for all experiments, only vary the seed for the models
        sss = StratifiedShuffleSplit(
            n_splits=cross_val, test_size=test_size, random_state=42
        )

        folds = sss.split(sentences, trans_labels)

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

            # only upsample the training data
            if up_down_sample_data:
                # upsample the training data
                (
                    upsampled_input_ids,
                    upsampled_attention_masks,
                    upsampled_labels,
                ), keep_indices = data.up_down_sample_data(
                    sentences=original_sentences,
                    labels=labels,
                    indices=train_index,
                    languages=languages,
                    model_name=model_name,
                    max_length=max_length,
                    min_length=min_length,
                    minority_threshold=threshold,
                    t5_model=t5_model,
                )

                if keep_indices:
                    # keep only the determined indices
                    input_ids_train = input_ids_train[keep_indices]
                    attention_masks_train = attention_masks_train[keep_indices]
                    labels_train = labels_train[keep_indices]

                if upsampled_input_ids != []:
                    # add upsampled data to original train split
                    input_ids_train = torch.cat(
                        [input_ids_train, upsampled_input_ids], dim=0
                    )

                    attention_masks_train = torch.cat(
                        [attention_masks_train, upsampled_attention_masks], dim=0
                    )

                    labels_train = torch.cat([labels_train, upsampled_labels], dim=0)

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

    # no cross validation
    else:
        train_dataset, val_dataset = data.build_dataset(
            input_ids=input_ids, attention_masks=attention_masks, labels=labels
        )

        train_loader, val_loader = data.create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )

        # Loading fine-tuned model
        logging.info(f"{GREEN} Loading fine-tuned model from {model_path}{RESET}")

        model = train_utils.load_fine_tuned_model(
            model_id=model_path, model_name=model_name
        )
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

        best_model, val_loss = train_model(
            model=model,
            train_dataloader=train_loader,
            validation_dataloader=val_loader,
            epochs=epochs,
            optimizer=optimizer,
        )
    # ----------------------------------------------------------------------- #
    # save the best model with a model identifier containing the date
    new_model_path = "/".join(model_path.split("/")[:-1])
    new_model_id = os.path.join(new_model_path, f"model_weights_re_finetuned{DATE}.pth")
    wandb.log({"model_id_new": new_model_id})

    logging.info(f"{GREEN} Model ID: {new_model_id}{RESET}")
    train_utils.save_fine_tuned_model(best_model, model_id=new_model_id)

    # get test data
    (
        test_input_ids,
        test_attention_masks,
        test_labels,
        langs_test,
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
    # run model on test data
    evaluate_on_testset(
        model=best_model,
        prediction_dataloader=test_dataloader,
        num_sentences=num_test_sentences,
        model_name=model_name,
        languages=langs_test,
    )
