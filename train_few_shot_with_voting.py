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
    source_traindev_data = config["source_train_dev_data"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]
    post_processing = config.get("post_processing", 0)
    # learning_rate = config["learning_rate"]
    # train_sampler = config["train_sampler"]
    up_down_sample_data = config["augment_data"]
    threshold = config["sampling_threshold"]
    t5_model = config["t5_model"]
    pos_max = config.get("pos_max", 0)

    wandb.config.epochs = epochs
    wandb.config.min_length = min_length
    wandb.config.model_name = model_name
    wandb.config.patience = patience
    wandb.config.model_paths = model_paths
    wandb.config.test_data = test_data

    wandb.config.train_dev_data = train_dev_data
    wandb.config.source_traindev_data = source_traindev_data
    wandb.config.batch_size = batch_size
    wandb.config.max_length = max_length

    wandb.config.up_down_sample_data = up_down_sample_data
    wandb.config.threshold = threshold
    wandb.config.t5_model = t5_model

    wandb.config.pos_max = pos_max

    sweep_config = wandb.config

    # in some modes we do not need negative samples (per_class) or
    # source samples (per_class, add_neg)
    additional_negative_samples = sweep_config.get("additional_neg", 0)
    additional_samples_source = sweep_config.get("additional_source", 0)
    num_shots = sweep_config["shots"]
    mode = sweep_config["mode"]
    seed = sweep_config["seed"]
    train_sampler = sweep_config["train_sampler"]

    if batch_size > num_shots:
        batch_size = num_shots
        # wandb.config.batch_size = batch_size

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
        source_traindev_data = (
            "data/old/forum_data/combined/TEST_traindevset_combined.jsonl"
        )
        model_paths = [
            "/home/lisa/projects/adr_classification/fine_tuned/model_weights_14_12_2021_12_01_28.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/model_weights_14_12_2021_12_01_28.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/model_weights_14_12_2021_12_01_28.pth",
        ]
        model_name = "xlmroberta"
        num_shots = 4
        additional_negative_samples = 5
        additional_samples_source = 5
        # mode = "add_neg"
        if batch_size > num_shots:
            batch_size = num_shots

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

    # chosen via sweep: seed, mode, num_shots
    for model_number, model_path in enumerate(model_paths):

        print(f"model number {model_number}: {model_path}")

        if mode == "add_neg" or mode == "add_neg_plus_source":
            if num_shots == 70:
                num_shots == 35
                wandb.config.num_shots = shots
        # prepare the train/dev based on the current seed and current mode
        # "train" each model on the sampled train/dev
        # apply the re-finetuned model on the test set
        # save each vote for each example
        # finally, choose the majority vote for each example
        # compare the majority vote with the gold test data

        # prepare data for the train and dev dataloaders
        docs, labels, _, _ = data.prepare_data(
            train_dev_data, min_num_tokens=min_length, max_num_tokens=max_length
        )

        # take only the texts from the dictionary
        sentences = [doc["text"] for doc in docs]

        (input_ids, attention_masks, labels) = data.tokenize(
            model_name=model_name,
            sentences=sentences,
            labels=labels,
            max_length=max_length,
        )

        # make sure to keep the seed for the data splits
        sampler = FewShotSampler(num_shots=num_shots, random_state=seed, shuffle=True)

        if mode == "per_class":

            logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

            train_index, val_index = sampler.sample_per_class(
                list_of_sentences=docs, list_of_labels=labels
            )

        elif mode == "as_distribution":
            logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

            train_index, val_index = sampler.sample_according_to_distribution(
                list_of_sentences=docs, list_of_labels=labels, pos_max=pos_max
            )

        elif mode == "random":
            logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

            train_index, val_index = sampler.sample_randomly(
                list_of_sentences=docs, list_of_labels=labels
            )

        elif mode == "add_neg":
            logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

            train_index, val_index = sampler.sample_and_add_negatives(
                list_of_sentences=docs,
                list_of_labels=labels,
                num_added=additional_negative_samples,
            )

        elif mode == "add_neg_plus_source":
            logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

            source_docs, source_labels, _, _ = data.prepare_data(
                source_traindev_data,
                min_num_tokens=min_length,
                max_num_tokens=max_length,
            )

            source_sentences = [doc["text"] for doc in source_docs]

            (source_input_ids, source_attention_masks, source_labels) = data.tokenize(
                model_name=model_name,
                sentences=source_sentences,
                labels=source_labels,
                max_length=max_length,
            )

            train_index, val_index = sampler.sample_and_add_negatives(
                list_of_sentences=docs,
                list_of_labels=labels,
                num_added=additional_negative_samples,
            )
        else:

            logging.warning(
                f"{RED} No mode for splitting given. Using per class few shot split.{RESET}"
            )

        input_ids_train = input_ids[train_index]
        attention_masks_train = attention_masks[train_index]
        labels_train = labels[train_index]

        input_ids_val = input_ids[val_index]
        attention_masks_val = attention_masks[val_index]
        labels_val = labels[val_index]

        if mode == "add_neg_plus_source":

            # get random indices from the source language data (do not sort!)
            source_idx = random.sample(
                range(0, len(source_docs) - 1), additional_samples_source * 2
            )

            # print(f"source lang idx: {source_idx}")

            # divide the indices in half to get one for train and one for val
            source_idx_train = source_idx[:additional_samples_source]
            source_idx_val = source_idx[additional_samples_source:]

            # get the source input data for train and val
            source_input_ids_train = source_input_ids[source_idx_train]
            source_attention_masks_train = source_attention_masks[source_idx_train]
            source_labels_train = source_labels[source_idx_train]

            wandb.log({"additional_source_train": source_labels_train})

            source_input_ids_val = source_input_ids[source_idx_val]
            source_attention_masks_val = source_attention_masks[source_idx_val]
            source_labels_val = source_labels[source_idx_val]

            wandb.log({"additional_source_val": source_labels_val})

            # add the source data to the shot train split
            input_ids_train = torch.cat(
                [input_ids_train, source_input_ids_train], dim=0
            )

            attention_masks_train = torch.cat(
                [attention_masks_train, source_attention_masks_train], dim=0
            )

            labels_train = torch.cat([labels_train, source_labels_train], dim=0)

            # add the source data to the shot val split
            input_ids_val = torch.cat([input_ids_val, source_input_ids_val], dim=0)

            attention_masks_val = torch.cat(
                [attention_masks_val, source_attention_masks_val], dim=0
            )

            labels_val = torch.cat([labels_val, source_labels_val], dim=0)

            logging.info(f"{GREEN} Adding source data done.{RESET}")

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

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)  # lr=2e-5,

        # fine tune model
        (newest_model, avg_val_loss, avg_train_loss, new_macro_F1) = train_model(
            model=model,
            train_dataloader=train_loader,
            validation_dataloader=val_loader,
            epochs=epochs,
            optimizer=optimizer,
            patience=patience,
            fold=model_number,
            seed=seed,
        )

        logging.info(f"{GREEN} new macro F1: {new_macro_F1}{RESET}")

        # global_val_loss += avg_val_loss
        # global_train_loss += avg_train_loss
        # global_macro_F1 += new_macro_F1

        # if avg_val_loss < lowest_loss:
        #     lowest_loss = avg_val_loss
        #     # best_model = newest_model
        # # update the model if the F1 score increases
        # if new_macro_F1 > macro_F1:
        #     macro_F1 = new_macro_F1
        #     best_model = newest_model

        wandb.log(
            {
                f"global_val_loss_{model_number}": avg_val_loss,
                f"global_train_loss_{model_number}": avg_train_loss,
                f"global_macro_F1_{model_number}": new_macro_F1,
                f"model_path_{model_number}": model_path,
            }
        )

        # ----------------------------------------------------------------------- #
        # save the best model with a model identifier containing the date
        new_model_path = "/".join(model_path.split("/")[:-1])
        new_model_id = os.path.join(
            new_model_path,
            f"model_few_shot_{num_shots}_{mode}_{model_name}_{model_number}_seed_{seed}_freeze_{freeze_all}_lr_{learning_rate}_{train_sampler}_neg_{additional_negative_samples}_source_{additional_samples_source}.pth",
        )
        wandb.log({f"model_id_new_{model_name}_{model_number}": new_model_id})

        logging.info(f"{GREEN} Model ID: {new_model_id}{RESET}")
        train_utils.save_fine_tuned_model(newest_model, model_id=new_model_id)

        # run model on test data; return a dict with a prediction per sample:
        # {sample_1: 1, smaple_2: 0, sample_1: 0, ...}
        results_per_doc, y_true, = evaluate_on_testset_for_votings(
            model=newest_model,
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
