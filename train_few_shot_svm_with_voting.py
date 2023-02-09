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

import pandas as pd

from datetime import datetime
from operator import itemgetter
from utils import training_utils as train_utils

from utils.colors import GREEN
from utils.colors import RED
from utils.colors import RESET
from utils.evaluate import calculate_voting_winners
from utils.evaluate import evaluate_on_testset
from utils.evaluate import evaluate_on_testset_for_votings
from utils.few_shot_sampler import FewShotSampler
from utils.trainer import train_model

from baselines import run_svm_on_embeddings_sweep


colorama.init()


wandb.init(project="final_binary_classification", entity="lraithel")

DATE = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
logging.basicConfig(level=logging.INFO)


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
    # epochs = config["epochs"]
    # debug = config["debug"]
    min_length = config["min_length"]
    max_length = config["max_length"]
    model_name = config["model_name"]
    # model_paths = config["model_paths"]
    # patience = config["patience"]
    test_data = config["test_data"]
    train_dev_data = config["train_dev_data"]
    source_traindev_data = config["source_train_dev_data"]
    # batch_size = config["batch_size"]
    # max_length = config["max_length"]
    # learning_rate = config["learning_rate"]
    # train_sampler = config["train_sampler"]
    # up_down_sample_data = config["augment_data"]
    # threshold = config["sampling_threshold"]
    # t5_model = config["t5_model"]
    # pos_max = config.get("pos_max", 0)

    # wandb.config.epochs = epochs
    wandb.config.min_length = min_length
    wandb.config.model_name = model_name
    # wandb.config.patience = patience
    # wandb.config.model_paths = model_paths
    wandb.config.test_data = test_data

    wandb.config.train_dev_data = train_dev_data
    wandb.config.source_traindev_data = source_traindev_data
    # wandb.config.batch_size = batch_size
    wandb.config.max_length = max_length

    sweep_config = wandb.config

    additional_negative_samples = sweep_config.get("additional_neg", 0)
    additional_samples_source = sweep_config.get("additional_source", 0)
    num_shots = sweep_config["shots"]
    mode = sweep_config["mode"]
    seed = sweep_config["seed"]

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
            "/home/lisa/projects/adr_classification/fine_tuned/dummy_model.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/dummy_model.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/dummy_model.pth",
        ]
        num_shots = 4
        # mode = "add_neg"
        if batch_size > num_shots:
            batch_size = num_shots

    logging.info(f"{GREEN} Training with {model_name}\n{RESET}")

    print("\nPreparing test data ...")
    (docs_test, labels_test, lang_test, _) = data.prepare_data(
        test_data, min_num_tokens=min_length, max_num_tokens=max_length
    )

    # create the results dict
    all_predictions = {}

    for i in range(0, len(docs_test)):
        all_predictions[i] = []

    # prepare data for the train and dev dataloaders
    docs, labels, _, _ = data.prepare_data(
        train_dev_data, min_num_tokens=min_length, max_num_tokens=max_length
    )

    # make sure to keep the seed for the data splits
    sampler = FewShotSampler(num_shots=num_shots, random_state=seed, shuffle=True)

    if mode == "per_class":

        logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

        train_index, _ = sampler.sample_per_class(
            list_of_sentences=docs, list_of_labels=labels
        )

    elif mode == "as_distribution":
        logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

        train_index, _ = sampler.sample_according_to_distribution(
            list_of_sentences=docs, list_of_labels=labels, pos_max=pos_max
        )

    elif mode == "random":
        logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

        train_index, _ = sampler.sample_randomly(
            list_of_sentences=docs, list_of_labels=labels
        )

    elif mode == "add_neg":
        logging.info(f"{GREEN} Running in '{mode}' mode.\n{RESET}")

        train_index, _ = sampler.sample_and_add_negatives(
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

        # source_sentences = [doc["text"] for doc in source_docs]

        # (source_input_ids, source_attention_masks, source_labels) = data.tokenize(
        #     model_name=model_name,
        #     sentences=source_sentences,
        #     labels=source_labels,
        #     max_length=max_length,
        # )

        train_index, _ = sampler.sample_and_add_negatives(
            list_of_sentences=docs,
            list_of_labels=labels,
            num_added=additional_negative_samples,
        )
    else:

        logging.warning(
            f"{RED} No mode for splitting given. Using per class few shot split.{RESET}"
        )

    # input_ids_train = input_ids[train_index]
    # attention_masks_train = attention_masks[train_index]
    # labels_train = labels[train_index]
    # X_train = docs[train_index]
    # y_train = labels[train_index]

    X_train = list(itemgetter(*train_index)(docs))
    y_train = list(itemgetter(*train_index)(labels))

    # input_ids_val = input_ids[val_index]
    # attention_masks_val = attention_masks[val_index]
    # labels_val = labels[val_index]

    # X_val = docs[val_index]
    # y_val = labels[val_index]

    if mode == "add_neg_plus_source":

        # get random indices from the source language data (do not sort!)
        source_idx = random.sample(
            range(0, len(source_docs) - 1), additional_samples_source * 2
        )

        # print(f"source lang idx: {source_idx}")

        # divide the indices in half to get one for train and one for val
        source_idx_train = source_idx[:additional_samples_source]
        # source_idx_val = source_idx[additional_samples_source:]

        # get the source input data for train and val
        # source_input_ids_train = source_input_ids[source_idx_train]
        # source_attention_masks_train = source_attention_masks[source_idx_train]
        # source_labels_train = source_labels[source_idx_train]
        # X_source_train = source_docs[source_idx_train]
        # y_source_train = source_labels[source_idx_train]

        X_source_train = list(itemgetter(*source_idx_train)(source_docs))
        y_source_train = list(itemgetter(*source_idx_train)(source_labels))

        wandb.log({"additional_source_train": y_source_train})

        logging.info(
            f"{GREEN} Each split will contain additional {len(X_source_train)} random samples from the source language data!\n{RESET}"
        )

        # source_input_ids_val = source_input_ids[source_idx_val]
        # source_attention_masks_val = source_attention_masks[source_idx_val]
        # source_labels_val = source_labels[source_idx_val]
        # X_source_val = source_docs[source_idx_val]
        # y_source_val = source_labels[source_idx_val]

        # wandb.log({"additional_source_val": y_source_val})

        # add the source data to the shot train split
        # input_ids_train = torch.cat(
        #     [input_ids_train, source_input_ids_train], dim=0
        # )
        # attention_masks_train = torch.cat(
        #     [attention_masks_train, source_attention_masks_train], dim=0
        # )

        # labels_train = torch.cat([labels_train, source_labels_train], dim=0)

        X_train = X_train + X_source_train
        y_train = y_train + y_source_train

        assert len(X_train) == len(y_train)

        # add the source data to the shot val split
        # input_ids_val = torch.cat([input_ids_val, source_input_ids_val], dim=0)

        # attention_masks_val = torch.cat(
        #     [attention_masks_val, source_attention_masks_val], dim=0
        # )

        # labels_val = torch.cat([labels_val, source_labels_val], dim=0)

        # X_val = X_val.extend(X_source_val)
        # y_val = y_val.extend(y_source_val)

        logging.info(f"{GREEN} Adding source data done.{RESET}")

    # train_loader = data.get_data_loader(
    #     input_ids_train,
    #     attention_masks_train,
    #     labels_train,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     sampler=train_sampler,
    #     model_name=model_name,
    # )

    # # we do not need sample weights in the validation data
    # val_loader = data.get_data_loader(
    #     input_ids_val,
    #     attention_masks_val,
    #     labels_val,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     sampler=False,
    # )

    results, y_pred = run_svm_on_embeddings_sweep(
        train_data=X_train,
        train_labels=y_train,
        test_data=docs_test,
        test_labels=labels_test,
        aligned_embeds=True,
        seed=seed,
    )

    wandb.log({"true_labels": labels_test})

    wandb.log({"final_f1_class_1": results["1"]["f1-score"]})

    wandb.log(results)

    # plot_results(results_list=[scores], descriptions=[f"TODO"])
    test_sentences = [doc["text"] for doc in docs_test]

    results_df = pd.DataFrame(
        list(zip(test_sentences, lang_test, y_pred, labels_test)),
        columns=["text", "language", "predicted", "gold"],
    )
    wandb.log({"final_predictions": wandb.Table(dataframe=results_df)})

    # add a wandb table containing scores per language
    train_utils.get_results_per_language(results_df)

    # calculate_voting_winners(
    #     all_predictions=all_predictions,
    #     y_true=test_labels,
    #     decoded_sentences=test_sentences,
    #     languages=langs_test,
    # )
