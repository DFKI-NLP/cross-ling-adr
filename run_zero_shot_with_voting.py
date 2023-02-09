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
    debug = config["debug"]
    min_length = config["min_length"]
    model_name = wandb.config["models"]
    if model_name == "xlmroberta":
        model_paths = config["model_paths_roberta"]
    else:
        model_paths = config["model_paths_bioreddit"]

    test_data = config["test_data"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]

    wandb.config.min_length = min_length
    wandb.config.model_name = model_name
    wandb.config.model_paths = model_paths
    wandb.config.test_data = test_data
    wandb.config.batch_size = batch_size
    wandb.config.max_length = max_length

    sweep_config = wandb.config

    # overwrite with test data in debug mode
    if config["debug"]:
        logging.info(f"{RED} Running in debug mode.{RESET}")

        batch_size = 4
        min_length = 3
        max_length = 50
        test_data = "data/old/forum_data/combined/TEST_testset_combined.jsonl"
        model_paths = [
            "/home/lisa/projects/adr_classification/fine_tuned/dummy_model.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/dummy_model.pth",
            "/home/lisa/projects/adr_classification/fine_tuned/dummy_model.pth",
        ]
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

        model = train_utils.load_fine_tuned_model(
            model_id=model_path, model_name=model_name
        )

        model.eval()
        model.to(device)

        wandb.log({f"model_path_{model_number}": model_path})

        # run model on test data; return a dict with a prediction per sample:
        # {sample_1: 1, smaple_2: 0, sample_1: 0, ...}
        results_per_doc, y_true = evaluate_on_testset_for_votings(
            model=model,
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
    )
