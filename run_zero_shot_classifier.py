"""..."""
import argparse
import colorama
import data
import json
import torch
import wandb

from datetime import datetime

from evaluate import evaluate_on_testset
from utils import training_utils as train_utils


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

    batch_size = config["batch_size"]
    max_length = config["max_length"]
    min_length = config["min_length"]
    model_name = config["model_name"]
    model_path = config["model_path"]
    test_data = config["test_data"]

    wandb.config.batch_size = batch_size
    wandb.config.max_length = max_length
    wandb.config.min_length = min_length
    wandb.config.model_name = model_name
    wandb.config.model_path = model_path
    wandb.config.test_data = test_data

    sweep_config = wandb.config

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
    # print("Reloaded model results:\n")
    model_id = model_path
    loaded_model = train_utils.load_fine_tuned_model(
        model_id=model_id, model_name=model_name
    )

    # run model on test data
    evaluate_on_testset(
        model=loaded_model,
        prediction_dataloader=test_dataloader,
        num_sentences=num_test_sentences,
        model_name=model_name,
        languages=langs_test,
    )
