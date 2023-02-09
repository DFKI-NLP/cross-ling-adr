"""..."""
import colorama

# import data
import gc

# import json
import logging
import numpy as np

# import os
import pandas as pd
import random

# import sys
import time
import torch
import wandb

from collections import namedtuple

from datetime import datetime

# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedShuffleSplit

# from transformers import AdamW
# from transformers import BertForSequenceClassification
# from transformers import XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from utils import training_utils as train_utils
from utils import utils


colorama.init()


# wandb.init(project="final_binary_classification", entity="lraithel")
# config = wandb.config


GREEN = colorama.Fore.GREEN
MAGENTA = colorama.Fore.MAGENTA
RED = colorama.Fore.RED
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET

DATE = datetime.now().strftime("%d_%m_%Y_%H_%M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Scores = namedtuple("scores", ["y_true", "y_score", "y_pred", "metrics"])


def train_model(
    model,
    train_dataloader,
    validation_dataloader,
    epochs,
    optimizer,
    patience=5,
    fold=0,
    seed=42,
):
    """..."""
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/
    # 5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    # Set the seed value all over the place to make this reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Total number of training steps is [#batches] x [#epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        # Default value in run_glue.py
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    early_stopping = train_utils.EarlyStopping(patience=patience)
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    wandb.watch(model)
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        logging.info(
            f"{GREEN}\n======== Epoch {epoch_i + 1} / {epochs} "
            f"========\nTraining ...{RESET}"
        )

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = train_utils.format_time(time.time() - t0)

                # Report progress.
                logging.info(
                    f"{GREEN} Batch {step} of {len(train_dataloader)}" f".{RESET}"
                )
                logging.info(f" Elapsed: {elapsed}.{RESET}")

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU
            # using the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels

            # print(f"labels: {batch[2]}")
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before
            # performing a backward pass. PyTorch doesn't do this automatically
            # because accumulating the gradients is "convenient while training
            # RNNs".(source: https://stackoverflow.com/questions/48001598/
            # why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this train batch).

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we
            # can calculate the average loss at the end. `loss` is a Tensor
            # containing a single value; the `.item()` function just returns
            # the Python value from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            gc.collect()
            torch.cuda.empty_cache()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = train_utils.format_time(time.time() - t0)

        logging.info(f" Avrg training loss: {avg_train_loss:.3g}")
        logging.info(f" Training epoch took: {training_time}")

        wandb.log({f"fold{fold}_avg_train_loss": avg_train_loss})

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance
        # on validation set.

        logging.info(f"{GREEN} Running Validation...{RESET}")

        t0 = time.time()

        # Put the model in evaluation mode -- dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Tracking variables
        predictions = []
        y_true = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU
            # using the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph
            # during the forward pass, since this is only needed for backprop
            # (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.

                # Get the "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like
                # the softmax.
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                loss = outputs.loss
                logits = outputs.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += train_utils.flat_accuracy(logits, label_ids)

            label_ids_as_list = label_ids.tolist()

            # Store predictions and true labels
            predictions_per_batch = list(np.argmax(logits, axis=1))
            predictions += predictions_per_batch

            y_true += label_ids_as_list

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logging.info(f" Accuracy: {avg_val_accuracy:.2f}")

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = train_utils.format_time(time.time() - t0)

        logging.info(f" Validation Loss: {avg_val_loss:.2f}")
        logging.info(f" Validation took: {validation_time:}")

        # Record all statistics from this epoch.
        training_stats.append(
            {
                "fold": fold,
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

        wandb.log(
            {
                f"fold{fold}_val_accuracy": avg_val_accuracy,
                f"fold{fold}_avg_val_loss": avg_val_loss,
                # to make wandb happy
                "avg_val_loss": avg_val_loss,
            }
        )

        # check early stopping criteria
        early_stopping(val_loss=avg_val_loss, model=model)

        if early_stopping.early_stop:
            logging.info(
                f"{GREEN} Fold {fold}: Early stopping after epoch "
                f"{epoch_i + 1}.\n{RESET}"
            )

            wandb.log({f"fold_{fold + 1}": {"early_stopping_after_epoch": epoch_i + 1}})

            break

    logging.info(f"{GREEN} Training complete!\n{RESET}")

    # after the training is finished, print out the training statistics
    stats = train_utils.plot_training_stats(training_stats)
    # plot_losses(stats=stats)

    total_train_time = train_utils.format_time(time.time() - total_t0)
    logging.info(f"{GREEN} Total training took {total_train_time:}{RESET}")

    results = train_utils.get_results(
        y_true=y_true, y_pred=predictions, print_results=False
    )

    macro_F1 = results["macro avg"]["f1-score"]

    return model, avg_val_loss, avg_train_loss, macro_F1
