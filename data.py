"""Prepare data for use with pytorch."""
import colorama
import json
import logging
import numpy as np
import pandas as pd
import random
import re
import sys
import torch
import wandb

from tqdm import tqdm

# https://pypi.org/project/absum/
# from absum import Augmentor

from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import MT5ForConditionalGeneration
from transformers import MT5Tokenizer
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import XLMRobertaTokenizer

from transformers import pipeline

from utils import augmentor
from utils import training_utils as train_utils
from utils import utils


from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split

from collections import Counter
from collections import defaultdict

from utils.ekphrasis_tokenizer import text_processor

colorama.init()

GREEN = colorama.Fore.GREEN
MAGENTA = colorama.Fore.MAGENTA
RED = colorama.Fore.RED
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer(model_name):
    """Get tokenizer belonging to model."""
    if model_name == "xlmroberta":
        return XLMRobertaTokenizer.from_pretrained(
            "xlm-roberta-base"
        )  # , force_download=True)

    elif model_name == "multibert":
        return BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")

    elif model_name == "character-bert":
        return AutoTokenizer.from_pretrained("helboukkouri/character-bert")

    elif model_name == "bioreddit-bert":
        return AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")

    else:
        logging.warning(f"{RED} Please specify a correct model name.{RESET}")

        assert False


def tokenize(model_name, sentences, labels, max_length):
    """Tokenize, truncate, and pad the sentences.

    Returns input IDs, attention masks and labels as tensors.
    """
    tokenizer = get_tokenizer(model_name)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    for sent in sentences:

        prepro_sent = text_processor.pre_process_doc(sent)
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            prepro_sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            # pad_to_max_length=True,
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from
        # non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def decode_ids(model_name, id_sequences):
    """..."""
    decoded_sentences = []
    tokenizer = get_tokenizer(model_name)

    for id_seq in id_sequences:
        tokens = tokenizer.convert_ids_to_tokens(id_seq, skip_special_tokens=True)
        sentence = tokenizer.convert_tokens_to_string(tokens)
        decoded_sentences.append(sentence)

    return decoded_sentences


def get_class_counts(labels):
    """Count the number of documents for each class."""
    label_counts = Counter(labels)
    logging.info(f"{GREEN} label_counts: {label_counts}{RESET}")
    labels.sort()
    # return label counts in increasing order
    return [label_counts[label] for label in labels]


def get_sampling_lists_per_language(labels, languages, minority_threshold=0.5):
    """Count the labels per language.

    Get the number of samples per labels and per language.
    Inspect the distribution of labels per language and decide, via
    a minority_threshold, if one of those labels needs to be upsampled.
    """
    counter = defaultdict(lambda: defaultdict(int))

    for label, lang in zip(labels, languages):
        counter[lang][int(label)] += 1

    label_lang_pair_to_upsample = []
    label_lang_pair_to_downsample = []
    majority_threshold = 2.0

    label_lang_dist_df = pd.DataFrame.from_dict(counter)
    label_lang_dist_df.reset_index(inplace=True)
    label_lang_dist_df.rename(columns={"index": "class"}, inplace=True)

    if wandb_active:
        wandb.log(
            {f"original_label_distribution": wandb.Table(dataframe=label_lang_dist_df)}
        )

    for lang in set(languages):
        labels_dist = counter[lang]

        if labels_dist[1] < labels_dist[0]:
            minority_ratio = labels_dist[1] / labels_dist[0]
            majority_ratio = labels_dist[0] / labels_dist[1]

            minority_label = 1
            majority_label = 0
        else:
            minority_ratio = labels_dist[0] / labels_dist[1]
            majority_ratio = labels_dist[1] / labels_dist[0]

            minority_label = 0
            majority_label = 1

        logging.info(
            f"{GREEN} {lang}: minority ratio: {minority_ratio}, "
            f"majority ratio: {majority_ratio}{RESET}"
        )

        if minority_ratio <= minority_threshold:
            label_lang_pair_to_upsample.append((lang, minority_label))
            if wandb_active:
                wandb.log({f"upsampling_{lang}": f"class: {minority_label}"})

        if majority_ratio > majority_threshold:
            label_lang_pair_to_downsample.append((lang, majority_label))
            wandb.log({f"downsampling_{lang}": f"class: {majority_label}"})

    return label_lang_pair_to_upsample, label_lang_pair_to_downsample


def up_down_sample_data(
    sentences,
    labels,
    indices,
    languages,
    model_name,
    max_length,
    min_length,
    minority_threshold=0.5,
    t5_model="t5",
):
    """..."""
    logging.info(f"{GREEN} Upsampling data.\n{RESET}")

    counter = 0
    split_labels = []
    split_languages = []
    split_docs = []
    # get the training data for the current split
    for idx in indices:
        split_labels.append(labels[idx])
        split_languages.append(languages[idx])
        split_docs.append(sentences[idx])

    # get lists of label-language pairs to up- or downsample
    up_filter_list, down_filter_list = get_sampling_lists_per_language(
        split_labels, split_languages, minority_threshold=minority_threshold
    )

    if up_filter_list:

        if t5_model == "t5":
            aug = augmentor.Augmentor(
                min_length=min_length,
                max_length=max_length,
                model=T5ForConditionalGeneration.from_pretrained(
                    "t5-small"
                ),  # 'google/mt5-base' # 't5-small'
                tokenizer=T5Tokenizer.from_pretrained(
                    "t5-small"
                ),  # 'google/mt5-base' #'t5-small'
                debug=False,
                device=device,
            )
        else:
            aug = augmentor.Augmentor(
                min_length=min_length,
                max_length=max_length,
                model=MT5ForConditionalGeneration.from_pretrained("google/mt5-base"),
                tokenizer=MT5Tokenizer.from_pretrained("google/mt5-base"),
                debug=False,
                device=device,
            )
        pattern = r"<extra_id_\d+>"

    upsampled_sentences = []
    upsampled_labels = []
    downsample_indices = []
    proba_of_downsampling = 0.6
    updated_counter = defaultdict(lambda: defaultdict(int))

    # iterate over all samples and decide which one to upsample, leave be or
    # save the index of to downsample later
    for new_index, (label, lang, doc) in tqdm(
        enumerate(zip(split_labels, split_languages, split_docs))
    ):
        if (lang, label) in up_filter_list:
            output = aug.get_abstractive_summarization(doc)
            output = re.sub(pattern, "", output).strip()

            upsampled_sentences.append(output)
            upsampled_labels.append(label)
            counter += 1
            # add index that will not be removed
            downsample_indices.append(new_index)

            # because of the current upsampling strategy, we augment every doc
            # with this label and language, resulting in the double number
            # of the original documents
            updated_counter[lang][label.item()] += 2
        # get all indices in the current split that correspond to one of the
        # downsample tuples
        elif (lang, label) in down_filter_list:
            # with a probability of `proba_of_downsampling` ...
            downsample = np.random.choice(
                np.arange(0, 2), p=[proba_of_downsampling, 1 - proba_of_downsampling]
            )
            if not downsample:
                # add probability with which to NOT downsample
                downsample_indices.append(new_index)
                updated_counter[lang][label.item()] += 1

        # all other language-label tuples are left untouched
        else:
            downsample_indices.append(new_index)
            updated_counter[lang][label.item()] += 1

    logging.info(f"{GREEN} Added {counter} more samples via upsampling..\n" f"{RESET}")

    label_lang_dist_df = pd.DataFrame.from_dict(updated_counter)
    label_lang_dist_df.reset_index(inplace=True)
    label_lang_dist_df.rename(columns={"index": "class"}, inplace=True)
    wandb.log(
        {f"augmented_label_distribution": wandb.Table(dataframe=label_lang_dist_df)}
    )

    if upsampled_sentences:

        return (
            tokenize(
                model_name=model_name,
                sentences=upsampled_sentences,
                labels=upsampled_labels,
                max_length=max_length,
            ),
            downsample_indices,
        )

    # in case there is nothing to up/downsample, return empty lists
    return ([], [], []), downsample_indices


def get_data_loader(
    input_ids,
    att_masks,
    labels,
    batch_size=16,
    shuffle=True,
    sampler=False,
    model_name=None,
):
    """..."""
    # if the sampler is set to a string, it's about the train data loader
    # if sampler:
    #     oversample_minority_class(input_ids=input_ids, labels=labels,
    #                               model_name=model_name)
    # assert False
    dataset = TensorDataset(input_ids, att_masks, labels)

    if sampler == "weighted":
        logging.info(" Using weighted sampler (train).")

        logging.info(f" #train samples in split: {len(dataset)}")
        sample_weights = get_sample_weights(dataset)

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    elif sampler == "random":
        logging.info(" Using random sampler (train).")
        logging.info(f" #train samples in split: {len(dataset)}")

        sampler = RandomSampler(dataset)

        return DataLoader(
            dataset, shuffle=False, sampler=sampler, batch_size=batch_size
        )

    else:
        logging.info(" Using sequential sampler (dev).")

        sampler = SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler
        )


def build_dataset(input_ids, attention_masks, labels):
    """Build dataset from tokenized data when not using cross validation."""
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logging.info(f" {len(train_dataset):>5,} training samples")
    logging.info(f" {len(val_dataset):>5,} validation samples")

    return train_dataset, val_dataset


def get_sample_weights(dataset):
    """Calculate a weight for each sample."""
    # get the labels of the current dataset (usually only the training set)
    labels = torch.tensor([sample[2] for sample in dataset])
    # get all unique labels
    unique_labels = torch.unique(labels, sorted=True)
    # compute the sample weights (each sample should get its own weight)
    class_sample_count = torch.tensor([(labels == t).sum() for t in unique_labels])
    logging.info(f" Class sample count: {class_sample_count}")
    # invert the weights
    weight = 1.0 / class_sample_count.float()
    logging.info(f" weights: {weight}")
    # assign a corresponding weight to each sample
    sample_weights = torch.tensor([weight[t] for t in labels])

    return sample_weights


# def build_train_dev_set(input_ids, attention_masks, labels):
#     """Return a tensor dataset of the given input data."""
#     return TensorDataset(input_ids, attention_masks, labels)


def create_data_loaders(
    train_dataset, val_dataset, batch_size, shuffle=True, sampler=True
):
    """..."""
    # The DataLoader needs to know batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a
    # batch size of 16 or 32.
    if sampler == "weighted":

        # trainloader = DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     shuffle=shuffle,
        #     sampler=sampler)
        logging.info("Using weighted sampler (train) and sequential sampler (dev).")

        sample_weights = get_sample_weights(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            sampler=WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights)
            ),
            batch_size=batch_size,
        )
        # For validation the order doesn't matter, so we'll just read them
        # sequentially.
        validation_loader = DataLoader(
            val_dataset,  # The validation samples.
            # Pull out batches sequentially.
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size,  # Evaluate with this batch size.
        )

    else:
        logging.info("Using random sampler (train) and sequential sampler (dev).")

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_loader = DataLoader(
            train_dataset,
            shuffle=False,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
        )

        # For validation the order doesn't matter, so we'll just read them
        # sequentially.
        validation_loader = DataLoader(
            val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
        )

    return train_loader, validation_loader


def prepare_data(data_file, min_num_tokens=None, max_num_tokens=100, mode="train"):
    """Load data and return two lists."""
    docs = []
    labels = []
    sentence_len = []
    data = []
    languages = []
    # add a counter for counting documents that were removed because
    # they are too short/too long
    removed_count = 0
    set_labels_to_0 = False

    lenght_per_language_per_label = defaultdict(lambda: defaultdict(list))
    removed_per_language_per_label = defaultdict(lambda: defaultdict(int))

    with open(data_file, "r") as read_handle:
        for line in read_handle:
            post = json.loads(line)
            sentence = post["text"]
            post["original_sentence"] = post["text"]
            tokens = sentence.split()
            num_tokens = len(tokens)

            if min_num_tokens is None:
                if num_tokens > max_num_tokens:
                    cut_off = tokens[:max_num_tokens]
                    post["text"] = " ".join(cut_off)
                    sentence_len.append(len(cut_off))
                    lenght_per_language_per_label[int(post["label"])][
                        post["language"]
                    ].append(len(cut_off))
                else:
                    sentence_len.append(num_tokens)
                    lenght_per_language_per_label[int(post["label"])][
                        post["language"]
                    ].append(num_tokens)
                docs.append(post)
                try:
                    labels.append(int(post["label"]))
                # assume that the labels are "unk" for unlabeled test sets
                except ValueError:
                    set_labels_to_0 = True
                    labels.append(0)

                languages.append(post["language"])
                data.append(post)

            elif num_tokens >= min_num_tokens:
                if num_tokens > max_num_tokens:
                    cut_off = tokens[:max_num_tokens]
                    post["text"] = " ".join(cut_off)
                    sentence_len.append(len(cut_off))
                    lenght_per_language_per_label[int(post["label"])][
                        post["language"]
                    ].append(len(cut_off))
                else:
                    sentence_len.append(num_tokens)
                    lenght_per_language_per_label[int(post["label"])][
                        post["language"]
                    ].append(num_tokens)

                docs.append(post)
                labels.append(int(int(post["label"])))
                languages.append(post["language"])
                data.append(post)
            else:
                removed_count += 1
                removed_per_language_per_label[post["language"]][
                    int(post["label"])
                ] += 1

    assert len(docs) == len(labels), "#labels and #sentences do not correspond"

    logging.info(f" #posts: {len(docs)}, #labels: {len(labels)}")
    logging.info(f" longest post: {max(sentence_len)}")
    logging.info(f" avrg post length: {sum(sentence_len) / len(sentence_len)}")

    lenght_per_language_per_label_avrg = defaultdict(lambda: defaultdict(int))
    lenght_per_language_per_label_max = defaultdict(lambda: defaultdict(int))
    lenght_per_language_per_label_min = defaultdict(lambda: defaultdict(int))

    for lang, d in lenght_per_language_per_label.items():
        for label, sent_lens in d.items():
            lenght_per_language_per_label_avrg[label][lang] = sum(sent_lens) / len(
                sent_lens
            )
            lenght_per_language_per_label_max[label][lang] = max(sent_lens)
            lenght_per_language_per_label_min[label][lang] = min(sent_lens)

    data_stats = {
        "avrg_post_length_after_pruning": lenght_per_language_per_label_avrg,
        "longest_post_after_pruning": lenght_per_language_per_label_max,
        "shortest_post_after_pruning": lenght_per_language_per_label_min,
        # "min_num_tokens": min_num_tokens,
        # "max_num_tokens": max_num_tokens,
        "removed": removed_per_language_per_label,
    }
    if set_labels_to_0:
        logging.warning(
            " Received labels that could not be converted to 0 "
            "or 1, set all labels to 0."
        )

        trans_labels = None

    else:
        logging.info(
            f" Removed {removed_count} docs with length < " f"{min_num_tokens}"
        )
        trans_labels, labels_per_lang = utils.create_label_language_combi(
            labels=labels, languages=languages
        )

        data_stats.update({"labels_per_language": labels_per_lang})

    for outer_key, inner_dict in data_stats.items():
        stats_df = pd.DataFrame(inner_dict)
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={"index": "class"}, inplace=True)
        wandb.log({f"{outer_key}_{mode}": wandb.Table(dataframe=stats_df)})

    return docs, labels, languages, trans_labels


def prepare_test_data(test_data_file, min_num_tokens, model_name, max_num_tokens, wandb_active=True):
    """..."""
    # Create sentence and label lists
    docs, labels, _, _ = prepare_data(
        test_data_file, min_num_tokens=min_num_tokens, mode="test"
    )
    sentences = [doc["text"] for doc in docs]
    languages = [doc["language"] for doc in docs]

    assert len(sentences) == len(labels)

    if wandb_active:
        wandb.log({"test_docs": len(sentences)})

    input_ids, attention_masks, labels = tokenize_test_data(
        model_name=model_name,
        sentences=sentences,
        labels=labels,
        max_length=max_num_tokens,
    )

    return input_ids, attention_masks, labels, languages, sentences


def tokenize_test_data(model_name, sentences, labels, max_length):
    """..."""
    return tokenize(
        model_name=model_name, sentences=sentences, labels=labels, max_length=max_length
    )


def get_random_shots(sentences, labels, num_shots, max_length, min_length, wandb_active=True):
    """Randomly sample num_shots examples from the test data."""
    shot_sentences = []
    shot_labels = []
    print(f"Collecting {num_shots} shots.")
    while len(shot_sentences) < num_shots:
        random_int = random.randint(0, len(sentences) - 1)

        shot_sent = sentences[random_int]
        # before removing the sentence, check if it fits the criteria
        if min_length <= len(shot_sent.split()) <= max_length:

            shot_sent = sentences.pop(random_int)
            shot_label = labels.pop(random_int)
            shot_sentences.append(shot_sent)
            shot_labels.append(shot_label)
        print(f"#shots collected: {len(shot_sentences)}")

    shots_df = pd.DataFrame(
        list(zip(shot_sentences, shot_labels)), columns=["sample", "label"]
    )
    shots_df.reset_index(inplace=True)
    if wandb_active:
        wandb.log({f"Shots": wandb.Table(dataframe=shots_df)})

    return shot_sentences, shot_labels, sentences, labels


def get_test_data_loader(input_ids, attention_masks, labels, batch_size):
    """Build a data laoder for the test data."""
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data,
        sampler=prediction_sampler,
        batch_size=batch_size,
        shuffle=False,
    )

    return prediction_dataloader, len(input_ids)
