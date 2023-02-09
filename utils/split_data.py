"""Create a tran/dev test split.

Script for splitting up a given jsonl dataset into train/dev and test set
without duplicates.

Hash-based de-duplication taken from
https://gist.github.com/SilentFlame/310136be346fda5caf03e6acb27a5a9b
(24/02/2021)

python utils/split_data.py -input_data data/twitter_data/fr/french_twitter_all.jsonl
-proportion_test_set 0.1
-train_dev_output data/twitter_data/fr/traindev_set_twitter_french.jsonl
-test_output data/twitter_data/fr/testset_twitter_french.jsonl
-language french
--stratify_by_lang

"""

import argparse
import hashlib
import json
import numpy as np
import sys
import uuid

from random import randint

import utils

from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit


hashSet = set()


def hash_text(text):
    """Create a hash value for the given text."""
    # uuid is used to generate a random number
    return hashlib.md5(text).hexdigest()


def remove_duplicates(data_list):
    """Remove duplicates via hash functions."""
    removed_dups = []
    label_list = []
    document_counter = 0
    duplicate_counter = 0
    not_available_counter = 0
    label_counter = defaultdict(lambda: defaultdict(int))
    languages = []
    hash_set = set()

    for dp in data_list:
        document_counter += 1
        text = dp["text"].strip()

        # ignore not longer existing tweets
        if text == "Not Available":
            not_available_counter += 1
            continue

        # encoding for cases of special chars in sentence.
        hash_value = hash_text(text.encode("utf-8"))

        if hash_value in hash_set:
            duplicate_counter += 1
            continue
        else:
            removed_dups.append(dp)
            label_list.append(dp["label"])
            label_counter[dp["language"]][dp["label"]] += 1
            languages.append(dp["language"])

            hash_set.add(hash_value)

    print(
        f"\n#incoming docs: {document_counter}\n"
        f"Found & removed {duplicate_counter} duplicates."
    )

    if not_available_counter:
        print(
            f"Found & removed {not_available_counter} no longer existing " "documents."
        )

    print(f"\nlabels: {json.dumps(label_counter, indent=2)}")
    num_samples = len(removed_dups)
    print(f"#samples: {num_samples}")

    return removed_dups, label_list, languages


def stratify_data_by_language(data, labels, languages, stratifier):
    """Stratify data by labels AND language."""
    print("Stratify data by label AND language.")

    # create a list that combines all language and all labels as pseudo labels
    # for stratification (ONLY FOR STRATIFICATION)
    trans_labels, labels_per_lang = utils.create_label_language_combi(
        labels=labels, languages=languages
    )

    print(json.dumps(labels_per_lang, indent=2))

    print(len(data), print(len(trans_labels)))
    # create the split in the transformed labels, but use the original
    # label list for creating the data
    for train_idx, test_idx in stratifier.split(data, trans_labels):
        X_train = np.array(data)[train_idx]
        X_test = np.array(data)[test_idx]

    return X_train, X_test


def balance_data_per_language(data, labels, languages, stratifier):
    """...."""
    print("Balance data.")
    # create a list that combines all language and all labels as pseudo labels
    # for stratification (ONLY FOR STRATIFICATION)
    _, labels_per_lang = utils.create_label_language_combi(
        labels=labels, languages=languages
    )
    balanced_data = []
    print(f"#data before: {len(data)}")
    for lang, labels_dist in labels_per_lang.items():
        minority_label = int(min(labels_dist, key=labels_dist.get))
        majority_label = int(max(labels_dist, key=labels_dist.get))
        minority_amount = labels_dist[minority_label]
        majority_data = [
            x for x in data if x["label"] == majority_label and x["language"] == lang
        ]
        minority_data = [
            x for x in data if x["label"] == minority_label and x["language"] == lang
        ]

        print(f"{lang}: {len(minority_data)}")

        assert len(minority_data) == minority_amount
        balanced_data.extend(minority_data)

        print(f"balanced data after adding {lang} minority: {len(balanced_data)}")

        randomly_picked_majority = []
        # from the majority class, pick the same amount of documents as there
        # are in the minority class
        while minority_amount > 0:
            # get a random number
            # Pick a random number between 1 and 100.
            random_num = randint(0, len(majority_data) - 1)
            randomly_picked_majority.append(majority_data.pop(random_num))
            minority_amount -= 1

        print(f"#majority picked: {len(randomly_picked_majority)}")
        balanced_data.extend(randomly_picked_majority)

        print(f"balanced data after adding {lang} majority: {len(balanced_data)}")

    print(f"#data after: {len(balanced_data)}")

    # assert len(balanced_data) == ()
    # get the document
    # check if it is in the majority class
    # if yes, add it to the data list
    balanced_labels = [doc["label"] for doc in balanced_data]
    balanced_lang = [doc["language"] for doc in balanced_data]

    _, balanced_labels_per_lang = utils.create_label_language_combi(
        labels=balanced_labels, languages=balanced_lang
    )
    print(json.dumps(balanced_labels_per_lang, indent=2))

    for train_idx, test_idx in stratifier.split(balanced_data, balanced_labels):
        X_train = np.array(balanced_data)[train_idx]
        X_test = np.array(balanced_data)[test_idx]

    return X_train, X_test


def split_data(
    data_file,
    test_file,
    train_file,
    language=None,
    proportion_test_set=0.2,
    create_debug_files=False,
    stratify_by_language=False,
    balance=False,
):
    """Shuffle and split data into stratified train/dev and test set."""
    data_list = []
    # collect all data in file
    with open(data_file, "r") as file_handler:
        for line in file_handler:
            try:
                data = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print(line)
                raise e
            if language is not None:
                data["language"] = language
                data["label"] = int(data["label"])
            data_list.append(data)

    # remove duplicates
    data_wo_duplicates, label_list, languages = remove_duplicates(data_list)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=proportion_test_set, random_state=42
    )

    if stratify_by_language:
        X_train, X_test = stratify_data_by_language(
            data=data_wo_duplicates,
            labels=label_list,
            languages=languages,
            stratifier=sss,
        )

    elif balance:
        X_train, X_test = balance_data_per_language(
            data=data_wo_duplicates,
            labels=label_list,
            languages=languages,
            stratifier=sss,
        )

    else:

        for train_idx, test_idx in sss.split(data_wo_duplicates, label_list):
            X_train = np.array(data_wo_duplicates)[train_idx]
            X_test = np.array(data_wo_duplicates)[test_idx]

    if create_debug_files:
        X_train = X_train[:100]
        X_test = X_test[:30]

    # with open("data/forum_data/combined/combined_corpus_filtered.jsonl", "w") as write_handle:
    #     for dp in X_train:

    #         write_handle.write(json.dumps(dp, ensure_ascii=False))
    #         write_handle.write("\n")
    #     for dp in X_test:

    #         write_handle.write(json.dumps(dp, ensure_ascii=False))
    #         write_handle.write("\n")

    # assert False

    testset_count = defaultdict(lambda: defaultdict(int))
    testset_count[1] = 0
    testset_count[0] = 0

    trainset_count = defaultdict(lambda: defaultdict(int))
    trainset_count[1] = 0
    trainset_count[0] = 0

    test_lang_count = defaultdict(int)
    train_lang_count = defaultdict(int)

    with open(train_file, "w") as train_r:
        for dp in X_train:
            trainset_count[dp["language"]][dp["label"]] += 1
            trainset_count[dp["label"]] += 1
            train_lang_count[dp["language"]] += 1
            train_lang_count["samples"] += 1

            train_r.write(json.dumps(dp, ensure_ascii=False))
            train_r.write("\n")

    with open(test_file, "w") as test_r:
        for dp in X_test:
            testset_count[dp["language"]][dp["label"]] += 1
            testset_count[dp["label"]] += 1
            test_lang_count[dp["language"]] += 1
            test_lang_count["samples"] += 1

            test_r.write(json.dumps(dp, ensure_ascii=False))
            test_r.write("\n")

    testset_count["ratio_neg_pos"] = testset_count[0] / testset_count[1]

    trainset_count["ratio_neg_pos"] = trainset_count[0] / trainset_count[1]

    # update the label counters to include languages separately
    for lang in languages:
        testset_count[lang]["ratio_neg_pos"] = (
            testset_count[lang][0] / testset_count[lang][1]
        )

        trainset_count[lang]["ratio_neg_pos"] = (
            trainset_count[lang][0] / trainset_count[lang][1]
        )

        train_lang_count[f"ratio_{lang}"] = (
            train_lang_count[lang] / train_lang_count["samples"]
        )

        test_lang_count[f"ratio_{lang}"] = (
            test_lang_count[lang] / test_lang_count["samples"]
        )

    print(f"\nlabels train: {json.dumps(trainset_count, indent=2)}")
    print(f"labels test: {json.dumps(testset_count, indent=2)}")

    print(f"\n\nlanguages train: {json.dumps(train_lang_count, indent=2)}")
    print(f"languages test: {json.dumps(test_lang_count, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-input_data", type=str)
    parser.add_argument("-proportion_test_set", type=float, default=0.1)
    parser.add_argument("-train_dev_output", type=str, default="traindevset.jsonl")
    parser.add_argument("-test_output", type=str, default="testset.jsonl")
    parser.add_argument(
        "-language",
        type=str,
        default=None,
        help=(
            "If the dataset does not contain the languages as key, please specify it."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--stratify_by_lang",
        action="store_true",
        help=("Stratify the data not only by label, but also by language."),
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help=(
            "Balance the data, so that for each language, "
            "there is an equal amount of documents per "
            "label."
        ),
    )

    args = parser.parse_args()

    if args.stratify_by_lang and args.balance:
        print(
            "Stratification by language AND balancing is not implemented. " "Aborting."
        )
        sys.exit()
    create_debug_files = args.debug

    if create_debug_files:
        test_data_file = (
            "/home/lisa/projects/adr_classification/data/"
            "twitter_data/fr/test_testset_fr_twitter.jsonl"
        )
        train_data_file = (
            "/home/lisa/projects/adr_classification/data/"
            "twitter_data/fr/test_traindevset_fr_twitter.jsonl"
        )

    else:
        test_data_file = args.test_output
        train_data_file = args.train_dev_output

    split_data(
        data_file=args.input_data,
        language=args.language,
        test_file=test_data_file,
        train_file=train_data_file,
        proportion_test_set=args.proportion_test_set,
        create_debug_files=create_debug_files,
        stratify_by_language=args.stratify_by_lang,
        balance=args.balance,
    )
