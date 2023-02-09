"""Some general utils."""
import errno
import os

from collections import defaultdict


def make_sure_path_exists(path):
    """If the dir does not exit, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_label_language_combi(labels, languages):
    """Create a mapping of label-language pairs to index and vice versa."""
    # get all combinations of language + label
    combis = [x + str(y) for x in set(languages) for y in set(labels)]
    combi2id = {combi: i for i, combi in enumerate(combis)}
    # id2combi = {i: combi for i, combi in enumerate(combis)}

    transformed_labels = []
    labels_per_lang = defaultdict(lambda: defaultdict(int))

    for lang, label in zip(languages, labels):
        transformed_labels.append(combi2id[lang + str(label)])
        labels_per_lang[lang][label] += 1

    return transformed_labels, labels_per_lang
