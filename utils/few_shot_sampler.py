"""..."""
import logging
import numpy as np
import random
import sys

from .colors import GREEN
from .colors import RED
from .colors import RESET

logging.basicConfig(level=logging.INFO)


class FewShotSampler(object):
    """Splits data according to few shot criteria."""

    def __init__(self, num_splits=5, num_shots=2, random_state=42, shuffle=True):
        """..."""
        # self.num_pos = num_pos
        # self.num_neg = num_neg
        self.num_shots = num_shots
        self.random_state = random_state
        self.num_splits = num_splits
        self.shuffle = shuffle
        random.seed(random_state)

    def collect_indices_per_class(self, data, labels):
        """Create one list per class."""
        positive_indices = []
        negative_indices = []

        for i, (doc, label) in enumerate(zip(data, labels)):
            if int(label) == 1:
                positive_indices.append(i)
            else:
                negative_indices.append(i)

        return positive_indices, negative_indices

    def check_sample_size(self, num_pos_overall, num_neg_overall, num_pos, num_neg):
        """Check if there are enough samples for the desired splits."""
        if num_pos_overall < num_pos * 2:
            logging.error(f"{RED} Not enough positive samples!\n{RESET}")

            return False

        elif num_neg_overall < num_neg * 2:
            logging.error(f"{RED} Not enough negative samples!\n{RESET}")
            return False

        return True

    def check_labels(self, docs, labels):
        """..."""
        assert len(docs) == len(labels), logging.error(
            f"{RED} The number of samples ({len(docs)}) and the number of labels ({len(labels)}) do not agree.\n{RESET}"
        )

    def create_train_dev(self, pos_indices, neg_indices, num_pos, num_neg):
        """..."""
        # Step 1: create a copy of the positive and negative indices
        temp_pos_indices = pos_indices.copy()
        temp_neg_indices = neg_indices.copy()

        train_indices = []
        val_indices = []

        # Step 2: get random indices from the pos and neg indices
        # - sort indices in reverse order to keep index structure when deleting
        # indices later (Step 3)
        try:
            random_pos_idx_train = sorted(
                random.sample(range(0, len(temp_pos_indices)), num_pos), reverse=True
            )

        # if there is no positive example, continue with only negative ones
        except ValueError:
            random_pos_idx_train = []

        try:
            random_neg_idx_train = sorted(
                random.sample(range(0, len(temp_neg_indices)), num_neg), reverse=True
            )
        except ValueError:
            random_neg_idx_train = []

        # Step 3: collect the positive and negative indices for the train list
        # - make sure the pos/neg indices are not the same for train and val
        #   by popping the index in question from the list (only do this
        #   with reverse-sorted indices)
        for idx in random_pos_idx_train:
            # add the indices to the train list and remove it from the
            train_indices.append(temp_pos_indices.pop(idx))

        for idx in random_neg_idx_train:
            # add the indices to the train list and remove it from the
            train_indices.append(temp_neg_indices.pop(idx))

        # Step 4: do the exact same procedure for the validation set,
        # but this time with a smaller set of indices per label
        # (since we already removed some indices)
        try:
            random_pos_idx_val = sorted(
                random.sample(range(0, len(temp_pos_indices)), num_pos), reverse=True
            )

        except ValueError:
            random_pos_idx_val = []

        try:
            random_neg_idx_val = sorted(
                random.sample(range(0, len(temp_neg_indices)), num_neg), reverse=True
            )
        except ValueError:
            random_neg_idx_val = []
        # update the positive indices for val
        for idx in random_pos_idx_val:
            # add the indices to the val list and remove it from the
            val_indices.append(temp_pos_indices[idx])

        for idx in random_neg_idx_val:
            # add the indices to the val list and remove it from the
            val_indices.append(temp_neg_indices[idx])

        random.shuffle(train_indices)
        random.shuffle(val_indices)

        return train_indices, val_indices

    def sample_per_class(self, list_of_sentences, list_of_labels):
        """Generate indices to split data into training and test set."""
        self.check_labels(docs=list_of_sentences, labels=list_of_labels)

        num_pos = num_neg = int(self.num_shots / 2)

        # divide the data into positives and negatives
        pos_indices, neg_indices = self.collect_indices_per_class(
            data=list_of_sentences, labels=list_of_labels
        )

        # check that there are enough samples
        if not self.check_sample_size(
            num_pos_overall=len(pos_indices),
            num_neg_overall=len(neg_indices),
            num_pos=num_pos,
            num_neg=num_neg,
        ):
            sys.exit(1)

        logging.info(
            f"{GREEN} Each split will contain {num_pos} positive(s) and {num_neg} negative(s)!\n{RESET}"
        )

        train_indices, val_indices = self.create_train_dev(
            pos_indices=pos_indices,
            neg_indices=neg_indices,
            num_pos=num_pos,
            num_neg=num_neg,
        )

        assert (
            list(set(train_indices) & set(val_indices)) == []
        ), "Found duplicate indices!"

        return train_indices, val_indices

    def sample_randomly(self, list_of_sentences, list_of_labels):
        """..."""
        self.check_labels(docs=list_of_sentences, labels=list_of_labels)

        pos_indices, neg_indices = self.collect_indices_per_class(
            data=list_of_sentences, labels=list_of_labels
        )

        # allow at least one negative sample and make sure there are enough
        # samples for both train and val set
        num_pos = random.randint(0, self.num_shots - 1)
        num_neg = self.num_shots - num_pos

        # make sure there are enough samples and to not accidentally repeat
        # the by-class few shot splits
        while (
            not self.check_sample_size(
                num_pos_overall=len(pos_indices),
                num_neg_overall=len(neg_indices),
                num_pos=num_pos,
                num_neg=num_neg,
            )
        ) or num_pos == num_neg:
            # allow at least one negative sample and make sure there are enough
            # samples for both train and val set
            num_pos = random.randint(0, self.num_shots - 1)
            num_neg = self.num_shots - num_pos

        logging.info(
            f"{GREEN} Each split will contain {num_pos} positive(s) and {num_neg} negative(s)!\n{RESET}"
        )

        train_indices, val_indices = self.create_train_dev(
            pos_indices=pos_indices,
            neg_indices=neg_indices,
            num_pos=num_pos,
            num_neg=num_neg,
        )
        assert (
            list(set(train_indices) & set(val_indices)) == []
        ), "Found duplicate indices!"

        return train_indices, val_indices

    def sample_according_to_distribution(
        self, list_of_sentences, list_of_labels, pos_max=1
    ):
        """..."""
        self.check_labels(docs=list_of_sentences, labels=list_of_labels)

        pos_indices, neg_indices = self.collect_indices_per_class(
            data=list_of_sentences, labels=list_of_labels
        )

        num_pos = pos_max
        num_neg = self.num_shots - num_pos

        # check that there are enough samples
        if not self.check_sample_size(
            num_pos_overall=len(pos_indices),
            num_neg_overall=len(neg_indices),
            num_pos=num_pos,
            num_neg=num_neg,
        ):
            sys.exit(1)

        logging.info(
            f"{GREEN} Each split will contain {num_pos} positive(s) and {num_neg} negative(s)!\n{RESET}"
        )

        train_indices, val_indices = self.create_train_dev(
            pos_indices=pos_indices,
            neg_indices=neg_indices,
            num_pos=num_pos,
            num_neg=num_neg,
        )
        assert (
            list(set(train_indices) & set(val_indices)) == []
        ), "Found duplicate indices!"

        return train_indices, val_indices

    def sample_and_add_negatives(
        self, list_of_sentences, list_of_labels, num_added=100
    ):
        """..."""
        self.check_labels(docs=list_of_sentences, labels=list_of_labels)

        pos_indices, neg_indices = self.collect_indices_per_class(
            data=list_of_sentences, labels=list_of_labels
        )

        num_pos = self.num_shots
        num_neg = num_added

        # check that there are enough samples
        if not self.check_sample_size(
            num_pos_overall=len(pos_indices),
            num_neg_overall=len(neg_indices),
            num_pos=num_pos,
            num_neg=num_neg,
        ):
            sys.exit(1)

        logging.info(
            f"{GREEN} Each split will contain {num_pos} positive(s) and {num_neg} negative(s)!\n{RESET}"
        )

        train_indices, val_indices = self.create_train_dev(
            pos_indices=pos_indices,
            neg_indices=neg_indices,
            num_pos=num_pos,
            num_neg=num_neg,
        )
        assert (
            list(set(train_indices) & set(val_indices)) == []
        ), "Found duplicate indices!"

        return train_indices, val_indices


if __name__ == "__main__":

    X = np.array(
        [
            "positive_sentence_1",
            "negative_sentence_2",
            "negative_sentence_3",
            "positive_sentence_4",
            "positive_sentence_5",
            "positive_sentence_6",
            "negative_sentence_7",
            "negative_sentence_8",
            "negative_sentence_9",
            "negative_sentence_10",
            "positive_sentence_11",
            "positive_sentence_12",
            "positive_sentence_13",
            "negative_sentence_14",
            "negative_sentence_15",
            "negative_sentence_16",
            "negative_sentence_17",
            "negative_sentence_18",
            "negative_sentence_19",
            "negative_sentence_20",
            "negative_sentence_21",
            "positive_sentence_21",
            "positive_sentence_22",
            "positive_sentence_23",
            "positive_sentence_24",
        ]
    )
    y = np.array(
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    )

    X_source = np.array(
        [
            "negative_sentence_source_1",
            "negative_sentence_source_2",
            "negative_sentence_source_3",
            "positive_sentence_source_4",
            "positive_sentence_source_5",
            "positive_sentence_source_6",
            "positive_sentence_source_7",
            "negative_sentence_source_8",
            "negative_sentence_source_9",
            "negative_sentence_source_10",
        ]
    )
    y_source = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

    num_source_lang = 2

    for seed in [2, 342, 43, 21, 534]:
        sampler = FewShotSampler(
            num_splits=2, num_shots=2, random_state=seed, shuffle=True
        )

        train_index, val_index = sampler.sample_per_class(
            list_of_sentences=X, list_of_labels=y
        )

        # train_index, val_index = sampler.sample_according_to_distribution(
        #     list_of_sentences=X, list_of_labels=y, pos_max=1
        # )

        # train_index, val_index = sampler.sample_randomly(
        #     list_of_sentences=X, list_of_labels=y
        # )

        # train_index, val_index = sampler.sample_and_add_negatives(
        #     list_of_sentences=X, list_of_labels=y, num_added=7
        # )

        print(f"train docs: {X[train_index]}, {y[train_index]}")
        print(f"val docs: {X[val_index]}, {y[val_index]}")

        # source_lang_idx = random.sample(range(0, len(X_source) - 1), num_source_lang * 2)

        # print(f"source lang idx: {source_lang_idx}")

        # source_lang_idx_train = source_lang_idx[:num_source_lang]
        # source_lang_idx_val = source_lang_idx[num_source_lang:]

        # print(
        #     f"source lang train: {source_lang_idx_train}\nsource lang val: {source_lang_idx_val}"
        # )
