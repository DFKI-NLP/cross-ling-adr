"""Some baselines to compare to the more advanced models.

Contains the following models:

1) minority classifier
2) majority classifier
3) uniform classifier
4) per-language classifier
5) Naive Bayes (Gaussian & Multinomial)

TODO:
6) Decision Tree
7) SVM

Usage:
python baselines.py data/combined/traindev_data.jsonl
                    data/combined/test_data.jsonl nb
"""

import argparse
import csv
import datetime
import json
import sys
import wandb

from collections import namedtuple
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np

import fasttext
import fasttext.util
import stopwordsiso as stopwords

import data

from utils import training_utils as train_utils
from utils import visualizations
from utils.embedder import MeanEmbeddingVectorizer

wandb.init(project="final_binary_classification")

Scores = namedtuple("scores", ["y_true", "y_score", "y_pred", "metrics"])


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Extracts language and post length from each document."""

    def __init__(self):
        """..."""
        pass

    def fit(self, x, y=None):
        """..."""
        return self

    def transform(self, data):
        """..."""
        return self.extract_language_and_length(data)

    def extract_language_and_length(self, data):
        """..."""
        feature_dicts = []

        for doc in data:
            feature_dict = {
                "language": doc["language"],
                "length": len(doc["text"].strip().split(" ")),
            }
            feature_dicts.append(feature_dict)
        return feature_dicts


class ItemSelector(BaseEstimator, TransformerMixin):
    """..."""

    def __init__(self, key):
        """..."""
        self.key = key

    def fit(self, x, y=None):
        """..."""
        return self

    def transform(self, data):
        """..."""
        return [x[self.key] for x in data]


def run_majority_classifier(train_data, train_labels, test_data, test_labels):
    """..."""
    print(
        "Run random classifier with most_frequent strategy: always predicts "
        "the most frequent label in the training set.."
    )

    train_data = [doc["text"] for doc in train_data]
    test_data = [doc["text"] for doc in test_data]

    clf = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", DummyClassifier(strategy="most_frequent", random_state=42)),
        ]
    )

    clf.fit(train_data, train_labels)

    y_pred = clf.predict(test_data)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = clf.predict_proba(test_data)
    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_uniform_classifier(train_data, train_labels, test_data, test_labels):
    """..."""
    print(
        "Run random classifier with uniform strategy: generates predictions "
        "uniformly at random."
    )

    X_train = [x["text"] for x in train_data]
    X_test = [x["text"] for x in test_data]

    clf = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", DummyClassifier(strategy="uniform", random_state=42)),
        ]
    )

    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = clf.predict_proba(X_test)

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_minority_classifier(train_data, train_labels, test_data, test_labels):
    """..."""
    print(
        "Run random classifier with constant strategy: always predicts the "
        "minority label in the training set.."
    )

    X_train = [x["text"] for x in train_data]
    X_test = [x["text"] for x in test_data]

    clf = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", DummyClassifier(strategy="constant", random_state=42, constant=0)),
        ]
    )

    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = clf.predict_proba(X_test)
    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def predict_according_to_language(test_data, test_labels, language):
    """..."""
    print(
        "Run classifier that predicts '1' for English, '0' for all other " "languages."
    )

    y_pred = []
    for i in range(len(test_data)):
        if language[i] == "english":
            y_pred.append(1)
        else:
            y_pred.append(0)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    # y_scores = [float(x) for x in y_pred]
    # print(y_scores)
    y_scores = np.empty(shape=(len(y_pred), 2))

    for i, x in enumerate(y_pred):
        if x == 1:
            y_scores[i] = [0, 1]

        else:
            y_scores[i] = [1, 0]

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def get_lang_len_features(data, language, language_dict):
    """Convert the data to two-data (language and post length) only."""
    X = []

    for i, post in enumerate(data):
        # get post length
        post_len = len(post["text"].strip().split(" "))
        # get language index
        lang_idx = language_dict[language[i]]

        X.append([post_len, lang_idx])

    return X


def run_gaussian_naive_bayes(
    train_data, train_labels, train_lang, test_data, test_labels, test_lang
):
    """Train a two-features only Naive Bayes classifier."""
    print(
        "\nRun Gaussian Naive Bayes with two features: " "language and post length ..."
    )
    lang_dict = {}

    for j, lang in enumerate(set(train_lang)):
        lang_dict[lang] = j

    X_train = get_lang_len_features(
        data=train_data, language=train_lang, language_dict=lang_dict
    )

    X_test = get_lang_len_features(
        data=test_data, language=test_lang, language_dict=lang_dict
    )

    clf = GaussianNB()

    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = clf.predict_proba(X_test)

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_multinomial_naive_bayes(
    train_data, train_labels, train_lang, test_data, test_labels, test_lang
):
    """..."""
    print("\nRun multinomial Naive Bayes ...")

    X_train = [x["text"] for x in train_data]
    X_test = [x["text"] for x in test_data]

    clf = Pipeline(
        [
            ("vect", TfidfVectorizer(analyzer="word")),
            ("clf", MultinomialNB()),
        ]
    )

    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)

    results = train_utils.get_results(y_true=train_labels, y_pred=y_pred)

    y_scores = clf.predict_proba(X_test)
    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_NB_on_tfidf_lang_and_len(train_data, train_labels, test_data, test_labels):
    """..."""
    print("\nRun multinomial Naive Bayes on cleaned words + language + length " "...")

    bow_pipeline = Pipeline(
        steps=[
            ("selector", ItemSelector(key="text")),
            # TODO: get stop word lists
            ("tfidf", TfidfVectorizer(analyzer="word")),
        ]
    )
    manual_pipeline = Pipeline(
        steps=[
            ("language_length", CustomTransformer()),
            ("dict_vect", DictVectorizer()),
        ]
    )

    combined_features = FeatureUnion(
        transformer_list=[
            ("bow", bow_pipeline),
            ("manual", manual_pipeline),
        ]
    )

    final_pipeline = Pipeline(
        steps=[
            ("combined_features", combined_features),
            ("clf", MultinomialNB()),
        ],
        verbose=False,
    )
    final_pipeline.fit(train_data, train_labels)

    y_pred = final_pipeline.predict(test_data)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = final_pipeline.predict_proba(test_data)

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_multinomial_nb(train_data, train_labels, test_data, test_labels):
    """..."""
    print("Run multinomial Naive Bayes ...")

    X_train = [x["text"] for x in train_data]
    X_test = [x["text"] for x in test_data]

    clf = Pipeline(
        [
            ("vect", TfidfVectorizer(analyzer="char")),
            ("clf", MultinomialNB()),
        ]
    )

    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = clf.predict_proba(X_test)
    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def plot_tree(clf, features):
    """..."""
    import graphviz

    dot_data = tree.export_graphviz(
        clf, out_file=None, filled=True, class_names=["0", "1"], feature_names=features
    )

    graph = graphviz.Source(dot_data)
    graph.render("DecisionTree")


def run_decision_tree_classifier(train_data, train_labels, test_data, test_labels):
    """..."""
    X_train = [x["text"] for x in train_data]
    X_test = [x["text"] for x in test_data]

    decision_tree_pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer(analyzer="char")),
            (
                "clf",
                DecisionTreeClassifier(
                    max_leaf_nodes=20, random_state=42, min_samples_split=100
                ),
            ),
        ]
    )

    decision_tree_pipeline.fit(X_train, train_labels)
    y_pred = decision_tree_pipeline.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    word2id = decision_tree_pipeline["vect"].vocabulary_
    print(f"#words in vocab: {len(word2id)}")
    id2word = {idx: word for word, idx in word2id.items()}

    feature_importances = decision_tree_pipeline["clf"].feature_importances_
    most_imp = np.argpartition(feature_importances, -20)[-20:]

    y_scores = decision_tree_pipeline.predict_proba(X_test)
    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_svm_on_embeddings(
    train_data, train_labels, test_data, test_labels, aligned_embeds=True, seed=42
):
    """Run a SVM on aligned embeddings."""
    available_languages = []
    for doc in train_data + test_data:
        available_languages.append(doc["language"])

    embedder = MeanEmbeddingVectorizer(
        use_aligned_embeddings=aligned_embeds, languages=set(available_languages)
    )
    pipeline = Pipeline(
        [
            ("embedder", embedder),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=seed)),
        ]
    )

    pipeline.fit(train_data, train_labels)
    y_pred = pipeline.predict(test_data)
    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = pipeline.predict_proba(test_data)

    print(f"y_scores: {y_scores}")

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_svm_on_embeddings_sweep(
    train_data, train_labels, test_data, test_labels, aligned_embeds=True, seed=42
):
    """Run a SVM on aligned embeddings."""
    available_languages = []
    for doc in train_data + test_data:
        available_languages.append(doc["language"])

    embedder = MeanEmbeddingVectorizer(
        use_aligned_embeddings=aligned_embeds, languages=set(available_languages)
    )
    pipeline = Pipeline(
        [
            ("embedder", embedder),
            ("clf", SVC(class_weight="balanced", random_state=seed)),
        ]
    )

    pipeline.fit(train_data, train_labels)
    y_pred = pipeline.predict(test_data)
    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    # y_scores = pipeline.predict_proba(test_data)

    # return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)
    return results, y_pred


def run_svm_on_bow(train_data, train_labels, test_data, test_labels):
    """SVM Classifier on bag-of-words."""
    X_train = [doc["text"] for doc in train_data]
    X_test = [doc["text"] for doc in test_data]

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("clf", SVC(probability=True, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, train_labels)

    y_pred = pipeline.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = pipeline.predict_proba(X_test)

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def run_svm_on_tfidf(train_data, train_labels, test_data, test_labels):
    """SVM Classifier on bag-of-words."""
    X_train = [doc["text"] for doc in train_data]
    X_test = [doc["text"] for doc in test_data]

    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", SVC(probability=True, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, train_labels)

    y_pred = pipeline.predict(X_test)

    results = train_utils.get_results(y_true=test_labels, y_pred=y_pred)

    y_scores = pipeline.predict_proba(X_test)

    return Scores(y_true=test_labels, y_score=y_scores, y_pred=y_pred, metrics=results)


def plot_results(results_list, descriptions):
    """..."""
    visualizations.plot_precision_recall_curve(results_list, descriptions, "baselines")
    visualizations.plot_roc_curve(results_list, descriptions, "baselines")


def write_predictions_to_file(predictions_dict, docs_test, labels_test):
    """..."""
    assert len(labels_test) == len(docs_test)

    fieldnames = ["gold"] + list(predictions_dict.keys()) + ["text"]

    date = datetime.datetime.now().strftime("%d_%m_%y_%H_%M")

    with open(f"predictions_baselines_{date}.csv", "w") as write_handle:

        writer = csv.DictWriter(write_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, doc in enumerate(docs_test):
            row = {"text": doc["text"], "gold": labels_test[i]}

            for classif, predictions in predictions_dict.items():
                row[classif] = predictions[i]

            writer.writerow(row)


if __name__ == "__main__":

    choices = [
        "majority",
        "minority",
        "uniform",
        "language",
        "nb_tfidf_lang_len",
        "nb_lang_len",
        "mnb",
        "dt",
        "svm_aligned",
        "svm_separated",
        "svm_bow",
        "svm_tfidf",
        "all",
    ]
    parser = argparse.ArgumentParser()

    parser.add_argument("train_dev_data", help=("Path to train/dev data."))

    parser.add_argument("test_data", default=None, help="Path to test data.")

    parser.add_argument(
        "--mode", "-m", choices=choices, action="append", help="The sort of classifier."
    )

    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot precision-recall & roc curve"
    )

    parser.add_argument(
        "-sp",
        "--store_predictions",
        action="store_true",
        help="Store predictions of all classifiers in a file.",
    )

    args = parser.parse_args()

    print(
        "\n\n########### Running multiple classifiers at once does not work "
        "with wandb! ###########\n\n"
    )

    min_num_tokens = 4
    max_num_tokens = 300
    wandb.config.cross_val = False
    wandb.config.epochs = False
    wandb.config.train_data = args.train_dev_data
    wandb.config.test_data = args.test_data

    wandb.config.max_length = max_num_tokens
    wandb.config.min_length = min_num_tokens

    print("Preparing training data ...")
    (docs_train, labels_train, lang_train, _) = data.prepare_data(
        args.train_dev_data,
        min_num_tokens=min_num_tokens,
        max_num_tokens=max_num_tokens,
    )
    print("\nPreparing test data ...")
    (docs_test, labels_test, lang_test, _) = data.prepare_data(
        args.test_data, min_num_tokens=min_num_tokens, max_num_tokens=max_num_tokens
    )

    results = []
    classifier_descriptions = []
    predictions_dict = {}

    classifiers = args.mode

    if "all" in args.mode:
        classifiers = choices
        classifiers.remove("all")

    for classifier in classifiers:
        # will be overwritten when using more than one model
        wandb.config.update({"model_name": classifier}, allow_val_change=True)

        if classifier == "majority":
            result = run_majority_classifier(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )

            predictions_dict[classifier] = result.y_pred

            results.append(result)
            classifier_descriptions.append("Classification according to majority class")

        if classifier == "minority":
            result = run_minority_classifier(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("Classification according to minority class")

        if classifier == "uniform":
            result = run_uniform_classifier(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append(
                "Classification according to uniform distribution"
            )

        if classifier == "language":
            result = predict_according_to_language(
                test_data=docs_test, test_labels=labels_test, language=lang_test
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("Classification according to language")

        if classifier == "nb_lang_len":
            result = run_gaussian_naive_bayes(
                train_data=docs_train,
                train_labels=labels_train,
                train_lang=lang_train,
                test_data=docs_test,
                test_labels=labels_test,
                test_lang=lang_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("Gaussian NB on lang & len")

        if classifier == "mnb":

            result = run_multinomial_nb(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("Multinomial NB on char tf-idf")

        if classifier == "nb_tfidf_lang_len":
            result = run_NB_on_tfidf_lang_and_len(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("Multinomial NB on tf-idf, lang & length")

        if classifier == "dt":

            result = run_decision_tree_classifier(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)
            classifier_descriptions.append("Decision Tree on char tf-idf")

        if classifier == "svm_aligned":
            result = run_svm_on_embeddings(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
                aligned_embeds=True,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("SVM on aligned embeddings")

        if classifier == "svm_separated":
            result = run_svm_on_embeddings(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
                aligned_embeds=False,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("SVM on 'separate' embeddings")

        if classifier == "svm_bow":
            result = run_svm_on_bow(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("SVM on BOW.")

        if classifier == "svm_tfidf":
            result = run_svm_on_tfidf(
                train_data=docs_train,
                train_labels=labels_train,
                test_data=docs_test,
                test_labels=labels_test,
            )
            predictions_dict[classifier] = result.y_pred

            results.append(result)

            classifier_descriptions.append("SVM on tf-idf")

    if args.plot:
        plot_results(results_list=results, descriptions=classifier_descriptions)

    if args.store_predictions:
        write_predictions_to_file(predictions_dict, docs_test, labels_test)
