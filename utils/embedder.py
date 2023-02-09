"""..."""

import numpy as np

import fasttext
import fasttext.util
import stopwordsiso as stopwords

from utils.utils import make_sure_path_exists


class MeanEmbeddingVectorizer(object):
    """..."""

    def __init__(self, use_aligned_embeddings=False, languages=set()):
        """..."""
        self.aligned_embeds = use_aligned_embeddings
        self.embedding_dim = 300

        if "english" in languages:
            self.sw_en = stopwords.stopwords("en")
        if "spanish" in languages:
            self.sw_es = stopwords.stopwords("es")
        if "french" in languages:
            self.sw_fr = stopwords.stopwords("fr")
        if "ru" in languages:
            self.sw_ru = stopwords.stopwords("ru")
        if "german" in languages:
            self.sw_de = stopwords.stopwords("de")

        if not self.aligned_embeds:
            if "english" in languages:
                self.en_model = self.load_fasttext_model(language="en")
            if "spanish" in languages:
                self.es_model = self.load_fasttext_model(language="es")
            if "french" in languages:
                self.fr_model = self.load_fasttext_model(language="fr")
            if "ru" in languages:
                self.ru_model = self.load_fasttext_model(language="ru")
            if "german" in languages:
                self.de_model = self.load_fasttext_model(language="de")

        else:
            if "english" in languages:
                self.word2embed_en = self.read_aligned_vecs(
                    file_name="./data/embeddings/wiki.en.align.vec"
                )

            if "spanish" in languages:
                self.word2embed_es = self.read_aligned_vecs(
                    file_name="./data/embeddings/wiki.es.align.vec"
                )

            if "french" in languages:
                self.word2embed_fr = self.read_aligned_vecs(
                    file_name="./data/embeddings/wiki.fr.align.vec"
                )
            if "ru" in languages:
                self.word2embed_ru = self.read_aligned_vecs(
                    file_name="./data/embeddings/wiki.ru.align.vec"
                )
            if "german" in languages:
                self.word2embed_de = self.read_aligned_vecs(
                    file_name="./data/embeddings/wiki.de.align.vec"
                )

    def fit(self, X, y):
        """..."""
        return self

    def transform(self, X):
        """Transform documents into embeddings."""
        return self.create_embedding_matrices(X)

    def read_aligned_vecs(self, file_name):
        """Read embeddings from file into dict."""
        print(f"\nReading word vectors from file {file_name}.")

        make_sure_path_exists(file_name)

        word2embed = {}
        word2id = {}
        id2word = {}

        with open(file_name, "r") as read_handle:
            for i, line in enumerate(read_handle):
                if i == 0:
                    num_vecs = int(line.split()[0])
                    dim = int(line.split()[1])
                else:
                    data = line.split()
                    vec_list = []
                    token = ""
                    # very inconvenient and error-prone, but some tokens seem
                    # to be separated by space as well -- need to find position
                    # where vector starts
                    for t in data:
                        try:
                            vec_list.append(float(t))
                        except ValueError:
                            token = token + " " + t

                    token = token.strip()
                    embedding = np.array(vec_list)

                    # TODO: check why some vectors have dim (301, ) instead of
                    # (300, )
                    if np.shape(embedding)[0] == dim:
                        word2embed[token] = embedding
                    # if i == 1000:
                    #     break

        print(f"embedding dimension: {dim}")
        assert self.embedding_dim == dim, (
            "Given embedding dimension and " "loaded vector dimensions do not " "match."
        )
        print(f"#vectors: {num_vecs}")
        print(f"#words: {len(word2embed)}")
        print("     Done.")
        return word2embed

    def get_sent_embeddings(self, document, embeddings, stop_words):
        """For a document, average the embeddings of each token."""
        vecs = []
        counter = 0
        for token in document.split():
            if token in stop_words:
                continue
            try:
                vecs.append(embeddings[token])
            # ignore unseen tokens
            except KeyError:
                counter += 1
                pass

        # might be that there is no vector, use zeros
        if np.shape(vecs)[0] == 0:
            average = np.zeros((1, self.embedding_dim))
        else:
            average = np.average(vecs, axis=0)

            assert np.shape(average) == np.shape(vecs[0])

        return average

    def load_fasttext_model(self, language):
        """Load a fasttext model."""
        # can only download in working  directory
        # fasttext.util.download_model(language, if_exists='ignore')

        return fasttext.load_model(
            f"./data/embedding_models/cc.{language}." f"{self.embedding_dim}.bin"
        )

    def get_sent_embeds_from_model(self, document, model, stop_words):
        """..."""
        vecs = []
        for token in document.split():
            if token in stop_words:
                continue
            vecs.append(model.get_word_vector(token))

        # might be that there is no vector, use zeros
        if np.shape(vecs)[0] == 0:
            average = np.zeros((1, self.embedding_dim))
        else:
            average = np.average(vecs, axis=0)

            assert np.shape(average) == np.shape(vecs[0])

        return average

    def create_embedding_matrices(self, train_data):
        """Choose embedding model and calculate embeddings for each doc."""
        if self.aligned_embeds:

            X_train = np.zeros((len(train_data), self.embedding_dim))

            for i, doc in enumerate(train_data):
                if doc["language"] == "french":
                    sent_embed = self.get_sent_embeddings(
                        document=doc["text"],
                        embeddings=self.word2embed_fr,
                        stop_words=self.sw_fr,
                    )
                if doc["language"] == "spanish":
                    sent_embed = self.get_sent_embeddings(
                        document=doc["text"],
                        embeddings=self.word2embed_es,
                        stop_words=self.sw_es,
                    )
                if doc["language"] == "english":
                    sent_embed = self.get_sent_embeddings(
                        document=doc["text"],
                        embeddings=self.word2embed_en,
                        stop_words=self.sw_en,
                    )
                if doc["language"] == "ru":
                    sent_embed = self.get_sent_embeddings(
                        document=doc["text"],
                        embeddings=self.word2embed_ru,
                        stop_words=self.sw_ru,
                    )
                if doc["language"] == "german":
                    sent_embed = self.get_sent_embeddings(
                        document=doc["text"],
                        embeddings=self.word2embed_de,
                        stop_words=self.sw_de,
                    )

                X_train[i] = sent_embed

        else:
            X_train = np.zeros((len(train_data), self.embedding_dim))

            for i, doc in enumerate(train_data):
                if doc["language"] == "french":
                    sent_embed = self.get_sent_embeds_from_model(
                        document=doc["text"], model=self.fr_model, stop_words=self.sw_fr
                    )
                if doc["language"] == "spanish":
                    sent_embed = self.get_sent_embeds_from_model(
                        document=doc["text"], model=self.es_model, stop_words=self.sw_es
                    )
                if doc["language"] == "english":
                    sent_embed = self.get_sent_embeds_from_model(
                        document=doc["text"], model=self.en_model, stop_words=self.sw_en
                    )
                if doc["language"] == "ru":
                    sent_embed = self.get_sent_embeds_from_model(
                        document=doc["text"], model=self.ru_model, stop_words=self.sw_ru
                    )
                if doc["language"] == "german":
                    sent_embed = self.get_sent_embeds_from_model(
                        document=doc["text"], model=self.de_model, stop_words=self.sw_de
                    )

                X_train[i] = sent_embed

        return X_train
