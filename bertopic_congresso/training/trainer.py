from time import time

import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic_congresso.utils.hidden_prints import HiddenPrints
from bertopic_congresso.utils.constants import PATHS


class Trainer:
    def __init__(self):
        self.model = None
        self.n_topics = None
        self.max_topics = 50
        self.top_n_words = 10

        self._model_path = PATHS.get_file_path(name="model")
        self._corpus_path = PATHS.get_file_path(name="corpus")
        self._topics_path = PATHS.get_file_path(name="topics")
        self._topic_words_path = PATHS.get_file_path(name="topic_words")
        self._embedder = None
        self._umap = None
        self._hdbscan = None
        self._vectorizer = None
        self._ctfidf = None

    def setup_umap(self, parameters={}):
        self._umap = UMAP(**parameters)

    def setup_hdbscan(self, parameters={}):
        self._hdbscan = HDBSCAN(**parameters)

    def setup_vectorizer(self, parameters={}):
        self._vectorizer = CountVectorizer(**parameters)

    def setup_ctfidf(self, parameters={}):
        self._ctfidf = ClassTfidfTransformer(**parameters)

    def create_model(self, max_topics=50, top_n_words=10, embedding_model=None):
        self.max_topics = max_topics
        self.top_n_words = top_n_words
        self._embedder = embedding_model

        self.model = BERTopic(
            top_n_words=top_n_words,
            embedding_model=self._embedder,
            umap_model=self._umap,
            hdbscan_model=self._hdbscan,
            vectorizer_model=self._vectorizer,
            ctfidf_model=self._ctfidf,
            verbose=False,
        )

    def run(self, documents, embeddings):
        print("Iniciando treinamento do modelo...")
        topics, training_time = self._train(documents=documents, embeddings=embeddings)
        print(f"Treinamento concluído em {round(training_time)} segundos!\n")

        print("Executando tuning do modelo...")
        outliers = [topic for topic in topics if topic == -1]
        p_outliers = round(100 * len(outliers) / len(topics), 2)
        print(f"\tInício: {self.n_topics} tópicos e {p_outliers}% de outliers.")

        topics, tuning_time = self._tune(documents=documents, p_outliers=p_outliers)

        outliers = [topic for topic in topics if topic == -1]
        p_outliers = round(100 * len(outliers) / len(topics), 2)
        print(f"\tFim: {self.n_topics} tópicos e {p_outliers}% de outliers.")
        print(f"Tuning concluído em {round(tuning_time)} segundos!\n")

        print("Persistindo o modelo e seus resultados...")
        self._store(documents=documents)
        print("Concluído!")

    def _train(self, documents, embeddings):
        start_time = time()
        with HiddenPrints():
            topics, _ = self.model.fit_transform(
                documents=documents,
                embeddings=embeddings,
            )

        end_time = time()
        self.n_topics = len(set(topics)) - 1
        training_time = float(end_time - start_time)

        return topics, training_time

    def _tune(self, documents, p_outliers):
        start_time = time()
        if self.n_topics >= self.max_topics:
            self.model.reduce_topics(
                docs=documents,
                nr_topics=self.max_topics + 1,
            )

        if p_outliers > 0.0:
            topics = self.model.topics_
            new_topics = self.model.reduce_outliers(
                documents=documents,
                topics=topics,
            )

            self.model.update_topics(
                docs=documents,
                topics=new_topics,
                top_n_words=self.top_n_words,
                vectorizer_model=self._vectorizer,
                ctfidf_model=self._ctfidf,
            )

        end_time = time()
        topics = self.model.topics_
        self.n_topics = len(set(topics)) - 1
        tuning_time = float(end_time - start_time)

        return topics, tuning_time

    def _store(self, documents):
        topics = self.model.get_topics()
        topics_words = []

        for topic_id, words in topics.items():
            topic_info = {"topico": topic_id}
            for index, word in enumerate(words):
                topic_info[f"termo_{index + 1}"] = word[0]
            topics_words.append(topic_info)

        topics_words = pd.DataFrame(topics_words)
        topics_words.to_csv(self._topic_words_path, index=False)

        topic_per_document = pd.read_csv(self._corpus_path)
        documents_info = self.model.get_document_info(docs=documents)

        topic_per_document["topico"] = documents_info["Topic"]
        topic_per_document["representativo"] = documents_info["Representative_document"]
        topic_per_document = topic_per_document.drop("texto", axis=1)

        topic_per_document.to_csv(self._topics_path, index=False)

        with HiddenPrints():
            self.model.save(path=self._model_path)
