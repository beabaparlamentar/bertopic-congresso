from time import time

import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.model_selection import ParameterGrid
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

from bertopic_congresso.utils.hidden_prints import HiddenPrints
from bertopic_congresso.utils.constants import PATHS


class Evaluator:
    def __init__(self, dataset, embeddings, top_n_words=10, parameters={}):
        self.dataset = dataset
        self.embeddings = embeddings
        self.top_n_words = top_n_words
        self.parameters = parameters

        corpus = self.dataset.get_corpus()
        npmi = Coherence(texts=corpus, topk=self.top_n_words, measure="c_npmi")
        diversity = TopicDiversity(topk=self.top_n_words)

        self._evaluation_path = PATHS.get_file_path(name="evaluation")
        self._metrics = {"npmi": npmi, "diversity": diversity}

    def run(self):
        parameter_grid = ParameterGrid(self.parameters)
        n_models = len(parameter_grid)
        evaluation = []

        for i, model_parameters in enumerate(parameter_grid):
            print(f"Executando treinamento do modelo {i + 1} de {n_models}:")
            model = self._create_model(model_parameters=model_parameters)
            print("\tTreinando modelo...")
            model_output, training_time = self._train_model(model=model)

            print("\tAvaliando modelo...")
            model_results = self._evaluate_model(model_output=model_output)
            model_results["time"] = round(training_time, 2)

            for parameter, value in model_parameters.items():
                if parameter == "vectorizer_stopwords":
                    model_results[parameter] = value is not None
                else:
                    model_results[parameter] = str(value)

            evaluation.append(model_results)
            print("\tConcluído!\n")

        evaluation = pd.DataFrame(evaluation)
        evaluation.to_csv(self._evaluation_path, index=False)
        print("O treinamento dos modelos foi concluído!")

        return evaluation

    def _create_model(self, model_parameters):
        umap_parameters = {
            "low_memory": model_parameters.get("umap_low_memory", True),
            "metric": model_parameters.get("umap_metric", "cosine"),
            "n_components": model_parameters.get("umap_n_components", 10),
            "n_neighbors": model_parameters.get("umap_n_neighbors", 15),
            "min_dist": 0.0,
            "random_state": 42,
        }

        hdbscan_parameters = {
            "metric": model_parameters.get("hdbscan_metric", "euclidean"),
            "min_cluster_size": model_parameters.get("hdbscan_min_cluster_size", 5),
        }

        vectorizer_parameters = {
            "max_df": model_parameters.get("vectorizer_max_df", 1.0),
            "min_df": model_parameters.get("vectorizer_min_df", 0.0),
            "ngram_range": model_parameters.get("vectorizer_ngram_range", (1, 1)),
            "stop_words": model_parameters.get("vectorizer_stopwords", None),
            "token_pattern": "([a-zà-úA-ZÀ-Ú][a-zà-úA-ZÀ-Ú]+)",
            "strip_accents": False,
        }

        bertopic_parameters = {
            "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True),
            "hdbscan_model": HDBSCAN(**hdbscan_parameters),
            "umap_model": UMAP(**umap_parameters),
            "vectorizer_model": CountVectorizer(**vectorizer_parameters),
            "calculate_probabilities": False,
            "top_n_words": self.top_n_words,
            "nr_topics": "auto",
            "verbose": False,
        }

        return BERTopic(**bertopic_parameters)

    def _train_model(self, model):
        corpus = self.dataset.get_corpus()
        vocabulary = self.dataset.get_vocabulary()
        documents = [" ".join(document) for document in corpus]

        start_time = time()
        with HiddenPrints():
            topics, _ = model.fit_transform(
                documents=documents,
                embeddings=self.embeddings,
            )
        end_time = time()

        n_topics = len(set(topics)) - 1
        words_per_topic = []

        for i in range(n_topics):
            words = [word[0] for word in model.get_topic(i)[: self.top_n_words]]
            words = [w if w in vocabulary else vocabulary[0] for w in words]
            words_per_topic.append(words)

        model_output = {"topics": words_per_topic}
        training_time = float(end_time - start_time)

        return model_output, training_time

    def _evaluate_model(self, model_output):
        metric_scores = {}

        for metric, scorer in self._metrics.items():
            score = scorer.score(model_output)
            metric_scores[metric] = float(score)

        return metric_scores
