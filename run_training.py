import nltk
import pandas as pd
from spacy.lang.pt.stop_words import STOP_WORDS

from bertopic_congresso.training import Trainer
from bertopic_congresso.utils import Embeddings, HiddenPrints
from bertopic_congresso.utils.constants import PATHS


def get_stopwords():
    with HiddenPrints():
        nltk.download("stopwords")

    nltk_stopwords = nltk.corpus.stopwords.words("portuguese")
    spacy_stopwords = STOP_WORDS

    stopwords = set()
    stopwords.update(nltk_stopwords)
    stopwords.update(spacy_stopwords)
    stopwords = list(stopwords)

    print(f"Serão utilizadas {len(stopwords)} stopwords neste experimento.\n")

    return stopwords


def get_documents(corpus_path):
    corpus = pd.read_csv(corpus_path)
    documents = corpus["texto"].tolist()

    return documents


def create_embeddings(documents):
    embedder = Embeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedder.create(documents=documents, overwrite=True)
    embedding_model = embedder.get_model()

    return embeddings, embedding_model


def create_model(embedding_model, parameters):
    model_trainer = Trainer()

    model_trainer.setup_umap(parameters["umap"])
    model_trainer.setup_hdbscan(parameters["hdbscan"])
    model_trainer.setup_vectorizer(parameters["vectorizer"])
    model_trainer.setup_ctfidf(parameters["ctfidf"])

    model_trainer.create_model(
        max_topics=50,
        top_n_words=10,
        embedding_model=embedding_model,
    )

    return model_trainer


if __name__ == "__main__":
    corpus_path = PATHS.get_file_path(name="corpus")

    stopwords = get_stopwords()
    documents = get_documents(corpus_path=corpus_path)
    embeddings, embedding_model = create_embeddings(documents=documents)

    parameters = {
        "umap": {
            "n_components": 15,
            "n_neighbors": 50,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
            "low_memory": False,
        },
        "hdbscan": {
            "min_cluster_size": 100,
            "metric": "manhattan",
        },
        "vectorizer": {
            "ngram_range": (1, 1),
            "strip_accents": False,
            "stop_words": stopwords,
            "token_pattern": "([a-zà-úA-ZÀ-Ú][a-zà-úA-ZÀ-Ú]+)",
        },
        "ctfidf": {"reduce_frequent_words": True},
    }

    model = create_model(embedding_model=embedding_model, parameters=parameters)
    model.run(documents=documents, embeddings=embeddings)
