import pandas as pd

from bertopic_congresso.training import Trainer
from bertopic_congresso.utils import Embeddings
from bertopic_congresso.utils.constants import PATHS

PARAMETERS = {
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
        "stop_words": None,
        "token_pattern": "([a-zà-úA-ZÀ-Ú][a-zà-úA-ZÀ-Ú]+)",
    },
    "ctfidf": {"reduce_frequent_words": True},
}


def get_documents(corpus_path):
    corpus = pd.read_csv(corpus_path)
    documents = corpus["texto"].tolist()

    return documents


def create_embeddings(documents):
    embedder = Embeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedder.create(documents=documents, overwrite=True)
    embedding_model = embedder.get_model()

    return embeddings, embedding_model


def create_model(embedding_model):
    model_trainer = Trainer()

    model_trainer.setup_umap(PARAMETERS["umap"])
    model_trainer.setup_hdbscan(PARAMETERS["hdbscan"])
    model_trainer.setup_vectorizer(PARAMETERS["vectorizer"])
    model_trainer.setup_ctfidf(PARAMETERS["ctfidf"])

    model_trainer.create_model(
        max_topics=50,
        top_n_words=10,
        embedding_model=embedding_model,
    )

    return model_trainer


if __name__ == "__main__":
    corpus_path = PATHS.get_file_path(name="corpus")
    documents = get_documents(corpus_path=corpus_path)
    embeddings, embedding_model = create_embeddings(documents=documents)

    model = create_model(embedding_model=embedding_model)
    model.run(documents=documents, embeddings=embeddings)
