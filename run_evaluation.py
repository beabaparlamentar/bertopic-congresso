import nltk
import pandas as pd
from spacy.lang.pt.stop_words import STOP_WORDS

from bertopic_congresso.training import Evaluator
from bertopic_congresso.utils import Loader, Embeddings, HiddenPrints
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


def get_texts(corpus_path):
    corpus = pd.read_csv(corpus_path)
    texts = corpus["texto"].tolist()

    return texts


def create_embeddings(documents):
    embedder = Embeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedder.create(documents=documents, overwrite=True)

    return embeddings


if __name__ == "__main__":
    loader = Loader()
    corpus_path = PATHS.get_file_path(name="corpus")

    stopwords = get_stopwords()
    texts = get_texts(corpus_path=corpus_path)
    dataset = loader.create_documents(texts=texts)
    embeddings = create_embeddings(documents=loader.get_documents())

    test_parameters = {
        "hdbscan_metric": ["manhattan", "euclidean"],
        "hdbscan_min_cluster_size": [100, 250, 500],
        "vectorizer_ngram_range": [(1, 1)],
        "vectorizer_stopwords": [stopwords],
        "umap_low_memory": [False],
        "umap_metric": ["cosine"],
        "umap_n_components": [5, 10, 15],
        "umap_n_neighbors": [25, 50, 100],
    }

    model_evaluator = Evaluator(
        dataset=dataset,
        embeddings=embeddings,
        parameters=test_parameters,
        top_n_words=10,
    )

    model_evaluator.run()
