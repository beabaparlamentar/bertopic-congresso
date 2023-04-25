import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PathsNamespace:
    FILE_PATHS = {
        "corpus": os.path.join("data", "corpus.csv"),
        "corpus_dump": os.path.join("corpus_dump.txt"),
        "embeddings": os.path.join("data", "embeddings.npy"),
        "evaluation": os.path.join("data", "model_evaluation.csv"),
        "topics": os.path.join("data", "topics.csv"),
        "topic_words": os.path.join("data", "topic_words.csv"),
        "model": os.path.join("bertopic_model"),
    }
    FOLDER_PATHS = {"corpus": os.path.join("data", "corpus")}

    def get_file_path(self, name):
        return self.FILE_PATHS.get(name, "")

    def get_folder_path(self, name):
        return self.FOLDER_PATHS.get(name, "")


PATHS = PathsNamespace()
