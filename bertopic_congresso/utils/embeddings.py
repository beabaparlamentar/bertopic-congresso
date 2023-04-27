import os

import numpy as np
from sentence_transformers import SentenceTransformer

from bertopic_congresso.utils.constants import PATHS


class Embeddings:
    def __init__(self, model_name):
        self.model_name = model_name

        self._embeddings_path = PATHS.get_file_path(name="embeddings")
        self._model = SentenceTransformer(model_name)

    def get_model(self):
        return self._model

    def create(self, documents, overwrite=False):
        if overwrite and os.path.exists(self._embeddings_path):
            os.remove(self._embeddings_path)

        if os.path.exists(self._embeddings_path):
            embeddings = np.load(file=self._embeddings_path)
            print("Os embeddings foram carregados com sucesso!\n")
        else:
            print("Criando embeddings...")
            embeddings = self._model.encode(
                sentences=documents,
                show_progress_bar=True,
            )

            print(f"Embeddings armazenados em '{self._embeddings_path}'!\n")
            np.save(file=self._embeddings_path, arr=embeddings)

        return embeddings
