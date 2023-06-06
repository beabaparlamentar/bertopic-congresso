import os

from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing

from bertopic_congresso.utils.hidden_prints import HiddenPrints
from bertopic_congresso.utils.constants import PATHS


class Loader:
    def __init__(self):
        self.data = Dataset()
        self.documents = None

        self._corpus_folder = PATHS.get_folder_path(name="corpus")
        self._dump_path = PATHS.get_file_path(name="corpus_dump")
        self._preprocessor = Preprocessing(
            lemmatize=False,
            lowercase=False,
            remove_numbers=False,
            remove_punctuation=False,
            remove_stopwords_spacy=False,
            split=False,
        )

    def get_data(self):
        return self.data

    def get_documents(self):
        return self.documents

    def create_documents(self, documents):
        print("Carregando documentos...")

        with HiddenPrints():
            self._setup_corpus_files(documents=documents)
            self.data.load_custom_dataset_from_folder(path=self._corpus_folder)

        print("Os documentos foram carregados!\n")

        corpus = self.data.get_corpus()
        self.documents = [" ".join(document) for document in corpus]

        return self.data

    def _setup_corpus_files(self, documents):
        if os.path.exists(self._corpus_folder):
            for filename in os.listdir(self._corpus_folder):
                filepath = os.path.join(self._corpus_folder, filename)
                os.remove(filepath)

        content = "\n".join(documents)
        with open(self._dump_path, mode="wt", encoding="utf-8") as file:
            file.write(content)

        dataset = self._preprocessor.preprocess_dataset(documents_path=self._dump_path)
        dataset.save(path=self._corpus_folder)
        os.remove(self._dump_path)
