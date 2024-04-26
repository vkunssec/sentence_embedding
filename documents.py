import string
from typing import List

from pymongo import cursor, UpdateOne
from pymongo.results import BulkWriteResult
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from connect import Connection
from normalize_map import normalize_map


class EmbeddingDocuments:
    def __init__(self, uri: str, database: str, collection: str, model: str) -> None:
        self.connection = Connection().open(uri)
        self.database = self.connection[database]
        self.collection = self.database[collection]
        
        self.model = SentenceTransformer(model)


    def find_all(self, filter: dict, project: dict) -> cursor:
        return self.collection.find(filter, project)


    def update_documents(self, docs) -> BulkWriteResult:
        if len(docs) == 0:
            return None
        bulk = [
            UpdateOne(
                { "_id": doc["_id"] },
                { "$set": {
                    "embedding": doc["embedding"]
                }}
            ) for doc in tqdm(docs)
        ]
        return self.collection.bulk_write(bulk)


    def normalize(self, sentence: str) -> str:
        sentence = sentence.lower()
        sentence = sentence.strip()

        # remove accents
        sentence = sentence.translate(str.maketrans(normalize_map))

        # remove punctuation
        sentence = sentence.translate(sentence.maketrans("", "", string.punctuation)) 

        return sentence


    def embedding_string(self, comment: str) -> List[any]:
        return self.model.encode(self.normalize(comment))


    def embedding_list(self, docs: List[dict[str, any]]) -> List[dict[str, any]]:
        return [
            dict(doc, **{
                'embedding': self.embedding_string(doc["comment"]).tolist()
            }
        ) for doc in tqdm(docs)]

