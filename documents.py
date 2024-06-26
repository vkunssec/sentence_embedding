from collections import Counter
from typing import List

from joblib import Parallel, delayed
import numpy as np
from pymongo import cursor, UpdateOne
from pymongo.results import BulkWriteResult
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

from connect import Connection
from utils import Utils


class EmbeddingDocuments:
    def __init__(self, uri: str, database: str, collection: str, model: str) -> None:
        self.connection = Connection().open(uri)
        self.database = self.connection[database]
        self.collection = self.database[collection]

        self.model = SentenceTransformer(model, device="cuda")
        self.cluster_model = None


    def find_all(self, filter: dict, project: dict) -> cursor:
        return self.collection.find(filter, project)


    def update_documents(self, docs) -> BulkWriteResult:
        if len(docs) == 0:
            return None
        bulk = [
            UpdateOne(
                { "_id": doc["_id"] },
                { "$set": {
                    "embedding": doc["embedding"].tolist(),
                    "comment_normalized": doc["comment_normalized"],
                }}
            ) for doc in tqdm(docs)
        ]
        return self.collection.bulk_write(bulk)


    def normalize(self, sentence: str) -> str:
        return Utils().normalize(sentence)


    def embedding_string(self, comment: str) -> tuple:
        comment_normalized = " ".join(Counter(self.normalize(comment).split()).keys())
        embedding = self.model.encode(comment_normalized, convert_to_tensor=True).cpu().data.numpy()
        return (embedding, comment_normalized)


    def embedding_list(self, docs: List[dict[str, any]]) -> List[dict[str, any]]:
        embeddings = Parallel(n_jobs=6, prefer="threads")(delayed(self.embedding_string)(doc["comment"]) for doc in tqdm(docs))
        return [ dict(docs[i], **{
                'embedding': embeddings[i][0],
                'comment_normalized': embeddings[i][1],
            }
        ) for i in tqdm(range(len(docs))) ]


    def normalize_embedding(self, docs: List[dict[str, any]]) -> List[dict[str, any]]:
        docs_embeddings = []
        embedding = [ doc["embedding"] for doc in tqdm(docs) ]
        embeddings = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        i = 0
        for doc in tqdm(docs):
            docs_embeddings.append({
                "_id": doc["_id"],
                "comment": doc["comment"],
                "embedding": embeddings[i],
                "comment_normalized": doc["comment_normalized"]
            })
            i += 1
        return docs_embeddings


    def clustering(self, corpus_embedding: List) -> List:
        model = KMeans(n_clusters=1000)
        model.fit(corpus_embedding)
        self.cluster_model = model
        return model.labels_


    def group_cluster(self, docs: List, cluster: List) -> List:
        clustered = {}
        for sentence_id, cluster_id in enumerate(cluster):
            if cluster_id not in clustered:
                clustered[cluster_id] = []
            clustered[cluster_id].append(docs[sentence_id])
        return clustered

