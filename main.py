import os
from pprint import pprint
import sys
from typing import List, Literal

from dotenv import load_dotenv

from documents import EmbeddingDocuments

load_dotenv()


def main(_: List[str]) -> Literal[0]:
    uri = os.getenv("MONGODB_URI")
    database = os.getenv("MONGODB_DATABASE")
    collection = "questions"
    model = os.getenv("MODEL")

    print("[step] open connection and define model")
    documents = EmbeddingDocuments(uri, database, collection, model)

    print("[step] find documents")
    docs = [doc for doc in documents.find_all(
        # filter  = { "embedding": { "$exists": False }},
        filter = {},
        project = { "_id": 1, "comment": 1 })]

    print("[step] embedding documents")
    docs_update = documents.embedding_list(docs)

    print("[step] normalize documents")
    docs_normalized = documents.normalize_embedding(docs_update)

    print("[step] update embedded documents")
    res = documents.update_documents(docs_normalized)
    pprint(res.bulk_api_result if res != None else None)

    # print("[step] clustering")
    # docs_clustering = documents.clustering([doc["embedding"] for doc in docs_normalized])

    # print("[step] grouping cluster")
    # docs_grouping = documents.group_cluster(docs, docs_clustering)
    # pprint(docs_grouping, width=180)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

