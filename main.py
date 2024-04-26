import os
from pprint import pprint
import sys
from typing import List, Literal

from dotenv import load_dotenv

from documents import EmbeddingDocuments

load_dotenv()


def main(args: List[str]) -> Literal[0]:
    uri        = os.getenv("MONGODB_URI")
    database   = os.getenv("MONGODB_DATABASE")
    collection = "questions"
    model   = os.getenv("MODEL")
    
    print("[step] open connection and define model")
    documents = EmbeddingDocuments(uri, database, collection, model)

    print("[step] find documents")
    docs = [doc for doc in documents.find_all(
        filter  = { "embedding": { "$exists": False }},
        project = { "_id": 1, "comment": 1 })]
    
    print("[step] embedding documents")
    docs_update = documents.embedding_list(docs)

    print("[step] update embedded documents")
    res = documents.update_documents(docs_update)
    pprint(res.bulk_api_result if res != None else None)
    
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

