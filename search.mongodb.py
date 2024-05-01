import argparse
import json
import os
import sys

from dotenv import load_dotenv

from documents import EmbeddingDocuments

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", dest="query", help="Query string")


def main(_):
    args = parser.parse_args()

    uri = os.getenv("MONGODB_URI")
    db = os.getenv("MONGODB_DATABASE")
    col = "questions"

    model = os.getenv("MODEL")

    documents = EmbeddingDocuments(uri, db, col, model)

    q = args.query if args.query != None else "campo que será comparado"
    print("Comparação feito a partir da Query: ", q)
    query = documents.embedding_string(q)
    print("Query normalizada: ", query[1])
    query = query[0].tolist()

    pipeline = []
    pipeline.append({
        '$vectorSearch': {
            'queryVector': query,
            'path': 'embedding',
            # num_candidates <= 10_000
            #                > limit
            #                ~ (( 10 * limit ) <= num_candidate < ( 20 * limit ))
            'numCandidates': 100,
            'limit': 10,
            'index': 'vector_index',
        }
    })
    pipeline.append({
        '$project': {
            '_id': 0,
            'id': { '$toString': '$_id' },
            'comment': 1,
            'score': { "$meta": "vectorSearchScore" }
        }
    })

    docs = documents.collection.aggregate(pipeline=pipeline)
    print(json.dumps(list(docs), indent=4, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

