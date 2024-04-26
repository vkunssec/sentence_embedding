import json
import os

from dotenv import load_dotenv

from documents import EmbeddingDocuments

load_dotenv()


uri = os.getenv("MONGODB_URI")
db = os.getenv("MONGODB_DATABASE")
col = "questions"

model = os.getenv("MODEL")

documents = EmbeddingDocuments(uri, db, col, model)

query = documents.embedding_string('o comentário que você quer comparar').tolist()

pipeline = []
pipeline.append({
    '$vectorSearch': {
        'queryVector': query,
        'path': 'embedding',
        'numCandidates': 100,
        'limit': 5,
        'index': 'vector_index',
    }
})
pipeline.append({
    '$project': {
        '_id': 0,
        'id': { '$toString': '$_id' },
        'comment': 1,
        # 'embedding': 1,
    }
})

docs = documents.collection.aggregate(pipeline=pipeline)
print(json.dumps(list(docs), indent=4, ensure_ascii=False))

