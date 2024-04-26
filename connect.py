import pymongo


class Connection:
    def open(self, uri: str):
        return pymongo.MongoClient(uri)

