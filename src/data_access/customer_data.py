import pandas as pd
from pymongo import MongoClient


class CustomerData:

    def __init__(self):

        self.client = MongoClient(
            "mongodb://localhost:27017/"
        )


    def export_collection_as_dataframe(
        self,
        database_name: str,
        collection_name: str
    ) -> pd.DataFrame:

        database = self.client[database_name]

        collection = database[collection_name]

        records = list(collection.find())

        dataframe = pd.DataFrame(records)

        if "_id" in dataframe.columns:
            dataframe.drop(
                columns=["_id"],
                inplace=True
            )

        return dataframe
