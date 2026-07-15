import os
from pandas import DataFrame

from src.data_access.customer_data import CustomerData
from src.entity.config_entity import DataIngestionConfig
from src.constant.database import DATABASE_NAME, COLLECTION_NAME
from sklearn.model_selection import train_test_split
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self) -> DataFrame:

        print("========== DATA INGESTION STARTED ==========")

        customer_data = CustomerData()

        dataframe = customer_data.export_collection_as_dataframe(
            database_name=DATABASE_NAME,
            collection_name=COLLECTION_NAME
        )

        print("Data fetched successfully")
        print("Shape:", dataframe.shape)

        print("Directory to create:")
        print(self.data_ingestion_config.data_ingestion_dir)

        os.makedirs(
            self.data_ingestion_config.data_ingestion_dir,
            exist_ok=True
        )

        print("Directory created")

        print("Saving file at:")
        print(self.data_ingestion_config.raw_file_path)

        dataframe.to_csv(
            self.data_ingestion_config.raw_file_path,
            index=False
        )

        print("raw.csv saved successfully")

        return dataframe
    
    # train and test split
    def split_data_as_train_test(self, dataframe: DataFrame):

        train_set, test_set = train_test_split(
            dataframe,
            test_size=0.2,
            random_state=42
        )

        train_set.to_csv(
            self.data_ingestion_config.train_file_path,
            index=False
        )

        test_set.to_csv(
            self.data_ingestion_config.test_file_path,
            index=False
        )

        print("train.csv saved")
        print("test.csv saved")
        
    def initiate_data_ingestion(self):

        dataframe = self.export_data_into_feature_store()

        self.split_data_as_train_test(dataframe)

        data_ingestion_artifact = DataIngestionArtifact(
            train_file_path=self.data_ingestion_config.train_file_path,
            test_file_path=self.data_ingestion_config.test_file_path
        )

        print("Data Ingestion Completed")

        return data_ingestion_artifact
