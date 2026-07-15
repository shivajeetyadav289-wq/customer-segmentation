import os
import yaml
import pandas as pd

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact
)


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

    def validate_data(self):

        validation_status = True

        train_path = self.data_ingestion_artifact.train_file_path
        test_path = self.data_ingestion_artifact.test_file_path

        # Check whether train and test files exist
        if not os.path.exists(train_path):
            validation_status = False

        if not os.path.exists(test_path):
            validation_status = False

        if not validation_status:
            raise FileNotFoundError("Train or Test file not found.")

        # Read CSV files
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print("Train Shape :", train_df.shape)
        print("Test Shape  :", test_df.shape)

        # Required columns
        required_columns = [
            "ID",
            "Year_Birth",
            "Education",
            "Marital_Status",
            "Income",
            "Kidhome",
            "Teenhome",
            "Dt_Customer",
            "Recency",
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumWebVisitsMonth",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
            "AcceptedCmp1",
            "AcceptedCmp2",
            "Complain",
            "Z_CostContact",
            "Z_Revenue",
            "Response"
        ]

        # Check missing columns
        missing_columns = []

        for column in required_columns:
            if column not in train_df.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:
            validation_status = False

        # Missing values
        missing_values = train_df.isnull().sum().to_dict()

        # Duplicate rows
        duplicate_rows = train_df.duplicated().sum()

        # Create validation report directory
        os.makedirs(
            self.data_validation_config.data_validation_dir,
            exist_ok=True
        )

        # Validation report
        report = {
            "validation_status": validation_status,
            "missing_columns": missing_columns,
            "missing_values": missing_values,
            "duplicate_rows": int(duplicate_rows)
        }

        # Save report
        with open(
            self.data_validation_config.report_file_path,
            "w"
        ) as file:
            yaml.dump(report, file)

        print("Validation Report Saved Successfully")

        # Return Artifact
        data_validation_artifact = DataValidationArtifact(
            validation_status=validation_status,
            report_file_path=self.data_validation_config.report_file_path
        )

        return data_validation_artifact
        
