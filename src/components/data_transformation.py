import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



from src.entity.config_entity import (
    DataTransformationConfig
)

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact
)

class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig
    ):

        self.data_ingestion_artifact = data_ingestion_artifact

        self.data_transformation_config = data_transformation_config
        
        
    # Feature Engineering
    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the customer dataset.
        """

        data = data.copy()

        # -----------------------------
        # Age
        # -----------------------------
        current_year = datetime.today().year
        data["Age"] = current_year - data["Year_Birth"]

        # -----------------------------
        # Education Encoding
        # -----------------------------
        education_mapping = {
            "Basic": 0,
            "2n Cycle": 1,
            "Graduation": 2,
            "Master": 3,
            "PhD": 4
        }

        data["Education"] = data["Education"].map(education_mapping)

        # -----------------------------
        # Marital Status Encoding
        # -----------------------------
        marital_mapping = {
            "Married": 1,
            "Together": 1,
            "Absurd": 0,
            "Widow": 0,
            "YOLO": 0,
            "Divorced": 0,
            "Single": 0,
            "Alone": 0
        }

        data["Marital_Status"] = data["Marital_Status"].map(
            marital_mapping
        )

        # -----------------------------
        # Children
        # -----------------------------
        data["Children"] = (
            data["Kidhome"] +
            data["Teenhome"]
        )

        # -----------------------------
        # Family Size
        # -----------------------------
        data["Family_Size"] = (
            data["Marital_Status"] +
            data["Children"] +
            1
        )

        # -----------------------------
        # Total Spending
        # -----------------------------
        data["Total_Spending"] = (
            data["MntWines"] +
            data["MntFruits"] +
            data["MntMeatProducts"] +
            data["MntFishProducts"] +
            data["MntSweetProducts"] +
            data["MntGoldProds"]
        )

        # -----------------------------
        # Total Promotions Accepted
        # -----------------------------
        data["Total Promo"] = (
            data["AcceptedCmp1"] +
            data["AcceptedCmp2"] +
            data["AcceptedCmp3"] +
            data["AcceptedCmp4"] +
            data["AcceptedCmp5"]
        )

        # -----------------------------
        # Customer Since
        # -----------------------------
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"],format="%d-%m-%Y")

        today = datetime.today()
        data["Days_as_Customer"] = (
            today - data["Dt_Customer"]
        ).dt.days

        # -----------------------------
        # Offers Responded To
        # -----------------------------
        data["Offers_Responded_To"] = (
            data["AcceptedCmp1"] +
            data["AcceptedCmp2"] +
            data["AcceptedCmp3"] +
            data["AcceptedCmp4"] +
            data["AcceptedCmp5"] +
            data["Response"]
        )

        # -----------------------------
        # Parental Status
        # -----------------------------
        data["Parental Status"] = np.where(
            data["Children"] > 0,
            1,
            0
        )

        # -----------------------------
        # Drop Columns
        # -----------------------------
        data.drop(
            columns=[
                "Year_Birth",
                "Kidhome",
                "Teenhome"
            ],
            inplace=True
        )

        # -----------------------------
        # Rename Columns
        # -----------------------------
        data.rename(
            columns={
                "Marital_Status": "Marital Status",
                "MntWines": "Wines",
                "MntFruits": "Fruits",
                "MntMeatProducts": "Meat",
                "MntFishProducts": "Fish",
                "MntSweetProducts": "Sweets",
                "MntGoldProds": "Gold",
                "NumWebPurchases": "Web",
                "NumCatalogPurchases": "Catalog",
                "NumStorePurchases": "Store",
                "NumDealsPurchases": "Discount Purchases"
            },
            inplace=True
        )

        # -----------------------------
        # Final Feature Selection
        # -----------------------------
        data = data[
            [
                "Age",
                "Income",
                "Days_as_Customer",
                "Recency",
                "Wines",
                "Fruits",
                "Meat",
                "Fish",
                "Sweets",
                "Gold",
                "Web",
                "Catalog",
                "Store",
                "Discount Purchases",
                "NumWebVisitsMonth"
            ]
        ]

        return data
    # Data preprocessing
    
    def get_data_transformer_object(self):

        numerical_columns = [
            "Age",
            "Income",
            "Days_as_Customer",
            "Recency",
            "Wines",
            "Fruits",
            "Meat",
            "Fish",
            "Sweets",
            "Gold",
            "Web",
            "Catalog",
            "Store",
            "Discount Purchases",
            "NumWebVisitsMonth"
        ]

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_columns)
            ]
        )

        return preprocessor
    
    # read the train and test data
    def initiate_data_transformation(self):

        train_df = pd.read_csv(
            self.data_ingestion_artifact.train_file_path
        )

        test_df = pd.read_csv(
            self.data_ingestion_artifact.test_file_path
        )

        print("Train Shape:", train_df.shape)
        print("Test Shape:", test_df.shape)
        
        train_df = self.feature_engineering(train_df)

        test_df = self.feature_engineering(test_df)

        # print(train_df.head())
        train_df.to_csv(
            self.data_transformation_config.engineered_train_file_path,
            index=False
        ) 

        test_df.to_csv(
            self.data_transformation_config.engineered_test_file_path,
            index=False
        )

        print("Engineered train.csv saved")
        print("Engineered test.csv saved")
                
        preprocessor = self.get_data_transformer_object()
        
        # Apply transformation
        train_arr = preprocessor.fit_transform(train_df)
        test_arr = preprocessor.transform(test_df)

        print(train_arr.shape)
        print(test_arr.shape)

        # Create artifacts directory
        os.makedirs(
            self.data_transformation_config.data_transformation_dir,
            exist_ok=True
        )

        # Save preprocessor
        joblib.dump(
            preprocessor,
            self.data_transformation_config.preprocessor_object_file_path
        )

        print("Preprocessor saved successfully")

        # Save transformed train array
        np.save(
            self.data_transformation_config.transformed_train_file_path,
            train_arr
        )

        # Save transformed test array
        np.save(
            self.data_transformation_config.transformed_test_file_path,
            test_arr
        )

        print("Transformed train.npy saved")
        print("Transformed test.npy saved")

        # Return artifact
        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path= self.data_transformation_config.transformed_test_file_path,
            engineered_train_file_path= self.data_transformation_config.engineered_train_file_path,
            engineered_test_file_path= self.data_transformation_config.engineered_test_file_path,
            preprocessor_object_file_path= self.data_transformation_config.preprocessor_object_file_path
        )

        return data_transformation_artifact