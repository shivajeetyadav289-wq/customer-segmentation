import joblib
import pandas as pd
import os
from src.entity.config_entity import PredictionConfig
from src.entity.artifact_entity import PredictionArtifact

class PredictionPipeline:
    
    def __init__(self):
        self.prediction_config = PredictionConfig()
    
    preprocessor = joblib.load(
    "artifacts/data_transformation/preprocessor.pkl"
    )
    
    model = joblib.load(
    "artifacts/model_trainer/kmeans_model.pkl"
    )
    
    def predict(self, dataframe):

        preprocessor = joblib.load(
            "artifacts/data_transformation/preprocessor.pkl"
        )

        model = joblib.load(
            "artifacts/model_trainer/kmeans_model.pkl"
        )

        transformed_data = preprocessor.transform(
            dataframe
        )

        prediction = model.predict(
            transformed_data
        )
        cluster_info = {
            0: {
                "name": "Budget Customers",
                "recommendation": "Offer discounts, coupons, and promotional campaigns."
            },
            1: {
                "name": "High Value Customers",
                "recommendation": "Promote premium products, loyalty programs, and exclusive offers."
            }
        }
        
        cluster = int(prediction[0])
        
        prediction_result = {
            "cluster": cluster,
            "segment": cluster_info[cluster]["name"],
            "recommendation": cluster_info[cluster]["recommendation"]
        }
        # save the prediction report
        os.makedirs(
            self.prediction_config.prediction_dir,
            exist_ok=True
        )
        with open(
            self.prediction_config.prediction_report_file_path,
            "a"
        ) as file:

            file.write("CUSTOMER SEGMENTATION PREDICTION REPORT\n")
            file.write("=" * 45 + "\n\n")

            file.write("Customer Details\n")
            file.write("-" * 20 + "\n")
            file.write(dataframe.to_string(index=False))
            file.write("\n\n")

            file.write("Prediction Result\n")
            file.write("-" * 20 + "\n")
            file.write(f"Predicted Cluster : {prediction_result['cluster']}\n")
            file.write(f"Customer Segment  : {prediction_result['segment']}\n")
            file.write(f"Recommendation    : {prediction_result['recommendation']}\n")
                
        prediction_artifact = PredictionArtifact(
            prediction_report_file_path=
                self.prediction_config.prediction_report_file_path
        )

        return prediction_result
class CustomerData:

    def __init__(
        self,
        Age,
        Income,
        Days_as_Customer,
        Recency,
        Wines,
        Fruits,
        Meat,
        Fish,
        Sweets,
        Gold,
        Web,
        Catalog,
        Store,
        Discount_Purchases,
        NumWebVisitsMonth
    ):

        self.Age = Age
        self.Income = Income
        self.Days_as_Customer = Days_as_Customer
        self.Recency = Recency
        self.Wines = Wines
        self.Fruits = Fruits
        self.Meat = Meat
        self.Fish = Fish
        self.Sweets = Sweets
        self.Gold = Gold
        self.Web = Web
        self.Catalog = Catalog
        self.Store = Store
        self.Discount_Purchases = Discount_Purchases
        self.NumWebVisitsMonth = NumWebVisitsMonth
        
    def get_dataframe(self):

        data = {

            "Age":[self.Age],

            "Income":[self.Income],

            "Days_as_Customer":[self.Days_as_Customer],

            "Recency":[self.Recency],

            "Wines":[self.Wines],

            "Fruits":[self.Fruits],

            "Meat":[self.Meat],

            "Fish":[self.Fish],

            "Sweets":[self.Sweets],

            "Gold":[self.Gold],

            "Web":[self.Web],

            "Catalog":[self.Catalog],

            "Store":[self.Store],

            "Discount Purchases":[self.Discount_Purchases],

            "NumWebVisitsMonth":[self.NumWebVisitsMonth]

        }

        return pd.DataFrame(data)
