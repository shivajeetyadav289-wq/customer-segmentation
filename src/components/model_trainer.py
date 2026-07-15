import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.entity.config_entity import ModelTrainerConfig

from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    DataIngestionArtifact
    
)

class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
    def initiate_model_trainer(self):

        train_arr = np.load(
            self.data_transformation_artifact.transformed_train_file_path
        )

        print(train_arr.shape)
        
        # find the best number of clusters
        best_score = -1
        best_k = 2

        for k in range(2,11):

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(train_arr)

            score = silhouette_score(
                train_arr,
                labels
            )

            print(f"K={k}  Score={score:.4f}")

            if score > best_score:

                best_score = score

                best_k = k
                
        #train final model
        final_model = KMeans(

            n_clusters=best_k,

            random_state=42,

            n_init=10
        )

        final_model.fit(train_arr)
        
        # save the final model
        
        os.makedirs(
            self.model_trainer_config.model_trainer_dir,
            exist_ok=True
        )

        joblib.dump(

            final_model,

            self.model_trainer_config.trained_model_file_path
        )

        print("Model Saved Successfully")
        
       
        # original_train = pd.read_csv(
        #     self.data_ingestion_artifact.train_file_path
        # )
        engineered_train = pd.read_csv(
            self.data_transformation_artifact.engineered_train_file_path
        )

        # clusters = final_model.fit_predict(train_arr)
        # original_train["Cluster"] = clusters
        clusters = final_model.labels_
        engineered_train["Cluster"] = clusters

        # print(original_train.head()) 
    # save clustered data
        
        os.makedirs(
            self.model_trainer_config.model_trainer_dir,
            exist_ok=True
        )

        # original_train.to_csv(
        #     self.model_trainer_config.clustered_data_file_path,
        #     index=False
        # )
        engineered_train.to_csv(
            self.model_trainer_config.clustered_data_file_path,
            index=False
        )

        print("enginered Clustered Dataset Saved")
        
        # save model 
        
        joblib.dump(
            final_model,
            self.model_trainer_config.trained_model_file_path
        )

        print("KMeans Model Saved")
        
        return ModelTrainerArtifact(

            trained_model_file_path=
                self.model_trainer_config.trained_model_file_path,
            clustered_data_file_path=
                self.model_trainer_config.clustered_data_file_path,

            best_k=best_k,

            silhouette_score=best_score
        )