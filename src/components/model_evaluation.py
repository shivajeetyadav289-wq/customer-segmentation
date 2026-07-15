import numpy as np
import os
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import silhouette_score

from src.entity.config_entity import ModelEvaluationConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact
)

class ModelEvaluation:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact : ModelTrainerArtifact
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_evaluation_config = (
            model_evaluation_config
        )
        
        self.model_trainer_artifact = (
            model_trainer_artifact
        )
    # create evaluate model
    def evaluate_models(self):

        train_arr = np.load(
        self.data_transformation_artifact.transformed_train_file_path
        )

        best_k = self.model_trainer_artifact.best_k

        print(f"Best K from Model Trainer: {best_k}")
    
    # Evaluate k-means
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        )

        kmeans_labels = kmeans.fit_predict(train_arr)

        kmeans_score = silhouette_score(
            train_arr,
            kmeans_labels
        )
        
        # Evaluate Agglomeritive algo
        agg = AgglomerativeClustering(
            n_clusters=2
        )

        agg_labels = agg.fit_predict(train_arr)

        agg_score = silhouette_score(
            train_arr,
            agg_labels
        )
    # Evaluate DBSCAN
        
        dbscan = DBSCAN(
            eps=1.5,
            min_samples=5
        )

        db_labels = dbscan.fit_predict(train_arr)

        if len(set(db_labels)) > 1:

            db_score = silhouette_score(
                train_arr,
                db_labels
            )

        else:

            db_score = -1
    
    # compare algorithum
    
        scores = {

        "KMeans": kmeans_score,

        "Agglomerative": agg_score,

        "DBSCAN": db_score
    }
        best_algorithm = max(
            scores,
            key=scores.get
        )

        best_score = scores[best_algorithm]
        
    #print the comparision
        print("\nAlgorithm Comparison")
        print("------------------------------")

        for algorithm, score in scores.items():

            print(f"{algorithm:<20} {score:.4f}")

        print("------------------------------")

        print("Best Algorithm :", best_algorithm)

        print("Best Score :", best_score)
        
    # save the report 
        os.makedirs(
            self.model_evaluation_config.model_evaluation_dir,
            exist_ok=True
        )
    # write comparision in a file 
        with open(
            self.model_evaluation_config.report_file_path,
            "w"
        ) as file:

            file.write("Algorithm Comparison\n")
            file.write("--------------------------\n")

            for algorithm, score in scores.items():

                file.write(
                    f"{algorithm}: {score:.4f}\n"
                )

            file.write("\n")

            file.write(
                f"Best Algorithm: {best_algorithm}\n"
            )

            file.write(
                f"Best Score: {best_score:.4f}\n"
            )
            
        return ModelEvaluationArtifact(
            best_algorithm=best_algorithm,
            best_score=best_score,
            best_k=best_k,
            report_file_path=self.model_evaluation_config.report_file_path
        )