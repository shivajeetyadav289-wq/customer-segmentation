from dataclasses import dataclass
import os

# Data_ingestion
@dataclass
class DataIngestionConfig:

    artifact_dir = "artifacts"

    data_ingestion_dir = os.path.join(
        artifact_dir,
        "data_ingestion"
    )

    raw_file_path = os.path.join(
        data_ingestion_dir,
        "raw.csv"
    )

    train_file_path = os.path.join(
        data_ingestion_dir,
        "train.csv"
    )

    test_file_path = os.path.join(
        data_ingestion_dir,
        "test.csv"
    )
# data validation
@dataclass
class DataValidationConfig:

    artifact_dir: str = "artifacts"

    data_validation_dir: str = os.path.join(
        artifact_dir,
        "data_validation"
    )

    report_file_path: str = os.path.join(
        data_validation_dir,
        "report.yaml"
    )
    
#data Transformation
@dataclass
class DataTransformationConfig:

    artifact_dir: str = "artifacts"

    data_transformation_dir: str = os.path.join(
        artifact_dir,
        "data_transformation"
    )

    preprocessor_object_file_path: str = os.path.join(
        data_transformation_dir,
        "preprocessor.pkl"
    )

    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        "train.npy"
    )

    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        "test.npy"
    )
    engineered_train_file_path = os.path.join(
            data_transformation_dir,
            "engineered_train.csv"
        )

    engineered_test_file_path = os.path.join(
            data_transformation_dir,
            "engineered_test.csv"
        )
    
# Model training

@dataclass
class ModelTrainerConfig:

    artifact_dir: str = "artifacts"

    model_trainer_dir: str = os.path.join(
        artifact_dir,
        "model_trainer"
    )

    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        "kmeans_model.pkl"
    )
    
    clustered_data_file_path: str = os.path.join(
        model_trainer_dir,
        "clustered_data.csv"
    )
# Model Evaluation

@dataclass
class ModelEvaluationConfig:

    artifact_dir: str = "artifacts"

    model_evaluation_dir: str = os.path.join(
        artifact_dir,
        "model_evaluation"
    )

    report_file_path: str = os.path.join(
        model_evaluation_dir,
        "evaluation_report.txt"
    )
    
# Cluster analysis
@dataclass
class ClusterAnalysisConfig:

    artifact_dir: str = "artifacts"

    cluster_analysis_dir: str = os.path.join(
        artifact_dir,
        "cluster_analysis"
    )

    summary_file_path: str = os.path.join(
        cluster_analysis_dir,
        "cluster_summary.csv"
    )

    plots_dir: str = os.path.join(
        cluster_analysis_dir,
        "plots"
    )
    
# Project report
@dataclass
class PredictionConfig:

    artifact_dir: str = "artifacts"

    prediction_dir: str = os.path.join(
        artifact_dir,
        "prediction"
    )

    prediction_report_file_path: str = os.path.join(
        prediction_dir,
        "prediction_report.txt"
    )
