from dataclasses import dataclass

# Data_ingestion
@dataclass
class DataIngestionArtifact:

    train_file_path: str
    test_file_path: str
    
# Data_Validation
@dataclass
class DataValidationArtifact:

    validation_status: bool

    report_file_path: str

# Data Transformation
@dataclass
class DataTransformationArtifact:

    transformed_train_file_path: str

    transformed_test_file_path: str
    
    engineered_train_file_path: str

    engineered_test_file_path: str

    preprocessor_object_file_path: str

#Model training
    
@dataclass
class ModelTrainerArtifact:

    trained_model_file_path: str

    clustered_data_file_path: str

    best_k: int

    silhouette_score: float
    
# Model Evaluation

@dataclass
class ModelEvaluationArtifact:

    best_algorithm: str

    best_score: float
    
    best_k: int

    report_file_path: str

# cluster analysis
@dataclass
class ClusterAnalysisArtifact:

    summary_file_path: str

    plots_dir: str

# Project report

@dataclass
class PredictionArtifact:

    prediction_report_file_path: str
