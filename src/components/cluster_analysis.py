import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.entity.config_entity import ClusterAnalysisConfig
from src.entity.artifact_entity import (
    ClusterAnalysisArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact
)


class ClusterAnalysis:

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        cluster_analysis_config: ClusterAnalysisConfig
    ):

        self.model_trainer_artifact = model_trainer_artifact
        self.cluster_analysis_config = cluster_analysis_config
        self.data_transformation_artifact = data_transformation_artifact

    def initiate_cluster_analysis(self) -> ClusterAnalysisArtifact:

        print("\n------------------------")
        print("Cluster Analysis")
        print("------------------------")
        train_arr = np.load(
            self.data_transformation_artifact.transformed_train_file_path
        )

        # Read clustered dataset
        df = pd.read_csv(
            self.model_trainer_artifact.clustered_data_file_path
        )

        print(df.head())

        # Create directories
        os.makedirs(
            self.cluster_analysis_config.cluster_analysis_dir,
            exist_ok=True
        )

        os.makedirs(
            self.cluster_analysis_config.plots_dir,
            exist_ok=True
        )

        # -----------------------------
        # Cluster Size
        # -----------------------------
        print("\nCluster Counts")
        print(df["Cluster"].value_counts())

        # -----------------------------
        # Summary Statistics
        # -----------------------------
        summary = df.groupby("Cluster").mean(
            numeric_only=True
        )

        print("\nCluster Summary")
        print(summary)

        summary.to_csv(
            self.cluster_analysis_config.summary_file_path,
            index=True
        )

        print("Cluster summary saved successfully.")

        # -----------------------------
        # Cluster Size Plot
        # -----------------------------
        plt.figure(figsize=(6, 4))

        df["Cluster"].value_counts().sort_index().plot(
            kind="bar"
        )

        plt.title("Customers in Each Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Customers")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.cluster_analysis_config.plots_dir,
                "cluster_size.png"
            )
        )

        plt.close()

        print("Cluster size plot saved.")

        # -----------------------------
        # Income Boxplot
        # -----------------------------
        plt.figure(figsize=(6, 4))

        df.boxplot(
            column="Income",
            by="Cluster"
        )

        plt.title("Income Distribution by Cluster")
        plt.suptitle("")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.cluster_analysis_config.plots_dir,
                "income_boxplot.png"
            )
        )

        plt.close()

        print("Income boxplot saved.")
        # -----------------------------
        # Total Spending
        # -----------------------------
        
        df["Total_Spending"] = (
            df["Wines"] +
            df["Fruits"] +
            df["Meat"] +
            df["Fish"] +
            df["Sweets"] +
            df["Gold"]
        )
        plt.figure(figsize=(7,5))

        df.boxplot(
            column="Total_Spending",
            by="Cluster"
        )

        plt.title("Total Spending by Cluster")

        plt.suptitle("")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.cluster_analysis_config.plots_dir,
                "total_spending.png"
            )
        )

        plt.close()
        
        # -----------------------------
        # Corelation heatmap
        # -----------------------------

        corr = df.corr(numeric_only=True)

        plt.figure(figsize=(12,8))

        plt.imshow(corr)

        plt.colorbar()

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=90
        )

        plt.yticks(
            range(len(corr.columns)),
            corr.columns
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.cluster_analysis_config.plots_dir,
                "correlation_heatmap.png"
            )
        )

        plt.close()
        
        # ----------------------------
        # PCA Visualisation
        # ----------------------------
        
        pca = PCA(
            n_components=2,
            random_state=42
        )

        principal_components = pca.fit_transform(
            train_arr
        )
        # crate dataframe
        
        plot_df = pd.DataFrame()

        plot_df["PC1"] = principal_components[:,0]

        plot_df["PC2"] = principal_components[:,1]

        plot_df["Cluster"] = df["Cluster"]
        
        # Scattered plot
        plt.figure(figsize=(8,6))
        cluster_names = {
            0: "Budget / Price-Sensitive Customers",
            1: "High-Value Customers"
        }

        for cluster in sorted(plot_df["Cluster"].unique()):

            cluster_data = plot_df[
                plot_df["Cluster"] == cluster
            ]

            plt.scatter(
                cluster_data["PC1"],
                cluster_data["PC2"],
                label=f"Cluster {cluster}"
            )

        plt.legend()

        plt.title("Customer Segments (PCA)")

        plt.xlabel("Principal Component 1")

        plt.ylabel("Principal Component 2")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.cluster_analysis_config.plots_dir,
                "cluster_pca.png"
            )
        )

        plt.close()
        # -----------------------------
        # Return Artifact
        # -----------------------------
        cluster_analysis_artifact = ClusterAnalysisArtifact(
            summary_file_path=self.cluster_analysis_config.summary_file_path,
            plots_dir=self.cluster_analysis_config.plots_dir
        )

        print("Cluster Analysis Completed Successfully")

        return cluster_analysis_artifact
