import os
import warnings
import sys
import logging
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Function to evaluate model performance
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Download dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Error downloading dataset: %s", e)
        sys.exit(1)

    # Split dataset into training and test sets
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    # Features and target
    train_x = train.drop(columns=["quality"])
    test_x = test.drop(columns=["quality"])
    train_y = train["quality"]
    test_y = test["quality"]

    # Read hyperparameters from command line arguments (default: alpha=0.5, l1_ratio=0.5)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Set MLflow tracking server URI BEFORE starting the run
    remote_server_uri = "http://ec2-52-91-189-219.compute-1.amazonaws.com:5000/"
    mlflow.set_tracking_uri(remote_server_uri)

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id  # Get the run ID
        print(f"MLflow Run ID: {run_id}")

        # Train ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Make predictions
        predicted_qualities = lr.predict(test_x)

        # Evaluate model
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Print metrics
        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Check if the run ID exists before logging the model
        try:
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                # Register the model if tracking with a server
                mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
            else:
                mlflow.sklearn.log_model(lr, "model")

            print("Model logged successfully!")

        except Exception as e:
            logger.exception("Error logging model: %s", e)
