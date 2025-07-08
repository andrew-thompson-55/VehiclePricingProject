# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import logging
import sys
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--train_data", type=str, help="Path to train dataset"
    )  # Specify the type for train_data
    parser.add_argument(
        "--test_data", type=str, help="Path to test dataset"
    )  # Specify the type for test_data
    parser.add_argument(
        "--model_output", type=str, help="Path of output model"
    )  # Specify the type for model_output
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="The number of trees in the forest",
    )  # Specify the type and default value for n_estimators
    parser.add_argument(
        "--max_depth", type=int, default=None, help="The maximum depth of the tree"
    )  # Specify the type and default value for max_depth

    args = parser.parse_args()

    return args


def validate_data(X_train, y_train, X_test, y_test):
    """Validate data quality before training"""
    logger.info("Validating training and test data")
    
    # Check for empty data
    if X_train.empty or X_test.empty:
        raise ValueError("Training or test data is empty")
    
    # Check for null values in target
    if y_train.isnull().any() or y_test.isnull().any():
        raise ValueError("Target variable contains null values")
    
    # Check for null values in features
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        logger.warning("Input features contain null values")
    
    # Check for infinite values
    if np.isinf(X_train.values).any() or np.isinf(X_test.values).any():
        raise ValueError("Input features contain infinite values")
    
    # Check for consistent feature count
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature count mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}")
    
    logger.info(f"Data validation passed - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return True


def main(args):
    """Read train and test datasets, train model, evaluate model, save trained model"""

    try:
        logger.info("Starting model training process")
        
        # Read train and test data from csv
        logger.info(f"Reading training data from: {args.train_data}")
        train_df = pd.read_csv(Path(args.train_data) / "train.csv")
        logger.info(f"Training data loaded with shape: {train_df.shape}")
        
        logger.info(f"Reading test data from: {args.test_data}")
        test_df = pd.read_csv(Path(args.test_data) / "test.csv")
        logger.info(f"Test data loaded with shape: {test_df.shape}")

        # Split the data into input(X) and output(y)
        logger.info("Preparing features and target variables")
        y_train = train_df["price"]  # Specify the target column
        X_train = train_df.drop(columns=["price"])
        y_test = test_df["price"]
        X_test = test_df.drop(columns=["price"])
        
        logger.info(f"Feature columns: {list(X_train.columns)}")

        # Validate data
        validate_data(X_train, y_train, X_test, y_test)

        # Initialize and train a RandomForest Regressor
        logger.info(f"Training RandomForest with n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )  # Provide the arguments for RandomForestRegressor
        model.fit(X_train, y_train)  # Train the model
        logger.info("Model training completed successfully")

        # Log model hyperparameters
        mlflow.log_param("model", "RandomForestRegressor")  # Provide the model name
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])

        # Predict using the RandomForest Regressor on test data
        logger.info("Making predictions on test data")
        yhat_test = model.predict(X_test)  # Predict the test data

        # Compute and log mean squared error for test data
        mse = mean_squared_error(y_test, yhat_test)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - yhat_test))
        
        logger.info(
            "Model Performance Metrics:"
        )
        logger.info(f"  Mean Squared Error: {mse:.2f}")
        logger.info(f"  Root Mean Squared Error: {rmse:.2f}")
        logger.info(f"  Mean Absolute Error: {mae:.2f}")
        
        mlflow.log_metric("MSE", float(mse))  # Log the MSE
        mlflow.log_metric("RMSE", float(rmse))
        mlflow.log_metric("MAE", float(mae))

        # Save the model
        logger.info(f"Saving model to: {args.model_output}")
        mlflow.sklearn.save_model(sk_model=model, path=args.model_output)  # Save the model
        logger.info("Model saved successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        mlflow.log_metric("error", 1)
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        mlflow.log_metric("error", 1)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        mlflow.log_metric("error", 1)
        raise


if __name__ == "__main__":
    mlflow.start_run()

    try:
        # Parse Arguments
        args = parse_args()

        lines = [
            f"Train dataset input path: {args.train_data}",
            f"Test dataset input path: {args.test_data}",
            f"Model output path: {args.model_output}",
            f"Number of Estimators: {args.n_estimators}",
            f"Max Depth: {args.max_depth}",
        ]

        for line in lines:
            logger.info(line)

        main(args)
        logger.info("Model training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        mlflow.log_metric("error", 1)
        sys.exit(1)
    finally:
        mlflow.end_run()
