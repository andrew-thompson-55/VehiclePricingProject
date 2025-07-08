# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument(
        "--raw_data", type=str, help="Path to raw data"
    )  # Specify the type for raw data (str)
    parser.add_argument(
        "--train_data", type=str, help="Path to train dataset"
    )  # Specify the type for train data (str)
    parser.add_argument(
        "--test_data", type=str, help="Path to test dataset"
    )  # Specify the type for test data (str)
    parser.add_argument(
        "--test_train_ratio", type=float, default=0.2, help="Test-train ratio"
    )  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args


def validate_data(df):
    """Validate the input data for quality issues"""
    logger.info(f"Validating data with shape: {df.shape}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Input data is empty")
    
    # Check for required columns
    required_columns = ['Segment', 'price']  # Add other required columns as needed
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Data validation completed successfully")
    return True


def main(args):  # Write the function name for the main data preparation logic
    """Read, preprocess, split, and save datasets"""

    try:
        logger.info("Starting data preparation process")
        
        # Reading Data
        logger.info(f"Reading data from: {args.raw_data}")
        df = pd.read_csv(args.raw_data)
        logger.info(f"Successfully read data with shape: {df.shape}")
        
        # Validate data
        validate_data(df)
        
        # Encode categorical feature
        logger.info("Encoding categorical feature 'Segment'")
        le = LabelEncoder()
        df["Segment"] = le.fit_transform(
            df["Segment"]
        )  # Write code to encode the categorical feature
        logger.info("Successfully encoded categorical feature")

        # Split Data into train and test datasets
        logger.info(f"Splitting data with test ratio: {args.test_train_ratio}")
        train_df, test_df = train_test_split(
            df, test_size=args.test_train_ratio, random_state=42
        )  #  Write code to split the data into train and test datasets
        logger.info(f"Split completed - Train: {train_df.shape}, Test: {test_df.shape}")

        # Save the train and test data
        logger.info("Creating output directories")
        os.makedirs(
            args.train_data, exist_ok=True
        )  # Create directories for train_data and test_data
        os.makedirs(
            args.test_data, exist_ok=True
        )  # Create directories for train_data and test_data
        
        train_output_path = os.path.join(args.train_data, "train.csv")
        test_output_path = os.path.join(args.test_data, "test.csv")
        
        logger.info(f"Saving train data to: {train_output_path}")
        train_df.to_csv(
            train_output_path, index=False
        )  # Specify the name of the train data file
        
        logger.info(f"Saving test data to: {test_output_path}")
        test_df.to_csv(
            test_output_path, index=False
        )  # Specify the name of the test data file

        # log the metrics
        mlflow.log_metric("train size", train_df.shape[0])  # Log the train dataset size
        mlflow.log_metric("test size", test_df.shape[0])  # Log the test dataset size
        mlflow.log_metric("total features", len(df.columns))
        mlflow.log_metric("categorical_features_encoded", 1)
        
        logger.info("Data preparation completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        mlflow.log_metric("error", 1)
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        mlflow.log_metric("error", 1)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data preparation: {e}")
        mlflow.log_metric("error", 1)
        raise


if __name__ == "__main__":
    mlflow.start_run()

    try:
        # Parse Arguments
        args = parse_args()  # Call the function to parse arguments

        lines = [
            f"Raw data path: {args.raw_data}",  # Print the raw_data path
            f"Train dataset output path: {args.train_data}",  # Print the train_data path
            f"Test dataset path: {args.test_data}",  # Print the test_data path
            f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
        ]

        for line in lines:
            logger.info(line)

        main(args)
        logger.info("Data preparation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        mlflow.log_metric("error", 1)
        sys.exit(1)
    finally:
        mlflow.end_run()
