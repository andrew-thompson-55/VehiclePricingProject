# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Name under which model will be registered"
    )  # Hint: Specify the type for model_name (str)
    parser.add_argument(
        "--model_path", type=str, help="Model directory"
    )  # Hint: Specify the type for model_path (str)
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")

    return args


def validate_model_path(model_path):
    """Validate that the model path exists and contains required files"""
    logger.info(f"Validating model path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Check for required MLflow model files
    required_files = ['MLmodel', 'conda.yaml', 'python_env.yaml']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing MLflow model files: {missing_files}")
    
    logger.info("Model path validation completed")
    return True


def main(args):
    """Loads the best-trained model from the sweep job and registers it"""

    try:
        logger.info(f"Starting model registration for: {args.model_name}")

        # Validate model path
        validate_model_path(args.model_path)

        # Load model
        logger.info(f"Loading model from: {args.model_path}")
        model = mlflow.sklearn.load_model(args.model_path)  # Load the model from model_path
        logger.info("Model loaded successfully")

        # Log model using mlflow
        logger.info("Logging model with MLflow")
        mlflow.sklearn.log_model(
            model, args.model_name
        )  # Log the model using with model_name
        logger.info("Model logged successfully")

        # Register logged model using mlflow
        logger.info("Registering model in MLflow registry")
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{args.model_name}"
        mlflow_model = mlflow.register_model(
            model_uri, args.model_name
        )  # register the model with model_uri and model_name
        model_version = mlflow_model.version  # Get the version of the registered model
        logger.info(f"Model registered successfully with version: {model_version}")

        # Write model info
        logger.info("Writing model info to JSON file")
        model_info = {"id": f"{args.model_name}:{model_version}"}
        output_path = os.path.join(
            args.model_info_output_path, "model_info.json"
        )  # Specify the name of the JSON file (model_info.json)
        
        # Ensure output directory exists
        os.makedirs(args.model_info_output_path, exist_ok=True)
        
        with open(output_path, "w") as of:
            json.dump(model_info, of)  # write model_info to the output file
        
        logger.info(f"Model info written to: {output_path}")
        
        # Log additional metrics
        mlflow.log_metric("model_version", model_version)
        mlflow.log_metric("registration_success", 1)
        
        logger.info("Model registration completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        mlflow.log_metric("error", 1)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}")
        mlflow.log_metric("error", 1)
        raise


if __name__ == "__main__":
    mlflow.start_run()

    try:
        # Parse Arguments
        args = parse_args()

        lines = [
            f"Model name: {args.model_name}",
            f"Model path: {args.model_path}",
            f"Model info output path: {args.model_info_output_path}",
        ]

        for line in lines:
            logger.info(line)

        main(args)
        logger.info("Model registration pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Registration pipeline failed: {e}")
        mlflow.log_metric("error", 1)
        sys.exit(1)
    finally:
        mlflow.end_run()
