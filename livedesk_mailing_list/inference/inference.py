import mlflow
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from mlflow.exceptions import MlflowException

def get_experiment_and_run_id(model_name: str, model_version: int) -> str:
    """
    Retrieve the run ID associated with a specific model name and version from MLflow.

    Parameters:
    model_name (str): The name of the model registered in MLflow.
    model_version (int): The version of the model.

    Returns:
    str: The run ID associated with the given model version.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version(name=model_name, version=model_version)
        return model_version_details.run_id

    except MlflowException as mlflow_error:
        raise RuntimeError(f"MLflow error: Failed to retrieve run ID for model '{model_name}' version '{model_version}': {mlflow_error}")
    except KeyError as key_error:
        raise RuntimeError(f"KeyError: Missing run ID in model version details: {key_error}")

def load_model_pipeline(model_name: str, model_version: int) -> Pipeline:
    """
    Load a trained model pipeline from MLflow.

    Parameters:
    model_name (str): The name of the model.
    model_version (int): The version of the model.

    Returns:
    Pipeline: The loaded model pipeline.
    """
    try:
        return mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    except MlflowException as mlflow_error:
        raise RuntimeError(f"Failed to load model '{model_name}' version '{model_version}': {mlflow_error}")

def get_metrics_from_mlflow(run_id: str) -> Dict[str, Any]:
    """
    Retrieve metrics from an MLflow run.

    Parameters:
    run_id (str): The run ID from which to retrieve metrics.

    Returns:
    Dict[str, Any]: A dictionary containing the metrics logged in the specified MLflow run.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics
    except MlflowException as mlflow_error:
        raise RuntimeError(f"Failed to retrieve metrics from MLflow run '{run_id}': {mlflow_error}")

def analyze_article_impact(metrics: Dict[str, Any], prediction: float, z_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Analyze the impact of an article based on prediction scores and historical metrics.

    Parameters:
    metrics (Dict[str, Any]): Historical metrics for comparison.
    prediction (float): The prediction score for the article.
    z_threshold (float): Z-score threshold for impact recommendations.

    Returns:
    Dict[str, Any]: A dictionary containing the recommendation, z-score, and article impact.
    """
    mean_impact = metrics.get('mean_impact', 0.5)
    std_impact = metrics.get('std_impact', 0.1)

    if std_impact == 0:
        raise ValueError("Standard deviation cannot be zero for z-score calculation.")

    z_score = (prediction - mean_impact) / std_impact

    if z_score > z_threshold:
        recommendation = "High Impact"
    elif z_score < -z_threshold:
        recommendation = "Low Impact"
    else:
        recommendation = "Neutral Impact"

    return {
        'recommendation': recommendation,
        'z_score': z_score,
        'article_impact': prediction
    }

def perform_inference(pipeline: Pipeline, X_infer: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    """
    Perform inference and calculate user counts based on different thresholds.

    Parameters:
    pipeline (Pipeline): The trained model pipeline.
    X_infer (pd.DataFrame): The input features for inference.
    thresholds (List[float]): List of thresholds to evaluate.

    Returns:
    pd.DataFrame: DataFrame containing the threshold results.
    """
    predictions = pipeline.predict_proba(X_infer)[:, 1]
    results = [{'threshold': threshold, 'user_count': np.sum(predictions >= threshold)} for threshold in thresholds]
    return pd.DataFrame(results)

def lightweight_impact_analysis(metrics: Dict[str, Any], predictions: np.ndarray, z_threshold: float) -> pd.DataFrame:
    """
    Perform lightweight impact analysis on predictions.

    Parameters:
    metrics (Dict[str, Any]): Historical metrics for comparison.
    predictions (np.ndarray): Predictions from the model.
    z_threshold (float): Z-score threshold for impact recommendations.

    Returns:
    pd.DataFrame: DataFrame containing the impact analysis results.
    """
    analysis_results = [analyze_article_impact(metrics, prediction, z_threshold) for prediction in predictions]
    return pd.DataFrame(analysis_results)

def run_inference(model_name: str, model_version: int, X_infer: pd.DataFrame, thresholds: List[float], z_threshold: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Run the full inference pipeline, including loading the model, performing inference, 
    analyzing impact, and generating explanations.

    Parameters:
    model_name (str): The name of the model.
    model_version (int): The version of the model.
    X_infer (pd.DataFrame): The input features for inference.
    thresholds (List[float]): List of thresholds to evaluate.
    z_threshold (float): Z-score threshold for impact recommendations.

    Returns:
    Dict[str, pd.DataFrame]: Results of the threshold evaluation and impact analysis.
    """
    run_id = get_experiment_and_run_id(model_name, model_version)
    pipeline = load_model_pipeline(model_name, model_version)
    metrics = get_metrics_from_mlflow(run_id)
    
    threshold_results = perform_inference(pipeline, X_infer, thresholds)
    impact_analysis_results = lightweight_impact_analysis(metrics, threshold_results['user_count'], z_threshold)

    return {
        'threshold_results': threshold_results,
        'impact_analysis_results': impact_analysis_results
    }
