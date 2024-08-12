import mlflow
from livedesk_mailing_list.training.split import split_data
from livedesk_mailing_list.training.model_training import train_model
from livedesk_mailing_list.evaluation.evaluation_metrics import evaluate_model
from livedesk_mailing_list.utils.config_loader import load_yaml_config
from loguru import logger
import pandas as pd

def run_experiment(articles: pd.DataFrame, user_interactions: pd.DataFrame, config_path: str, optimize: bool = False):

    config = load_yaml_config(config_path)

    X_train, y_train, X_test, y_test = split_data(articles, user_interactions)

    pipeline, cv_results = train_model(X_train, y_train, config, optimize)

    mlflow.start_run()

    mlflow.log_params(config['fixed_params'])
    mlflow.log_metrics({
        'cv_mean_f1_score': cv_results['cv_mean'],
        'cv_std_f1_score': cv_results['cv_std']
    })

    for i, score in enumerate(cv_results['cv_scores'], 1):
        mlflow.log_metric(f'cv_score_fold_{i}', score)

    train_metrics = evaluate_model(pipeline, X_train, y_train)
    mlflow.log_metrics({f'train_{key}': value for key, value in train_metrics.items()})

    test_metrics = evaluate_model(pipeline, X_test, y_test, threshold=train_metrics['best_threshold'])
    mlflow.log_metrics({f'test_{key}': value for key, value in test_metrics.items()})

    mlflow.sklearn.log_model(pipeline, "model")

    logger.info("Experiment completed successfully. Metrics, model, and artifacts have been logged.")

    mlflow.end_run()
