import mlflow
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict, Any
from models.model_factory import create_pipeline
from training.hyperparameter_optimization import optimize_hyperparameters
from livedesk_mailing_list.evaluation.evaluation_metrics import evaluate_model
import pandas as pd
import numpy as np

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, config: Dict[str, Any], optimize: bool) -> Tuple[Any, Dict[str, Any]]:
    model_type = config['fixed_params']['model_type']
    
    if optimize:
        best_params = optimize_hyperparameters(X_train, y_train, config)
        pipeline = create_pipeline(config, model_type=model_type, model_params=best_params)
    else:
        pipeline = create_pipeline(config, model_type=model_type, model_params=config['fixed_params'])

    pipeline.fit(X_train, y_train)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config['cv_folds'], scoring=config.get('evaluation_metric', 'f1'))
    cv_results = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }

    return pipeline, cv_results
