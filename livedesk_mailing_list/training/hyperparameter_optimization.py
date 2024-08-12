from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
from livedesk_mailing_list.models.model_factory import create_pipeline
import pandas as pd
import numpy as np

def parse_hyperopt_space(space: Dict[str, Any]) -> Dict[str, Any]:
    from hyperopt import hp
    
    type_map = {
        'choice': lambda key, value: hp.choice(key, value['options']),
        'loguniform': lambda key, value: hp.loguniform(key, value['low'], value['high']),
        'uniform': lambda key, value: hp.uniform(key, value['low'], value['high'])
    }

    parsed_space = {}
    for key, value in space.items():
        if isinstance(value, dict) and 'type' in value:
            parsed_space[key] = type_map[value['type']](key, value)
        elif isinstance(value, dict):  # Handle nested dictionaries (like lgbm_params)
            nested_params = parse_hyperopt_space(value)
            parsed_space.update(nested_params)
        else:
            raise KeyError(f"Missing 'type' key in the hyperopt space for parameter: {key}")
    return parsed_space


def optimize_hyperparameters(X_train: pd.DataFrame, y_train: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    space = config.get('hyperopt', {}).get('space', None)
    if not space:
        raise ValueError("Hyperopt space is not defined in the configuration.")

    parsed_space = parse_hyperopt_space(space)
    cv_folds = config.get('cv_folds', 3)
    evaluation_metric = config.get('evaluation_metric', 'f1')

    def objective(params):
        model_type = params.pop('model_type')
        pipeline = create_pipeline(config, model_type=model_type, model_params=params)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=evaluation_metric).mean()
        return {'loss': -cv_score, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(objective, parsed_space, algo=tpe.suggest, max_evals=10, trials=trials)
    return best_params
