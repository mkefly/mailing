import pytest
from hyperopt import hp
from livedesk_mailing_list.models.model_factory import create_pipeline
from livedesk_mailing_list.training.hyperparameter_optimization import parse_hyperopt_space, optimize_hyperparameters
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock

@pytest.fixture
def hyperopt_space():
    return {
        'model_type': {'type': 'choice', 'options': ['random_forest', 'xgboost']},
        'lgbm_params': {
            'n_estimators': {'type': 'loguniform', 'low': 50, 'high': 500},
            'learning_rate': {'type': 'uniform', 'low': 0.01, 'high': 0.1}
        }
    }

@pytest.fixture
def mock_config(hyperopt_space):
    return {
        'hyperopt': {
            'space': hyperopt_space
        },
        'cv_folds': 3,
        'evaluation_metric': 'f1',
        'pipeline': {
            'features': {
                'use_content': True
            }
        }
    }

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return pd.DataFrame(X), y

def test_parse_hyperopt_space_choice(hyperopt_space):
    parsed_space = parse_hyperopt_space(hyperopt_space)
    assert isinstance(parsed_space['model_type'], hp.pyll.base.Apply)
    assert isinstance(parsed_space['lgbm_params']['n_estimators'], hp.pyll.base.Apply)
    assert isinstance(parsed_space['lgbm_params']['learning_rate'], hp.pyll.base.Apply)

def test_parse_hyperopt_space_missing_type():
    space = {
        'n_estimators': {'low': 50, 'high': 500}
    }
    with pytest.raises(KeyError, match="Missing 'type' key in the hyperopt space for parameter: n_estimators"):
        parse_hyperopt_space(space)

def test_parse_hyperopt_space_nested(hyperopt_space):
    parsed_space = parse_hyperopt_space(hyperopt_space)
    assert 'model_type' in parsed_space
    assert 'lgbm_params.n_estimators' not in parsed_space  # Nested structure should be flattened

def test_optimize_hyperparameters_no_space(sample_data, mock_config):
    mock_config['hyperopt'] = {}
    with pytest.raises(ValueError, match="Hyperopt space is not defined in the configuration."):
        optimize_hyperparameters(*sample_data, mock_config)

@patch('livedesk_mailing_list.models.model_factory.create_pipeline')
@patch('sklearn.model_selection.cross_val_score')
def test_optimize_hyperparameters(mock_cross_val_score, mock_create_pipeline, sample_data, mock_config):
    mock_cross_val_score.return_value = np.array([0.8, 0.82, 0.79])
    mock_pipeline = MagicMock()
    mock_create_pipeline.return_value = mock_pipeline

    best_params = optimize_hyperparameters(*sample_data, mock_config)
    
    # Ensure that fmin was called and that best_params contains the correct values
    assert 'model_type' in best_params
    assert 'n_estimators' in best_params
    assert 'learning_rate' in best_params

    # Ensure that create_pipeline was called with the correct arguments
    mock_create_pipeline.assert_called()

    # Ensure that cross_val_score was called with the correct arguments
    mock_cross_val_score.assert_called_with(mock_pipeline, sample_data[0], sample_data[1], cv=3, scoring='f1')

if __name__ == '__main__':
    pytest.main()
