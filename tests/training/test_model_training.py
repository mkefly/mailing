import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from livedesk_mailing_list.training.model_training import train_model

@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 20))
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def config():
    return {
        'fixed_params': {
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': 5
        },
        'cv_folds': 3,
        'evaluation_metric': 'f1',
        'hyperopt': {
            'space': {
                'n_estimators': {'type': 'uniform', 'low': 50, 'high': 150},
                'max_depth': {'type': 'choice', 'options': [3, 5, 7]}
            }
        }
    }

@patch('livedesk_mailing_list.training.hyperparameter_optimization.optimize_hyperparameters')
@patch('livedesk_mailing_list.models.model_factory.create_pipeline')
@patch('sklearn.model_selection.cross_val_score')
def test_train_model_with_optimization(mock_cross_val_score, mock_create_pipeline, mock_optimize_hyperparameters, sample_data, config):
    # Mock the optimize_hyperparameters to return some dummy best_params
    mock_optimize_hyperparameters.return_value = {'n_estimators': 120, 'max_depth': 7}
    
    # Mock create_pipeline to return a MagicMock pipeline
    mock_pipeline = MagicMock()
    mock_create_pipeline.return_value = mock_pipeline
    
    # Mock cross_val_score to return a dummy array of scores
    mock_cross_val_score.return_value = np.array([0.8, 0.82, 0.79])

    pipeline, cv_results = train_model(*sample_data, config, optimize=True)
    
    # Assertions to ensure correct calls and returns
    mock_optimize_hyperparameters.assert_called_once_with(sample_data[0], sample_data[1], config)
    mock_create_pipeline.assert_called_once_with(config, model_type='random_forest', model_params={'n_estimators': 120, 'max_depth': 7})
    mock_pipeline.fit.assert_called_once_with(sample_data[0], sample_data[1])
    mock_cross_val_score.assert_called_once_with(mock_pipeline, sample_data[0], sample_data[1], cv=3, scoring='f1')

    assert pipeline == mock_pipeline
    assert cv_results['cv_mean'] == np.mean([0.8, 0.82, 0.79])
    assert cv_results['cv_std'] == np.std([0.8, 0.82, 0.79])
    assert cv_results['cv_scores'].tolist() == [0.8, 0.82, 0.79]

@patch('livedesk_mailing_list.models.model_factory.create_pipeline')
@patch('sklearn.model_selection.cross_val_score')
def test_train_model_without_optimization(mock_cross_val_score, mock_create_pipeline, sample_data, config):
    # Mock create_pipeline to return a MagicMock pipeline
    mock_pipeline = MagicMock()
    mock_create_pipeline.return_value = mock_pipeline
    
    # Mock cross_val_score to return a dummy array of scores
    mock_cross_val_score.return_value = np.array([0.75, 0.78, 0.76])

    pipeline, cv_results = train_model(*sample_data, config, optimize=False)
    
    # Assertions to ensure correct calls and returns
    mock_create_pipeline.assert_called_once_with(config, model_type='random_forest', model_params=config['fixed_params'])
    mock_pipeline.fit.assert_called_once_with(sample_data[0], sample_data[1])
    mock_cross_val_score.assert_called_once_with(mock_pipeline, sample_data[0], sample_data[1], cv=3, scoring='f1')

    assert pipeline == mock_pipeline
    assert cv_results['cv_mean'] == np.mean([0.75, 0.78, 0.76])
    assert cv_results['cv_std'] == np.std([0.75, 0.78, 0.76])
    assert cv_results['cv_scores'].tolist() == [0.75, 0.78, 0.76]
