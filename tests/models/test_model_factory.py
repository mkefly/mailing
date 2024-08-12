import pytest
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from livedesk_mailing_list.models.transformers import BERTTransformer, Word2VecTransformer, TFIDFTransformer
from livedesk_mailing_list.models.model_factory import create_model, create_embedding_transformer, create_features_pipeline, create_pipeline
from loguru import logger


@pytest.fixture
def config():
    return {
        'pipeline': {
            'embedding_type': 'bert',
            'features': {
                'use_content': True,
                'use_title': False,
                'use_tags': False
            }
        },
        'bert_params': {
            'batch_size': 16,
            'max_length': 64
        },
        'word2vec_params': {
            'vector_size': 50,
            'window': 3
        },
        'tfidf_params': {
            'max_features': 500,
            'ngram_range': (1, 1)
        }
    }

@pytest.fixture
def model_params():
    return {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'invalid_param': 'should be filtered'
    }

def test_create_model_valid():
    model = create_model('lgbm', {'n_estimators': 100, 'learning_rate': 0.1})
    assert isinstance(model, lgb.LGBMClassifier)
    assert model.get_params()['n_estimators'] == 100
    assert model.get_params()['learning_rate'] == 0.1

def test_create_model_invalid():
    with pytest.raises(ValueError):
        create_model('unsupported_model', {})

def test_create_model_invalid_params(caplog):
    caplog.clear()
    model = create_model('xgboost', {'n_estimators': 100, 'learning_rate': 0.1, 'invalid_param': 'filtered'})
    assert isinstance(model, xgb.XGBClassifier)
    assert 'Filtered out invalid parameters' in caplog.text

def test_create_embedding_transformer_bert(config):
    transformer = create_embedding_transformer('content', config)
    assert isinstance(transformer, BERTTransformer)
    assert transformer.batch_size == 16
    assert transformer.max_length == 64

def test_create_embedding_transformer_word2vec(config):
    config['pipeline']['embedding_type'] = 'word2vec'
    transformer = create_embedding_transformer('content', config)
    assert isinstance(transformer, Word2VecTransformer)
    assert transformer.vector_size == 50
    assert transformer.window == 3

def test_create_embedding_transformer_tfidf(config):
    config['pipeline']['embedding_type'] = 'tfidf'
    transformer = create_embedding_transformer('content', config)
    assert isinstance(transformer, TFIDFTransformer)
    assert transformer.max_features == 500
    assert transformer.ngram_range == (1, 1)

def test_create_embedding_transformer_invalid(config):
    config['pipeline']['embedding_type'] = 'unknown'
    with pytest.raises(ValueError):
        create_embedding_transformer('content', config)

def test_create_features_pipeline(config):
    pipeline = create_features_pipeline(config['pipeline']['features'], config)
    assert isinstance(pipeline, FeatureUnion)
    assert len(pipeline.transformer_list) == 1  # Only content embedding should be included

def test_create_pipeline(config, model_params):
    pipeline = create_pipeline(config, 'random_forest', model_params)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.named_steps['preprocessor'], FeatureUnion)
    assert isinstance(pipeline.named_steps['model'], RandomForestClassifier)

def test_create_pipeline_with_xgboost(config, model_params):
    pipeline = create_pipeline(config, 'xgboost', model_params)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps['model'], xgb.XGBClassifier)
