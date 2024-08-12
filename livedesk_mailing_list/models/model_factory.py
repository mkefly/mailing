from sklearn.pipeline import Pipeline, FeatureUnion
from typing import Dict, Any
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from livedesk_mailing_list.models.transformers import BERTTransformer, Word2VecTransformer, TFIDFTransformer
from sklearn.base import BaseEstimator
from inspect import signature
from loguru import logger

def create_model(model_type: str, params: Dict[str, Any]) -> BaseEstimator:
    model_map = {
        'lgbm': lgb.LGBMClassifier,
        'xgboost': xgb.XGBClassifier,
        'random_forest': RandomForestClassifier,
        'sgd': SGDClassifier,
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_class = model_map[model_type]
    valid_params = model_class().get_params()
    filtered_params = {k: v for k, v in params.items() if k in valid_params}
    invalid_params = {k: v for k, v in params.items() if k not in valid_params}

    logger.info(f"Training {model_type} with parameters: {filtered_params}")
    if invalid_params:
        logger.warning(f"Filtered out invalid parameters for {model_type}: {invalid_params}")

    return model_class(**filtered_params)


def create_embedding_transformer(feature: str, config: Dict[str, Any]) -> BaseEstimator:
    transformer_map = {
        'bert': BERTTransformer,
        'word2vec': Word2VecTransformer,
        'tfidf': TFIDFTransformer
    }

    embedding_type = config['pipeline']['embedding_type']
    if embedding_type not in transformer_map:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    # Get the transformer's class
    transformer_class = transformer_map[embedding_type]

    # Get only the parameters relevant to this transformer
    transformer_params = config.get(f'{embedding_type}_params', {})
    valid_params = {k: v for k, v in transformer_params.items() if k in signature(transformer_class.__init__).parameters}

    return transformer_class(feature=feature, **valid_params)

def create_features_pipeline(feature_config: Dict[str, Any], config: Dict[str, Any]) -> FeatureUnion:
    transformers = {
        'content_embedding': lambda: create_embedding_transformer('content', config),
        'title_embedding': lambda: create_embedding_transformer('title', config),
        'tags_embedding': lambda: create_embedding_transformer('tags', config)
    }

    selected_transformers = [(name, transformer()) for name, transformer in transformers.items() if feature_config.get(f'use_{name.split("_")[0]}', False)]
    return FeatureUnion(transformer_list=selected_transformers)


def create_pipeline(config: Dict[str, Any], model_type: str, model_params: Dict[str, Any]) -> Pipeline:
    features_pipeline = create_features_pipeline(config['pipeline']['features'], config)
    model = create_model(model_type, model_params)
    steps = [
        ('preprocessor', features_pipeline),
        ('model', model)
    ]
    return Pipeline(steps)
