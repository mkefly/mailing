import os
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
import xgboost as xgb

from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec

from loguru import logger
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
import logging

# Setup logging with Loguru
logger.add("model_pipeline.log", rotation="500 MB")


# Configuration Functions
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_hyperopt_space(space: Dict[str, Any]) -> Dict[str, Any]:
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


# Custom Transformers
class BERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, batch_size: int = 32, max_length: int = 128):
        self.feature = feature
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
        torch.set_printoptions(profile="default")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def transform(self, X: pd.DataFrame):
        embeddings = []
        for i in range(0, len(X), self.batch_size):
            batch_texts = X[self.feature].iloc[i:i+self.batch_size].tolist()
            inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, **kwargs: Any):
        self.feature = feature
        self.vector_size = kwargs.get('vector_size', 100)
        self.window = kwargs.get('window', 5)
        self.min_count = kwargs.get('min_count', 1)

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'Word2VecTransformer':
        sentences = [text.split() for text in X[self.feature]]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        embeddings = []
        for text in X[self.feature]:
            words = text.split()
            word_vecs = [self.model.wv[word] for word in words if word in self.model.wv]
            if word_vecs:
                embeddings.append(np.mean(word_vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class TFIDFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, **kwargs: Any):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.feature = feature
        self.max_features = kwargs.get('max_features', 1000)
        self.ngram_range = tuple(kwargs.get('ngram_range', (1, 2)))
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'TFIDFTransformer':
        self.vectorizer.fit(X[self.feature])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.vectorizer.transform(X[self.feature]).toarray()


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.current_date = datetime.now()

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'DateFeatureTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'date' not in X.columns:
            raise ValueError("The 'date' column is missing from the input data.")
        X['days_since_published'] = X['date'].apply(lambda x: (self.current_date - pd.to_datetime(x)).days)
        return X


# Model and Pipeline Factories
def create_model(model_type: str, params: Dict[str, Any]) -> BaseEstimator:
    model_map = {
        'lgbm': lgb.LGBMClassifier,
        'xgboost': xgb.XGBClassifier,
        'random_forest': RandomForestClassifier,
        'sgd': SGDClassifier,
        'lambdamart': lgb.LGBMRanker
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

    return transformer_map[embedding_type](feature=feature, **config.get(f'{embedding_type}_params', {}))


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
        ('features', features_pipeline),
        ('model', model)
    ]
    return Pipeline(steps)


# Data Preparation
def split_data(articles: pd.DataFrame, user_interactions: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    merged_data = user_interactions.merge(articles, on='article_id')
    train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)
    X_train = train_data.drop(columns=['user_id', 'interaction_date'])
    y_train = train_data['interaction'].values
    X_test = test_data.drop(columns=['user_id', 'interaction_date'])
    y_test = test_data['interaction'].values
    return X_train, y_train, X_test, y_test


# Hyperparameter Optimization
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


# Model Evaluation
def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, k: int = 10, threshold: float = None) -> Dict[str, float]:
    def calculate_metrics(likelihood_scores, y_true):
        top_k_indices = np.argsort(likelihood_scores)[-k:]
        precision_at_k = np.mean(y_true[top_k_indices])

        sorted_indices = np.argsort(-likelihood_scores)
        sorted_labels = y_true[sorted_indices]
        average_precision_at_k = np.mean([
            precision_score(sorted_labels[:i+1], np.ones(i+1))
            for i in range(k) if sorted_labels[i]
        ])

        ndcg_at_k = ndcg_score([y_true], [likelihood_scores], k=k)
        auc_roc = roc_auc_score(y_true, likelihood_scores)
        avg_precision_score = average_precision_score(y_true, likelihood_scores)

        return precision_at_k, average_precision_at_k, ndcg_at_k, auc_roc, avg_precision_score

    def calculate_threshold_metrics(likelihood_scores, y_true, threshold):
        predicted_labels = (likelihood_scores >= threshold).astype(int)
        precision = precision_score(y_true, predicted_labels)
        recall = recall_score(y_true, predicted_labels)
        f1 = f1_score(y_true, predicted_labels)
        return precision, recall, f1

    likelihood_scores = pipeline.predict_proba(X)[:, 1]

    if not threshold:
        thresholds = np.linspace(0.1, 0.9, 9)
        best_threshold = 0.0
        best_f1_score = 0.0
        for th in thresholds:
            _, _, f1 = calculate_threshold_metrics(likelihood_scores, y, th)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = th
    else:
        best_threshold = threshold

    metrics = calculate_metrics(likelihood_scores, y)
    precision, recall, f1 = calculate_threshold_metrics(likelihood_scores, y, best_threshold)

    return {
        'precision_at_k': metrics[0],
        'map_at_k': metrics[1],
        'ndcg_at_k': metrics[2],
        'auc_roc': metrics[3],
        'average_precision_score': metrics[4],
        'best_threshold': best_threshold,
        'precision_at_best_threshold': precision,
        'recall_at_best_threshold': recall,
        'f1_score_at_best_threshold': f1,
    }


# Model Training
def train_model(X_train: pd.DataFrame, y_train: np.ndarray, config: Dict[str, Any], optimize: bool) -> Tuple[Pipeline, Dict[str, Any]]:
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


# Main Experiment Workflow
def run_experiment(articles: pd.DataFrame, user_interactions: pd.DataFrame, config_path: str, optimize: bool = False):
    config = load_yaml_config(config_path)
    X_train, y_train, X_test, y_test = split_data(articles, user_interactions)

    pipeline, cv_results = train_model(X_train, y_train, config, optimize)

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


# Generate Data using Faker
from faker import Faker

fake = Faker()

# Generate a larger articles dataset
n_articles = 1000
articles = pd.DataFrame({
    'article_id': np.arange(1, n_articles + 1),
    'title': [f"Title {i}" for i in range(1, n_articles + 1)],
    'content': [fake.text(max_nb_chars=200) for _ in range(n_articles)],
    'tags': [", ".join(fake.words(nb=3)) for _ in range(n_articles)],
    'date': [fake.date_this_decade().strftime("%Y-%m-%d") for _ in range(n_articles)]
})

# Generate a larger user interactions dataset
n_users = 50
user_interactions = pd.DataFrame({
    'user_id': np.random.randint(1, n_users + 1, size=n_articles * 5),
    'article_id': np.random.randint(1, n_articles + 1, size=n_articles * 5),
    'interaction_date': [fake.date_this_year().strftime("%Y-%m-%d") for _ in range(n_articles * 5)],
    'interaction': np.random.randint(0, 2, size=n_articles * 5)
})

# Run Experiment
run_experiment(articles, user_interactions, config_path='config.yaml')
