import pytest
import pandas as pd
from livedesk_mailing_list.training.split import split_data

@pytest.fixture
def articles_data():
    return pd.DataFrame({
        'article_id': [1, 2, 3],
        'title': ['Article 1', 'Article 2', 'Article 3'],
        'content': ['Content 1', 'Content 2', 'Content 3']
    })

@pytest.fixture
def user_interactions_data():
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4],
        'article_id': [1, 2, 3, 1],
        'interaction': [1, 0, 1, 1],
        'interaction_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    })

def test_split_data_shape(articles_data, user_interactions_data):
    X_train, y_train, X_test, y_test = split_data(articles_data, user_interactions_data)
    
    # Check that the split sizes are correct
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert len(y_train) == 3
    assert len(y_test) == 1

def test_split_data_columns(articles_data, user_interactions_data):
    X_train, _, X_test, _ = split_data(articles_data, user_interactions_data)
    
    # Check that the correct columns are in X_train and X_test
    assert 'user_id' not in X_train.columns
    assert 'interaction_date' not in X_train.columns
    assert 'user_id' not in X_test.columns
    assert 'interaction_date' not in X_test.columns
    
    # Ensure interaction column is not in features
    assert 'interaction' not in X_train.columns
    assert 'interaction' not in X_test.columns

def test_split_data_empty():
    articles = pd.DataFrame(columns=['article_id', 'title', 'content'])
    user_interactions = pd.DataFrame(columns=['user_id', 'article_id', 'interaction', 'interaction_date'])
    
    X_train, y_train, X_test, y_test = split_data(articles, user_interactions)
    
    # Check that the result is empty arrays and DataFrames
    assert X_train.empty
    assert X_test.empty
    assert y_train.size == 0
    assert y_test.size == 0

def test_split_data_no_matching_articles():
    articles = pd.DataFrame({
        'article_id': [10, 20, 30],
        'title': ['Article 10', 'Article 20', 'Article 30'],
        'content': ['Content 10', 'Content 20', 'Content 30']
    })
    user_interactions = pd.DataFrame({
        'user_id': [1, 2, 3],
        'article_id': [1, 2, 3],
        'interaction': [1, 0, 1],
        'interaction_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    
    X_train, y_train, X_test, y_test = split_data(articles, user_interactions)
    
    # Since there's no match, the resulting DataFrames should be empty
    assert X_train.empty
    assert X_test.empty
    assert y_train.size == 0
    assert y_test.size == 0
