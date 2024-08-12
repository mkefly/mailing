import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer, BertModel
from livedesk_mailing_list.models.transformers import BERTTransformer, Word2VecTransformer, TFIDFTransformer

@pytest.fixture
def sample_data():
    """Fixture that provides sample data for testing."""
    return pd.DataFrame({
        'text': [
            "This is a test sentence.",
            "Another test sentence.",
            "Yet another sentence for testing.",
            "Short sentence.",
            "Very very very long sentence that might be truncated in BERT."
        ]
    })

def test_bert_transformer(sample_data):
    bert_transformer = BERTTransformer(feature='text')
    
    # Test fit method
    bert_transformer.fit(sample_data)
    
    # Test transform method
    transformed = bert_transformer.transform(sample_data)
    
    # Check the shape of the transformed output
    assert transformed.shape == (len(sample_data), 768)

def test_word2vec_transformer(sample_data):
    w2v_transformer = Word2VecTransformer(feature='text')
    
    # Test fit method
    w2v_transformer.fit(sample_data)
    
    # Test transform method
    transformed = w2v_transformer.transform(sample_data)
    
    # Check the shape of the transformed output
    assert transformed.shape == (len(sample_data), w2v_transformer.vector_size)

def test_tfidf_transformer(sample_data):
    tfidf_transformer = TFIDFTransformer(feature='text', max_features=10)
    
    # Test fit method
    tfidf_transformer.fit(sample_data)
    
    # Test transform method
    transformed = tfidf_transformer.transform(sample_data)
    
    # Check the shape of the transformed output
    assert transformed.shape == (len(sample_data), 10)

def test_empty_dataframe():
    empty_data = pd.DataFrame({'text': []})
    
    # Instantiate transformers
    bert_transformer = BERTTransformer(feature='text')
    w2v_transformer = Word2VecTransformer(feature='text')
    tfidf_transformer = TFIDFTransformer(feature='text')
    
    # Test if transformers handle empty DataFrame
    bert_transformer.fit(empty_data)
    assert bert_transformer.transform(empty_data).shape == (0, 768)
    
    w2v_transformer.fit(empty_data)
    assert w2v_transformer.transform(empty_data).shape == (0, w2v_transformer.vector_size)
    
    tfidf_transformer.fit(empty_data)
    assert tfidf_transformer.transform(empty_data).shape == (0, 10)

def test_pipeline_integration(sample_data):
    pipeline = Pipeline([
        ('bert', BERTTransformer(feature='text')),
        ('tfidf', TFIDFTransformer(feature='text', max_features=10))
    ])
    
    # Test pipeline fitting
    pipeline.fit(sample_data)
    
    # Test pipeline transformation
    transformed = pipeline.transform(sample_data)
    
    # Check the shape of the transformed output
    assert transformed.shape == (len(sample_data), 10)

if __name__ == '__main__':
    pytest.main()