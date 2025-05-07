"""Tests for the GPTScorer class."""
import pytest
import json
import os
from unittest.mock import MagicMock, patch
from src.evaluation.gpt_scorer import GPTScorer

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [{
            "message": {
                "content": """
                Relevance: 5 - The answer directly addresses the query
                Completeness: 4 - Provides comprehensive information
                Accuracy: 5 - Information is accurate based on context
                Coherence: 4 - Well-structured and clear
                Use of Context: 4 - Effectively uses available context
                """
            }
        }]
    }

@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What books are similar to 'The Lord of the Rings'?"

@pytest.fixture
def sample_answer():
    """Sample answer for testing."""
    return "Based on the context, books similar to 'The Lord of the Rings' include 'The Hobbit' and 'The Silmarillion', which are also fantasy novels by J.R.R. Tolkien featuring similar themes of epic quests and mythical worlds."

@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return [
        {
            'asin': '0345538374',
            'author': 'J.R.R. Tolkien',
            'category': 'Books > Fantasy',
            'title': 'The Lord of the Rings'
        },
        {
            'asin': '0345534835',
            'author': 'J.R.R. Tolkien',
            'category': 'Books > Fantasy',
            'title': 'The Hobbit'
        }
    ]

@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing."""
    return {
        "results": {
            "graph_rag": {
                "relationship": [
                    {
                        "query": "What books are similar to 'The Lord of the Rings'?",
                        "answer": "Based on the context, similar books include 'The Hobbit'.",
                        "context": [{"asin": "0345534835", "title": "The Hobbit"}]
                    }
                ],
                "attribute": [
                    {
                        "query": "What are the highest-rated fantasy books?",
                        "answer": "The Lord of the Rings is a highly rated fantasy novel.",
                        "context": [{"asin": "0345538374", "title": "The Lord of the Rings"}]
                    }
                ]
            },
            "semantic": {
                "relationship": [
                    {
                        "query": "What books are similar to 'The Lord of the Rings'?",
                        "answer": "Similar books include other fantasy novels.",
                        "context": [{"asin": "0345534835", "title": "The Hobbit"}]
                    }
                ],
                "attribute": [
                    {
                        "query": "What are the highest-rated fantasy books?",
                        "answer": "Several fantasy books have high ratings.",
                        "context": [{"asin": "0345538374", "title": "The Lord of the Rings"}]
                    }
                ]
            }
        }
    }

def test_gpt_scorer_initialization():
    """Test GPTScorer initialization with API key."""
    scorer = GPTScorer(api_key='test_key')
    assert scorer.api_key == 'test_key'

def test_gpt_scorer_initialization_no_key():
    """Test GPTScorer initialization without API key."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'env_test_key'}):
        scorer = GPTScorer()
        assert scorer.api_key == 'env_test_key'

def test_create_scoring_prompt():
    """Test creation of scoring prompt."""
    scorer = GPTScorer(api_key='test_key')
    prompt = scorer._create_scoring_prompt(
        "What books are similar to X?",
        "Books A and B are similar.",
        [{"title": "Book A"}, {"title": "Book B"}]
    )
    assert isinstance(prompt, str)
    assert "Query:" in prompt
    assert "Answer:" in prompt
    assert "Context:" in prompt

def test_score_answer(sample_query, sample_answer, sample_context):
    """Test scoring a single answer."""
    scorer = GPTScorer()  # Will use environment API key
    result = scorer.score_answer(sample_query, sample_answer, sample_context)
    
    assert 'scores' in result
    assert 'explanations' in result
    assert 'raw_response' in result
    
    scores = result['scores']
    assert 'relevance' in scores
    assert 'completeness' in scores
    assert 'accuracy' in scores
    assert 'coherence' in scores
    assert 'use_of_context' in scores
    
    assert all(isinstance(score, int) for score in scores.values())
    assert all(1 <= score <= 5 for score in scores.values())

def test_score_evaluation_results(sample_evaluation_results, tmp_path):
    """Test scoring full evaluation results."""
    # Create temporary files
    results_path = tmp_path / "evaluation_results.json"
    output_path = tmp_path / "gpt_scores.json"
    
    with open(results_path, 'w') as f:
        json.dump(sample_evaluation_results, f)
    
    scorer = GPTScorer()  # Will use environment API key
    scored_results = scorer.score_evaluation_results(str(results_path), str(output_path))
    
    assert os.path.exists(output_path)
    assert "timestamp" in scored_results
    assert "original_results" in scored_results
    assert "gpt_scores" in scored_results
    
    # Check aggregated scores
    for method in ["graph_rag", "semantic"]:
        for query_type in ["relationship", "attribute"]:
            aggregate_key = f"{query_type}_aggregate"
            assert aggregate_key in scored_results["gpt_scores"][method]
            aggregate_scores = scored_results["gpt_scores"][method][aggregate_key]
            assert "relevance" in aggregate_scores
            assert "completeness" in aggregate_scores
            assert "accuracy" in aggregate_scores
            assert "coherence" in aggregate_scores
            assert "use_of_context" in aggregate_scores

def test_error_handling():
    """Test error handling in GPTScorer."""
    scorer = GPTScorer(api_key='test_key')
    
    # Test with invalid inputs
    result = scorer.score_answer(None, None, None)
    assert 'error' in result
    
    result = scorer.score_answer("", "", [])
    assert 'error' in result
    
    # Test with missing context
    result = scorer.score_answer("test query", "test answer", None)
    assert 'error' in result 