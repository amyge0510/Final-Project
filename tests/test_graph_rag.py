import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.graph_rag import GraphRAG

@pytest.fixture
def mock_driver():
    with patch('neo4j.GraphDatabase.driver') as mock:
        mock.return_value.session.return_value.__enter__.return_value.run.return_value = []
        yield mock

@pytest.fixture
def mock_llm():
    with patch('langchain.llms.OpenAI') as mock:
        mock.return_value.return_value = "Test answer"
        yield mock

@pytest.fixture
def graph_rag(mock_driver, mock_llm):
    return GraphRAG()

def test_get_graph_context(graph_rag, mock_driver):
    # Test with a simple query
    query = "test query"
    context = graph_rag.get_graph_context(query)
    
    assert isinstance(context, list)
    # Add more assertions based on expected behavior

def test_answer_query(graph_rag, mock_driver, mock_llm):
    # Test with a simple query
    query = "What are the best products?"
    answer = graph_rag.answer_query(query)
    
    assert isinstance(answer, str)
    assert answer == "Test answer"  # From mock_llm

def test_format_context(graph_rag):
    # Test context formatting
    context = [
        {
            "type": "product",
            "data": {
                "title": "Test Product",
                "description": "Test Description",
                "price": 99.99
            }
        },
        {
            "type": "review",
            "data": {
                "text": "Great product!",
                "rating": 5
            }
        }
    ]
    
    formatted = graph_rag._format_context(context)
    
    assert isinstance(formatted, str)
    assert "Test Product" in formatted
    assert "Test Description" in formatted
    assert "99.99" in formatted
    assert "Great product!" in formatted
    assert "5" in formatted

def test_close(graph_rag, mock_driver):
    # Test closing the driver
    graph_rag.close()
    mock_driver.return_value.close.assert_called_once() 