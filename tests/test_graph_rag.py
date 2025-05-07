"""Tests for the GraphRAG class."""
import pytest
from unittest.mock import MagicMock, patch
import os
import yaml
from src.retrieval.graph_rag import GraphRAG

@pytest.fixture
def mock_driver():
    """Mock Neo4j driver."""
    return MagicMock()

@pytest.fixture
def mock_llm():
    """Mock OpenAI LLM."""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value="Test answer based on context")
    mock.return_value = "Test answer based on context"  # For backward compatibility
    return mock

@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1536  # OpenAI embedding dimension
    mock.embed_documents.return_value = [[0.1] * 1536]  # Return list of embeddings for documents
    return mock

@pytest.fixture
def mock_config(tmp_path):
    """Create mock configuration file."""
    config = {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "test"
        },
        "openai": {
            "api_key": "test_key"
        },
        "retrieval": {
            "max_results": 5,
            "similarity_threshold": 0.7
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)

@pytest.fixture
def graph_rag(mock_config, mock_driver):
    """Create GraphRAG instance with real OpenAI API."""
    with patch('neo4j.GraphDatabase') as mock_graph_db:
        mock_graph_db.driver.return_value = mock_driver
        rag = GraphRAG(config_path=mock_config)
        rag.driver = mock_driver  # Ensure the mock driver is used
        return rag

def test_get_graph_context(graph_rag, mock_driver):
    """Test getting context from graph."""
    # Mock embeddings
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536  # OpenAI embedding dimension
    graph_rag.embeddings = mock_embeddings

    # Mock query result for anchor nodes
    mock_anchor_result = [
        {
            "asin": "B001",
            "title": "Test Book",
            "category": "Fiction",
            "description": "A test book",
            "price": 9.99,
            "score": 0.8
        }
    ]
    
    # Mock query result for product context
    mock_product_result = [{
        "product": {
            "asin": "B001",
            "title": "Test Book",
            "category": "Fiction",
            "description": "A test book",
            "price": 9.99
        },
        "related_products": []
    }]
    
    # Set up session and transaction mocks
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_run = MagicMock()
    mock_run.side_effect = [mock_anchor_result, mock_product_result]
    mock_tx.run = mock_run
    
    # Configure session to return transaction
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    context = graph_rag.get_graph_context("test query")
    assert len(context) > 0
    assert context[0]["type"] == "product"
    assert context[0]["data"]["asin"] == "B001"

def test_answer_query(graph_rag, mock_driver):
    """Test answering a query."""
    # Mock query result for anchor nodes
    mock_anchor_result = [
        {
            "asin": "B001",
            "title": "Test Book",
            "category": "Fiction",
            "description": "A test book",
            "price": 9.99,
            "score": 0.8
        }
    ]
    
    # Mock query result for product context
    mock_product_result = [{
        "product": {
            "asin": "B001",
            "title": "Test Book",
            "category": "Fiction",
            "description": "A test book",
            "price": 9.99
        },
        "related_products": []
    }]
    
    mock_session = MagicMock()
    mock_run = MagicMock()
    mock_run.side_effect = [mock_anchor_result, mock_product_result]
    mock_session.__enter__.return_value.run = mock_run
    mock_driver.session.return_value = mock_session
    
    result = graph_rag.answer_query("What is Test Book about?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "context" in result
    assert "retrieved_asins" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0

def test_format_context(graph_rag):
    """Test context formatting."""
    context = [
        {
            "type": "product",
            "data": {
                "asin": "B001",
                "title": "Test Book",
                "category": "Fiction",
                "rating": 4.5,
                "description": "A test book",
                "price": 9.99
            }
        }
    ]
    
    formatted = graph_rag._format_context(context)
    assert isinstance(formatted, str)
    assert "Test Book" in formatted
    assert "Fiction" in formatted
    assert "A test book" in formatted
    assert "9.99" in formatted

def test_close(graph_rag, mock_driver):
    """Test closing the connection."""
    graph_rag.close()
    mock_driver.close.assert_called_once() 