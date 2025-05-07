"""Test configuration and fixtures."""
import os
import pytest
import yaml
import pandas as pd

@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration file."""
    config = {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "test"
        },
        "openai": {
            "api_key": "test_key"
        },
        "evaluation": {
            "relationship_queries": {
                "co_purchase": {
                    "min_copurchases": 2
                },
                "review_pattern": {
                    "min_rating": 4.0
                }
            },
            "attribute_queries": {
                "rating": {
                    "min_rating": 4.5
                },
                "bestseller": {
                    "max_salesrank": 1000
                }
            }
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data files for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample products data
    products = pd.DataFrame({
        "asin": ["B001", "B002", "B003"],
        "title": ["Test Book 1", "Test Book 2", "Test Book 3"],
        "category": ["Fiction", "Fiction", "Non-Fiction"],
        "rating": [4.5, 4.0, 3.5],
        "salesrank": [100, 200, 300]
    })
    products.to_csv(data_dir / "products.csv", index=False)
    
    # Create sample reviews data
    reviews = pd.DataFrame({
        "product_asin": ["B001", "B002", "B001"],
        "user_id": ["U1", "U1", "U2"],
        "rating": [5, 4, 5]
    })
    reviews.to_csv(data_dir / "reviews.csv", index=False)
    
    # Create sample co-purchase data
    copurchases = pd.DataFrame({
        "source_asin": ["B001", "B001", "B002"],
        "target_asin": ["B002", "B003", "B003"]
    })
    copurchases.to_csv(data_dir / "copurchase.csv", index=False)
    
    return str(data_dir)

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
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