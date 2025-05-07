import os
import json
import pytest
from pathlib import Path
from src.evaluation.ground_truth import GroundTruthBuilder

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data files for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample products data
    products = [
        {
            "asin": "B001",
            "title": "Test Book 1",
            "category": "Fiction",
            "rating": 4.5,
            "salesrank": 100
        },
        {
            "asin": "B002",
            "title": "Test Book 2",
            "category": "Fiction",
            "rating": 4.0,
            "salesrank": 200
        }
    ]
    
    with open(data_dir / "products.csv", 'w') as f:
        f.write("asin,title,category,rating,salesrank\n")
        for p in products:
            f.write(f"{p['asin']},{p['title']},{p['category']},{p['rating']},{p['salesrank']}\n")
    
    # Create sample reviews data
    reviews = [
        {"product_asin": "B001", "user_id": "U1", "rating": 5},
        {"product_asin": "B002", "user_id": "U1", "rating": 4}
    ]
    
    with open(data_dir / "reviews.csv", 'w') as f:
        f.write("product_asin,user_id,rating\n")
        for r in reviews:
            f.write(f"{r['product_asin']},{r['user_id']},{r['rating']}\n")
    
    # Create sample co-purchase data
    copurchases = [
        {"source_asin": "B001", "target_asin": "B002"}
    ]
    
    with open(data_dir / "copurchase.csv", 'w') as f:
        f.write("source_asin,target_asin\n")
        for c in copurchases:
            f.write(f"{c['source_asin']},{c['target_asin']}\n")
    
    return str(data_dir)

@pytest.fixture
def mock_config(tmp_path):
    """Create mock configuration file."""
    config = {
        "evaluation": {
            "relationship_queries": {
                "co_purchase": {
                    "min_copurchases": 1  # Lower threshold to ensure co-purchase queries are generated
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
        import yaml
        yaml.dump(config, f)
    
    return str(config_path)

def test_ground_truth_builder_initialization(mock_config):
    """Test GroundTruthBuilder initialization."""
    builder = GroundTruthBuilder(config_path=mock_config)
    assert builder.config is not None
    assert "evaluation" in builder.config

def test_build_relationship_ground_truth(sample_data, mock_config):
    """Test relationship query ground truth generation."""
    builder = GroundTruthBuilder(config_path=mock_config)
    builder.load_data(
        products_path=os.path.join(sample_data, "products.csv"),
        reviews_path=os.path.join(sample_data, "reviews.csv"),
        copurchase_path=os.path.join(sample_data, "copurchase.csv")
    )
    
    relationship_queries = builder._build_relationship_ground_truth()
    assert len(relationship_queries) > 0
    for query in relationship_queries:
        assert "query" in query
        assert query["type"] == "relationship"
        assert "expected_asins" in query
        assert isinstance(query["expected_asins"], list)
        assert "metadata" in query
        assert "source_title" in query["metadata"]
        assert "source_category" in query["metadata"]

def test_build_attribute_ground_truth(sample_data, mock_config):
    """Test attribute query ground truth generation."""
    builder = GroundTruthBuilder(config_path=mock_config)
    builder.load_data(
        products_path=os.path.join(sample_data, "products.csv"),
        reviews_path=os.path.join(sample_data, "reviews.csv"),
        copurchase_path=os.path.join(sample_data, "copurchase.csv")
    )
    
    attribute_queries = builder._build_attribute_ground_truth()
    assert len(attribute_queries) > 0
    for query in attribute_queries:
        assert "query" in query
        assert query["type"] == "attribute"
        assert "expected_asins" in query
        assert isinstance(query["expected_asins"], list)
        assert "metadata" in query
        assert "category" in query["metadata"]
        assert "expected_titles" in query["metadata"]

def test_save_ground_truth(sample_data, mock_config, tmp_path):
    """Test saving ground truth data."""
    builder = GroundTruthBuilder(config_path=mock_config)
    builder.load_data(
        products_path=os.path.join(sample_data, "products.csv"),
        reviews_path=os.path.join(sample_data, "reviews.csv"),
        copurchase_path=os.path.join(sample_data, "copurchase.csv")
    )
    
    output_path = str(tmp_path / "ground_truth.json")
    builder.save_ground_truth(output_path)
    
    assert os.path.exists(output_path)
    with open(output_path, 'r') as f:
        data = json.load(f)
        assert "relationship_queries" in data
        assert "attribute_queries" in data
        assert "metadata" in data

def test_ground_truth_query_distribution(sample_data, mock_config):
    """Test distribution of query types in ground truth."""
    builder = GroundTruthBuilder(config_path=mock_config)
    builder.load_data(
        products_path=os.path.join(sample_data, "products.csv"),
        reviews_path=os.path.join(sample_data, "reviews.csv"),
        copurchase_path=os.path.join(sample_data, "copurchase.csv")
    )
    
    ground_truth = builder.build_ground_truth()
    
    # Check query type distribution
    assert len(ground_truth["relationship_queries"]) > 0
    assert len(ground_truth["attribute_queries"]) > 0
    
    # Check subtypes
    relationship_subtypes = set(q["subtype"] for q in ground_truth["relationship_queries"])
    attribute_subtypes = set(q["subtype"] for q in ground_truth["attribute_queries"])
    
    assert "co_purchase" in relationship_subtypes
    assert "review_pattern" in relationship_subtypes
    assert "rating" in attribute_subtypes
    assert "bestseller" in attribute_subtypes

def test_ground_truth_metadata_completeness(sample_data, mock_config):
    """Test completeness of metadata in ground truth queries."""
    builder = GroundTruthBuilder(config_path=mock_config)
    builder.load_data(
        products_path=os.path.join(sample_data, "products.csv"),
        reviews_path=os.path.join(sample_data, "reviews.csv"),
        copurchase_path=os.path.join(sample_data, "copurchase.csv")
    )
    
    ground_truth = builder.build_ground_truth()
    
    # Check metadata for each query type
    for query in ground_truth["relationship_queries"]:
        assert "metadata" in query
        assert "source_title" in query["metadata"]
        assert "source_category" in query["metadata"]
    
    for query in ground_truth["attribute_queries"]:
        assert "metadata" in query
        assert "category" in query["metadata"]
        assert "expected_titles" in query["metadata"] 