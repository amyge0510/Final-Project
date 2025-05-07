# Amazon Knowledge Graph with GraphRAG

A knowledge graph-based retrieval system for Amazon product data using Neo4j and GraphRAG.

## Project Overview

This project builds a knowledge graph from Amazon product data and implements GraphRAG (Graph-based Retrieval Augmented Generation) for intelligent product queries and recommendations.

### Key Features

- Amazon product data ingestion and graph construction
- Neo4j-based knowledge graph storage
- GraphRAG implementation for multi-hop retrieval
- Semantic search capabilities
- LLM-powered query answering

## Project Structure

```
.
├── data/                   # Raw and processed data
├── src/                    # Source code
│   ├── data_processing/    # Data ingestion and processing
│   ├── graph_construction/ # Neo4j graph building
│   ├── retrieval/         # GraphRAG implementation
│   └── api/               # API endpoints
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks for exploration
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Neo4j:
- Install Neo4j Desktop or use Neo4j Aura
- Update connection details in `config/neo4j_config.yaml`

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Usage

1. Data Ingestion:
```bash
python src/data_processing/ingest.py
```

2. Graph Construction:
```bash
python src/graph_construction/build_graph.py
```

3. Run the API:
```bash
python src/api/main.py
```

## Development

- Run tests: `pytest tests/`
- Format code: `black src/`
- Lint code: `flake8 src/`

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests. 