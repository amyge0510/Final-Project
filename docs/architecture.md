# Architecture Overview

## System Components

### 1. Data Processing Pipeline
- **Input**: Raw Amazon product data (JSON format)
- **Processing**: Data cleaning, normalization, and structuring
- **Output**: Processed data ready for graph construction

### 2. Knowledge Graph Construction
- **Database**: Neo4j graph database
- **Schema**:
  - Nodes: Product, Brand, Category, Feature, Review
  - Relationships: HAS_BRAND, BELONGS_TO, HAS_FEATURE, ALSO_BOUGHT, HAS_REVIEW
- **Constraints**: Unique constraints on primary keys

### 3. GraphRAG Implementation
- **Semantic Search**: Find relevant anchor nodes
- **Graph Traversal**: Multi-hop exploration from anchor nodes
- **Context Retrieval**: Extract relevant information from graph paths
- **LLM Integration**: Generate answers using retrieved context

### 4. API Layer
- **Framework**: FastAPI
- **Endpoints**:
  - POST /query: Submit queries to the knowledge graph
  - GET /health: Health check endpoint

## Data Flow

1. **Data Ingestion**
   ```
   Raw Data → Data Processing → Processed Data → Graph Construction → Neo4j
   ```

2. **Query Processing**
   ```
   User Query → Semantic Search → Graph Traversal → Context Retrieval → LLM → Answer
   ```

## Dependencies

- **Core**:
  - Neo4j: Graph database
  - LangChain: LLM integration
  - OpenAI: LLM provider
  - FastAPI: Web framework

- **Data Processing**:
  - Pandas: Data manipulation
  - PyYAML: Configuration management

- **Testing**:
  - pytest: Testing framework
  - unittest.mock: Mocking utilities

## Configuration

Configuration is managed through:
- Environment variables (.env)
- YAML files (config/neo4j_config.yaml)

## Security Considerations

1. **API Security**:
   - Environment variables for sensitive data
   - Input validation using Pydantic models

2. **Database Security**:
   - Secure Neo4j credentials
   - Connection encryption

3. **LLM Security**:
   - API key management
   - Input sanitization

## Scalability

The system is designed to scale horizontally:
- Neo4j clustering for database scalability
- API load balancing
- Caching mechanisms for frequent queries

## Monitoring and Logging

- API health checks
- Error logging
- Performance metrics
- Query analytics 