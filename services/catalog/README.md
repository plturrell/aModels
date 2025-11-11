# Semantic Metadata Catalog Service

The Semantic Metadata Catalog service provides a unified metadata registry using ISO/IEC 11179 standard and OWL/RDF semantics. It bridges Glean Catalog and the Extract service knowledge graph, enabling semantic interoperability and intelligent data discovery.

## Features

- **ISO 11179 Metamodel**: Standardized Data Element concepts (Data Element Concept + Representation)
- **OWL Ontology**: Semantic web representation of metadata with URIs and relationships
- **RDF Triplestore**: SPARQL endpoint for semantic queries (using Neo4j)
- **Glean Catalog Integration**: Bidirectional synchronization with Glean Catalog
- **Knowledge Graph Bridge**: Maps Neo4j graph nodes/edges to ISO 11179 Data Elements
- **SPARQL Query Endpoint**: Execute semantic queries against the catalog

## Architecture

### Components

1. **ISO 11179 Metamodel** (`iso11179/`): Core structures for metadata registry
2. **OWL Ontology** (`owl/`): OWL ontology generation and modeling
3. **Triplestore** (`triplestore/`): RDF storage and SPARQL query execution
4. **Glean Integration** (`glean/`): Glean Catalog synchronization
5. **Knowledge Graph Bridge** (`bridge/`): Neo4j to ISO 11179 mapping
6. **API** (`api/`): REST API handlers

## API Endpoints

### Data Elements

- `GET /catalog/data-elements` - List all data elements
- `GET /catalog/data-elements/{id}` - Get data element details
- `POST /catalog/data-elements` - Register new data element

### Semantic Search

- `POST /catalog/semantic-search` - Perform semantic search
  ```json
  {
    "query": "customer",
    "object_class": "Customer",
    "property": "Name",
    "source": "Extract Service"
  }
  ```

### SPARQL

- `GET /catalog/sparql?query=...` - Execute SPARQL query (GET)
- `POST /catalog/sparql` - Execute SPARQL query (POST)

### Ontology

- `GET /catalog/ontology` - Get OWL ontology metadata

## Configuration

Environment variables:

- `NEO4J_URI` - Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` - Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` - Neo4j password (default: `password`)
- `CATALOG_BASE_URI` - Base URI for catalog resources (default: `http://amodels.org/catalog`)
- `PORT` - Service port (default: `8084`)

## Usage

### Start the service

```bash
cd services/catalog
go run main.go
```

### Register a data element

```bash
curl -X POST http://localhost:8084/catalog/data-elements \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Age",
    "data_element_concept_id": "http://amodels.org/catalog/concept/customer_age",
    "representation_id": "http://amodels.org/catalog/representation/integer",
    "definition": "Age of a customer in years"
  }'
```

### Query via SPARQL

```bash
curl "http://localhost:8084/catalog/sparql?query=SELECT%20?element%20WHERE%20%7B%20?element%20rdf:type%20iso11179:DataElement%20.%20%7D"
```

## Integration

The catalog service integrates with:

1. **Extract Service**: Maps Neo4j knowledge graph to ISO 11179
2. **Glean Catalog**: Bidirectional synchronization with Glean facts
3. **Gateway**: All endpoints are proxied through the gateway at `/catalog/*`

## Development

### Build

```bash
cd services/catalog
go build -o catalog-service main.go
```

### Run

```bash
NEO4J_URI="bolt://localhost:7687" \
NEO4J_USERNAME="neo4j" \
NEO4J_PASSWORD="password" \
CATALOG_BASE_URI="http://amodels.org/catalog" \
./catalog-service
```

## Dependencies

- `github.com/neo4j/neo4j-go-driver/v5` - Neo4j driver
- Standard library: `encoding/json`, `net/http`, `log`

## Future Enhancements

- Full SPARQL parser (currently uses simplified translation)
- Open Deep Research integration for metadata discovery
- Advanced semantic reasoning
- Agent-based metadata operations

