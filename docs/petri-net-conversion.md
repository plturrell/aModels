# Petri Net Conversion for Control-M Workflows

## Overview

The Extract service now converts Control-M job definitions into **Petri nets**, providing a formal workflow representation that can be:
1. Stored in the catalog
2. Added to the knowledge graph
3. Used for AgentFlow/LangFlow conversion

## Petri Net Structure

A Petri net consists of:
- **Places**: Represent states/conditions (Control-M conditions)
- **Transitions**: Represent actions/events (Control-M jobs)
- **Arcs**: Represent connections between places and transitions
- **Tokens**: Represent workflow state (not shown in static representation)

## Control-M to Petri Net Mapping

### Places (States/Conditions)

**Input Conditions (InCond)** → **Places**
- Each `InCond` becomes a place
- Type: `condition` (input)
- Properties: condition name, odate, sign, and_or

**Output Conditions (OutCond)** → **Places**
- Each `OutCond` becomes a place
- Type: `condition` (output)
- Properties: condition name, odate, sign, type

**Initial State** → **Place**
- Special place for jobs without input conditions
- Type: `initial`
- Initial tokens: 1

### Transitions (Actions/Jobs)

**Control-M Jobs** → **Transitions**
- Each `JOB` becomes a transition
- Type: `job`
- Properties: job metadata (command, description, schedule, etc.)
- **SubProcesses**: SQL statements within the job

### Arcs (Connections)

**InCond → Job** → **Place-to-Transition Arc**
- Arc type: `place_to_transition`
- Weight: 1 (default)
- Properties: condition name, and_or logic

**Job → OutCond** → **Transition-to-Place Arc**
- Arc type: `transition_to_place`
- Weight: 1 (default)
- Properties: condition name

**Initial → Job (no InCond)** → **Place-to-Transition Arc**
- For jobs that can start immediately

## SQL Subprocesses

Individual SQL statements are embedded as **subprocesses** within transitions:

```json
{
  "id": "transition:job:load_orders",
  "label": "load_orders",
  "type": "job",
  "sub_processes": [
    {
      "id": "subprocess:load_orders:sql:0",
      "type": "sql",
      "label": "SQL 1",
      "content": "INSERT INTO orders SELECT * FROM staging_orders",
      "properties": {
        "sql_query": "INSERT INTO orders SELECT * FROM staging_orders",
        "order": 0
      }
    }
  ]
}
```

This allows:
1. **Granular workflow understanding**: See individual SQL operations within jobs
2. **AgentFlow conversion**: Each SQL subprocess can become an agent node
3. **LangFlow conversion**: SQL subprocesses map to LangFlow nodes

## Knowledge Graph Integration

Petri net elements are added to the knowledge graph as:

### Nodes

1. **Petri Net Root Node**
   - Type: `petri_net`
   - Properties: ID, name, place_count, transition_count, arc_count

2. **Place Nodes**
   - Type: `petri_place`
   - Properties: place_id, place_type, initial_tokens, properties
   - Connected to root via `HAS_PLACE` edge

3. **Transition Nodes**
   - Type: `petri_transition`
   - Properties: transition_id, transition_type, properties, subprocess_count
   - Connected to root via `HAS_TRANSITION` edge

4. **Subprocess Nodes**
   - Type: `petri_subprocess`
   - Properties: subprocess_id, subprocess_type, content, properties
   - Connected to transition via `HAS_SUBPROCESS` edge

### Edges

1. **HAS_PLACE**: Root → Place
2. **HAS_TRANSITION**: Root → Transition
3. **HAS_SUBPROCESS**: Transition → Subprocess
4. **PETRI_ARC**: Place ↔ Transition (workflow connections)
5. **HAS_PETRI_NET**: Root node → Petri net root

## Catalog Integration

Petri nets are stored in the catalog under `petri_nets`:

```json
{
  "petri_nets": {
    "controlm_petri_net": {
      "id": "controlm_petri_net",
      "name": "Control-M Workflow Petri Net",
      "type": "petri_net",
      "places": [...],
      "transitions": [...],
      "arcs": [...],
      "metadata": {
        "source": "controlm",
        "job_count": 5
      },
      "statistics": {
        "place_count": 8,
        "transition_count": 5,
        "arc_count": 12
      }
    }
  }
}
```

## Usage

### Automatic Conversion

Petri net conversion happens automatically when processing Control-M files:

```bash
POST /knowledge-graph
{
  "control_m_files": ["data/jobs.xml"],
  "sql_queries": [
    "INSERT INTO orders SELECT * FROM staging_orders",
    "UPDATE orders SET status = 'processed' WHERE ..."
  ]
}
```

### Querying Petri Nets

**Find all Petri nets:**
```cypher
MATCH (n) WHERE n.type = 'petri_net'
RETURN n.label, n.props
```

**Find transitions with SQL subprocesses:**
```cypher
MATCH (t:Node)-[:HAS_SUBPROCESS]->(s:Node)
WHERE t.type = 'petri_transition' 
  AND s.type = 'petri_subprocess'
  AND s.props.subprocess_type = 'sql'
RETURN t.label, s.props.content
```

**Find workflow paths:**
```cypher
MATCH path = (p1:Node)-[:PETRI_ARC]->(t:Node)-[:PETRI_ARC]->(p2:Node)
WHERE p1.type = 'petri_place' AND t.type = 'petri_transition'
RETURN p1.label, t.label, p2.label
```

## AgentFlow/LangFlow Conversion

The Petri net structure provides a direct mapping to AgentFlow/LangFlow:

### Mapping Strategy

1. **Places** → **Conditional Nodes**
   - Check if condition is met before proceeding

2. **Transitions** → **Agent Nodes**
   - Each job becomes an agent node
   - Agent executes the job logic

3. **SQL Subprocesses** → **SQL Agent Nodes**
   - Each SQL subprocess becomes a dedicated SQL agent
   - Embedded within the parent job agent

4. **Arcs** → **Flow Connections**
   - Direct connections between nodes
   - Conditional edges based on place conditions

### Example Conversion

**Control-M Job:**
```xml
<JOB JOBNAME="load_orders">
  <INCOND NAME="staging_ready"/>
  <OUTCOND NAME="orders_loaded"/>
  <COMMAND>sqlplus ... INSERT INTO orders...</COMMAND>
</JOB>
```

**Petri Net:**
- Place: `staging_ready`
- Transition: `load_orders` (with SQL subprocess)
- Place: `orders_loaded`
- Arcs: `staging_ready → load_orders → orders_loaded`

**AgentFlow/LangFlow:**
- Condition Node: Check `staging_ready`
- Agent Node: `load_orders` (executes SQL)
- Condition Node: Set `orders_loaded`
- Flow: `staging_ready` → `load_orders` → `orders_loaded`

## Benefits

1. **Formal Workflow Representation**: Petri nets provide mathematical rigor
2. **Visual Understanding**: Clear workflow visualization
3. **Conversion Ready**: Direct mapping to AgentFlow/LangFlow
4. **Subprocess Granularity**: SQL statements as embedded subprocesses
5. **Catalog Integration**: Stored for future reference
6. **Knowledge Graph**: Queryable workflow patterns

## Future Enhancements

1. **Petri Net Analysis**: Deadlock detection, reachability analysis
2. **Visualization**: Export to Graphviz/DOT format
3. **AgentFlow Generator**: Automatic LangFlow JSON generation
4. **SQL Pattern Recognition**: Identify common SQL patterns in subprocesses
5. **Workflow Optimization**: Suggest improvements based on Petri net structure

