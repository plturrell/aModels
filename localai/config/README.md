# Domain Configuration

This directory contains the configuration files that define the AI agent domains and their behavior.

## `domains.json`

This is the most important configuration file in the system. It contains a JSON object that defines all the agent domains that the server will load and manage. By externalizing the domain definitions into this file, we can easily add, remove, or modify agents without needing to recompile the application.

### Schema

The file contains a single root object with a `domains` key. The value of this key is an object where each key is a unique agent ID and the value is an object containing the domain's properties.

**Domain Object Properties:**

| Field               | Type           | Description                                                                                      |
| ------------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| `name`              | `string`       | The human-readable name of the agent domain (e.g., "SQL Agent").                                   |
| `layer`             | `string`       | The architectural layer the agent belongs to (e.g., "layer1", "layer2").                           |
| `team`              | `string`       | The functional team the agent is a part of (e.g., "DataTeam", "FinanceTeam").                      |
| `model_path`        | `string`       | The path to the model. For directory-based safetensor models this points at the folder; for quantized GGUF models point directly to the `.gguf` file.                               |
| `agent_id`          | `string`       | A unique identifier for the agent (e.g., "0x5678-SQLAgent").                                     |
| `attention_weights` | `object`       | A map of task-specific weights, used for fine-tuning the agent's focus.                          |
| `max_tokens`        | `integer`      | The default maximum number of tokens the model should generate for a response.                     |
| `temperature`       | `float`        | The default sampling temperature for the model (0.0 - 2.0). Controls the randomness of the output. |
| `tags`              | `array[string]`| A list of tags for categorizing the agent.                                                       |
| `keywords`          | `array[string]`| A list of keywords that will trigger the routing of a prompt to this agent. This is case-insensitive. |
| `backend_type`      | `string`       | Optional hint for the runtime (e.g., `gguf`, `hf-transformers`, `vaultgemma`, `deepseek-ocr`). |
| `enabled_env_var`   | `string`       | Optional environment variable that must be present (and truthy) for the domain to load.            |

If `enabled_env_var` is set, the domain is only registered when the named environment variable is present and not one of `0`, `false`, `no`, or `off` (case-insensitive). This allows optional domains—such as experimental model servers—to be toggled without editing the JSON file.

### Example

```json
{
  "domains": {
    "0x5678-SQLAgent": {
      "name": "SQL Agent",
      "layer": "layer1",
      "team": "DataTeam",
      "model_path": "models/sql-coder-7b",
      "agent_id": "0x5678-SQLAgent",
      "attention_weights": {},
      "max_tokens": 2048,
      "temperature": 0.1,
      "tags": ["sql", "database", "query"],
      "keywords": ["sql", "select", "from", "where", "insert", "update", "delete", "database", "query"]
    },
    "0x1234-BlockchainAgent": {
      "name": "Blockchain Agent",
      "layer": "layer1",
      "team": "FoundationTeam",
      "model_path": "models/blockchain-llama-3b",
      "agent_id": "0x1234-BlockchainAgent",
      "attention_weights": {},
      "max_tokens": 1024,
      "temperature": 0.5,
      "tags": ["blockchain", "transaction", "smart-contract"],
      "keywords": ["blockchain", "transaction", "eth", "wallet", "smart contract", "solidity"]
    }
  }
}
```


### Quantized GGUF models

All Layer‑1, Layer‑2, and Layer‑3 agents in `domains.json` now point at the quantized Gemma GGUF weights under
`agenticAiETH_layer4_Models`. When adding a new domain that should run on a quantized model, set `model_path`
to the `.gguf` file and set `backend_type` to `gguf`. The runtime will automatically load the file via go-llama.cpp
and apply the tighter token limits expected by the quantized models.
