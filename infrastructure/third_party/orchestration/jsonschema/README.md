# JSON Schema Package

This package provides a simple, Go-native way to define a JSON Schema.

## Purpose

Modern Large Language Models (LLMs), particularly those with "function calling" capabilities, can be instructed to generate output that conforms to a specific JSON schema. This package provides a set of Go structs that can be used to define such a schema programmatically.

By defining the schema in Go, you can ensure type safety and make it easier to construct the complex JSON structures that the LLM APIs expect.

## Core Components

The main component is the `Definition` struct, which represents a JSON Schema. It allows you to specify:

-   The `Type` of a field (e.g., `Object`, `String`, `Number`).
-   A `Description` of the field, which is very useful for the LLM.
-   The `Properties` of an object, which is a map of field names to their own `Definition`.
-   A list of `Required` fields for an object.
-   The `Items` of an array, which is a pointer to another `Definition`.

## How It's Used

This package is typically used to define the structure of a tool or function that you want an LLM to be able to call. The resulting schema can then be passed to the LLM as part of the API request.

### Example

Let's say you have a function `getWeather(city string, unit string)`. You could define its schema as follows:

```go
import "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/jsonschema"

// Define the schema for the getWeather function
schema := jsonschema.Definition{
    Type: jsonschema.Object,
    Properties: map[string]jsonschema.Definition{
        "city": {
            Type:        jsonschema.String,
            Description: "The city to get the weather for",
        },
        "unit": {
            Type:        jsonschema.String,
            Description: "The unit of temperature, either celsius or fahrenheit",
            Enum:        []string{"celsius", "fahrenheit"},
        },
    },
    Required: []string{"city"},
}

// Marshal the schema to JSON
jsonBytes, err := json.Marshal(schema)

// This JSON can now be sent to the LLM as part of a function calling request.
```
