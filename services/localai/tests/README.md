# Model Testing Framework

Comprehensive testing framework to verify all models produce expected outputs.

## Overview

This framework tests all configured models to ensure:
1. ✅ Models are accessible and respond to requests
2. ✅ Models produce real, expected outputs
3. ✅ Outputs meet quality thresholds
4. ✅ Models handle different prompt types correctly

## Quick Start

```bash
# Run all model tests
./run_model_tests.sh

# Or run directly with Python
python3 test_all_models.py
```

## Environment Variables

- `TRANSFORMERS_SERVICE_URL`: URL of transformers service (default: `http://localhost:9090`)
- `LOCALAI_SERVICE_URL`: URL of LocalAI service (default: `http://localhost:8081`)
- `TEST_TIMEOUT`: Timeout for each test in seconds (default: `60`)
- `RESULTS_DIR`: Directory to save test results (default: `/tmp/model_test_results`)

## Test Types

Each model is tested with:

1. **Health Check**: Verifies model is accessible via health endpoint
2. **Simple Prompt**: Basic functionality test
3. **Reasoning Prompt**: Tests reasoning capabilities
4. **Code Prompt**: Tests code generation (if applicable)
5. **Domain-Specific Prompt**: Tests domain-specific knowledge

## Output

The framework generates:

1. **JSON Results**: Detailed test results in JSON format
2. **Text Report**: Human-readable summary report
3. **Console Output**: Real-time test progress and results

Results are saved to `RESULTS_DIR` with timestamps.

## Quality Metrics

Each response is scored on:
- **Length**: Appropriate response length (10-2000 chars)
- **Coherence**: No obvious errors or error messages
- **Relevance**: Contains relevant words from the prompt
- **Completeness**: Non-empty, meaningful response

Quality scores range from 0.0 to 1.0, with 0.3+ considered passing.

## Example Output

```
============================================================
Testing model: phi-3.5-mini
============================================================
  [1/5] Health check...
    ✅ PASS: OK
  [2/5] Simple prompt...
    ✅ PASS: Hello, I am working correctly!
  [3/5] Reasoning prompt...
    ✅ PASS: 4
  [4/5] Code prompt...
    ✅ PASS: def add_numbers(a, b): return a + b
  [5/5] Domain-specific prompt...
    ✅ PASS: AI is artificial intelligence...
```

## Integration

Add to CI/CD pipeline:

```yaml
- name: Test All Models
  run: |
    cd services/localai/tests
    ./run_model_tests.sh
```

## Continuous Monitoring

Run periodically to ensure models remain functional:

```bash
# Add to cron
0 */6 * * * cd /path/to/aModels/services/localai/tests && ./run_model_tests.sh
```

## Troubleshooting

### Service Not Accessible

If tests fail with connection errors:
1. Verify services are running: `docker ps` or `systemctl status`
2. Check service URLs in environment variables
3. Verify network connectivity

### Model Not Found

If a model is not in the health check:
1. Verify model is in `transformers_cpu_server.py` MODEL_REGISTRY
2. Check model files exist in `/models` directory
3. Restart transformers service

### Low Quality Scores

If quality scores are low:
1. Check model is properly loaded
2. Verify model supports the test prompt type
3. Check for model-specific configuration issues
