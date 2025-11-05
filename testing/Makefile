.PHONY: test test-python test-go test-all clean

# Test targets
test-all: test-python test-go

test-python:
	@echo "Running Python integration tests..."
	python3 test_localai_integration_suite.py

test-go:
	@echo "Running Go integration tests..."
	go test -v -timeout 60s .

test:
	@echo "Running all tests..."
	@./run_all_tests.sh

clean:
	@echo "Cleaning test artifacts..."
	find . -name "*.test" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

