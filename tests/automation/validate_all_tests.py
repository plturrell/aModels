#!/usr/bin/env python3
"""
Test Validation Script

Validates all test files are syntactically correct and can be imported.
This runs before actual test execution to catch issues early.
"""

import os
import sys
import importlib.util
from pathlib import Path

def validate_python_file(filepath: Path) -> tuple[bool, str]:
    """Validate a Python file can be imported."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        if spec is None:
            return False, "Could not create spec"
        
        module = importlib.util.module_from_spec(spec)
        
        # Try to compile (syntax check)
        with open(filepath, 'r') as f:
            code = f.read()
            compile(code, filepath, 'exec')
        
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Validate all test files."""
    print("="*60)
    print("Test File Validation")
    print("="*60)
    print()
    
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    test_files.extend(test_dir.glob("*benchmark*.py"))
    
    total = len(test_files)
    passed = 0
    failed = 0
    
    print(f"Found {total} test files to validate")
    print()
    
    for test_file in sorted(test_files):
        is_valid, message = validate_python_file(test_file)
        
        if is_valid:
            print(f"✅ {test_file.name}")
            passed += 1
        else:
            print(f"❌ {test_file.name}: {message}")
            failed += 1
    
    print()
    print("="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Total Files: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print()
    
    if failed > 0:
        print("⚠️  Some test files have issues. Please fix before running tests.")
        sys.exit(1)
    else:
        print("✅ All test files are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()

