============================================================
STEP 0: SERVICE HEALTH CHECK - MANDATORY FIRST STEP
============================================================

BEFORE running any tests, you MUST run Step 0 to ensure all 
services are running and accessible.

COMMAND:
  ./testing/00_check_services.sh

This script will:
  1. Check Docker containers are running
  2. Verify service accessibility
  3. Set environment variables
  4. Exit with code 0 if all services ready, 1 if not

AFTER Step 0 passes, you can run tests:
  ./testing/run_all_tests_working.sh

OR use the combined script that includes Step 0:
  ./testing/run_all_tests_with_check.sh

============================================================
