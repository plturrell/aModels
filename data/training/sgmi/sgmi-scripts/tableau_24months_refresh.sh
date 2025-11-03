#!/bin/sh

JBS_APPHOME_BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "JBS_APPHOME_BIN " $JBS_APPHOME_BIN
error_code=$($JBS_APPHOME_BIN/invoke_spark_tableau_24months_refresh.sh $@)
if [[ $error_code == *"Data load util failed with error code: 1"* ]]; then
echo "Exiting with error code : 1"
exit 1
else
exit 0
fi
