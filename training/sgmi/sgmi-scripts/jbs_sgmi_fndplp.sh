#!/bin/bash
#MM FND:PLP BCRS_PRA ALL WS7
monthly=$1
stg=$2
system=$3
ctry=$4
ws=$5

cur_dir=/CTRLFW/sgmi/prd/appl/jbs

#export HBASE_HOME=/opt/cloudera/parcels/CDH/bin/
#export HBASE_CLASSPATH={{HBASE_CLASSPATH}}:/opt/cloudera/parcels/SPARK3/lib/spark3/hbase_connectors/lib/hbase-spark3.jar

source /CTRLFW/sgmi/prd/hive2spark/jbs_env.properties

echo "Passed Arguments:"

echo "$monthly - $stg - $system - $ctry - $ws"

echo "Yarn log file: spark_${ws}_fndplp_logs.txt"

spark3-submit --master yarn \
						--deploy-mode cluster \
						--driver-memory 4G \
						--executor-memory 10G \
						--num-executors 2 \
						--executor-cores 4 \
						--conf spark.network.timeout=36000 \
						--conf spark.rpc.askTimeout=600s \
						--conf spark.executor.heartbeatInterval=600s \
						--conf spark.sql.autoBroadcastJoinThreshold=-1 \
						--files jbs_env.properties,/CTRLFW/sgmi/prd/appl/bin/jbs.json,log4j_cluster.properties \
						--conf spark.executor.extraClassPath=/etc/hbase/conf/ \
						--conf spark.driver.extraClassPath=/etc/hbase/conf/ \
						--conf spark.kerberos.principal=g.sgmiprdapp.001@ZONE1.SCB.NET \
						--conf spark.driver.log.persistToDfs.enabled=false \
						--conf spark.kerberos.keytab=/home/g.sgmiprdapp.001/g.sgmiprdapp.001.keytab \
						--jars sgmi-prop-handler-1.0.jar,lamma_2.12-2.3.1.jar,json4s-native_2.12-3.6.11.jar,json4s-core_2.12-3.6.11.jar,json4s-ast_2.12-3.6.11.jar,hive-hbase-handler-3.1.3000.7.1.9.1023-3.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/audience-annotations-0.12.0.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/commons-logging-1.2.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/opentelemetry-api-0.12.0.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/opentelemetry-context-0.12.0.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/shaded-clients/hbase-shaded-mapreduce-2.4.17.7.1.9.1042-1.jar \
						--class master.invoker.PLPRun \
						JBS_FRAMEWORK_lean.jar $monthly $stg $system $ctry $ws > spark_${ws}_fndplp_logs.txt 2>&1

applicationId=$(cat spark_${ws}_fndplp_logs.txt | grep -o 'application_[0-9]\+_[0-9]\+' | head -1)

etl_stage=FND_PLP
currentDate=$(date "+%Y-%m-%dT%H-%m-%S.$(( $(date +%s) % 10000))")
logFile=jbs_PLPDataLoad_${etl_stage}_${system}_${ws}_${ctry}.${currentDate}.log

echo "log Path: ${logFile} and application id: ${applicationId}"

yarn logs --applicationId ${applicationId} > $logFile


#--conf spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j_cluster.properties \
#--conf spark.executor.extraJavaOptions=-Dlog4j.configuration=log4j_cluster.properties \