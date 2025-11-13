#!/bin/bash

source /CTRLFW/sgmi/prd/appl/bin/jbs_env.properties

spark3-submit --master yarn \
			--deploy-mode client \
			--driver-memory 3G\
			--files /CTRLFW/sgmi/prd/appl/bin/jbs_env.properties,/CTRLFW/sgmi/prd/appl/bin/jbs.json \
			--conf spark.executor.extraClassPath=/etc/hbase/conf/ \
            		--conf spark.driver.extraClassPath=/etc/hbase/conf/ \
			--conf spark.kerberos.principal=g.sgmiprdapp.001@ZONE1.SCB.NET \
   			--conf spark.kerberos.keytab=/home/g.sgmiprdapp.001/g.sgmiprdapp.001.keytab \
			--conf spark.driver.log.persistToDfs.enabled=false \
			--jars /CTRLFW/sgmi/prd/hive2spark/spark3jars/sgmi-prop-handler-1.0.jar,/CTRLFW/sgmi/prd/hive2spark/spark3jars/lamma_2.12-2.3.1.jar,/CTRLFW/sgmi/prd/hive2spark/spark3jars/json4s-native_2.12-3.6.11.jar,/CTRLFW/sgmi/prd/hive2spark/spark3jars/json4s-core_2.12-3.6.11.jar,/CTRLFW/sgmi/prd/hive2spark/spark3jars/json4s-ast_2.12-3.6.11.jar,/CTRLFW/sgmi/prd/hive2spark/hive-hbase-handler-3.1.3000.7.1.9.1023-3.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/audience-annotations-0.12.0.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/commons-logging-1.2.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/opentelemetry-api-0.12.0.jar,/opt/cloudera/parcels/CDH/lib/hbase/lib/client-facing-thirdparty/opentelemetry-context-0.12.0.jar,/opt/cloudera/parcels/CDH-7.1.9-1.cdh7.1.9.p1045.67903105/lib/hbase/lib/shaded-clients/hbase-shaded-mapreduce-2.4.17.7.1.9.1045-5.jar \
			--class master.invoker.refresh_current_reporting_period \
			/CTRLFW/sgmi/prd/appl/bin/JBS_FRAMEWORK_lean.jar