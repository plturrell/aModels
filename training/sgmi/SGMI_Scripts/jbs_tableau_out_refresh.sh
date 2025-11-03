#!/bin/sh

# Purpose: File watcher script checks the respective file exist in the $STAGING path. 

source /CTRLFW/sgmi/prd/appl/bin/jbs_env.properties

echo "Java version is"$JAVA_HOME
echo $JAVA_HOME
echo $(which tabcmd)



sh $JBS_APPHOME/bin/jbs_sgmi_gzip.sh

#Checking the input parameters started

#288 times

	v_total_time=28800
	v_sleep_time=1200

v_workspace_id=$5


DT=`date +%d%m%Y%H%M%S`

v_reporting_date1=$(hive -e "select curr_rpt_month from sgmiprdetl.jbs_current_reporting_period;")

v_reporting_date=$(echo $v_reporting_date1 | rev | cut -d"|" -f2 | rev )


v_year=$(echo $v_reporting_date |cut -b1-4)
v_month=$(echo $v_reporting_date |cut -b6-7)
v_date=$(echo $v_reporting_date |cut -b9-10)

echo "reporting date: $v_reporting_date "

#echo "Input Parameters are $1  $2 $3 $4 and $5"
v_no_of_loops=`expr $v_total_time / $v_sleep_time` # How many times loop should iterate
v_no_of_loops=`expr $v_no_of_loops + 1`
echo "Number of loops are $v_no_of_loops "



if [ "$v_workspace_id" = "WS9" ]
then
    next_ws=WS7
elif [ "$v_workspace_id" = "WS7" ]
then
    next_ws=WS9
elif [ "$v_workspace_id" = "WS7_ADJ" ]
then
    next_ws=WS9_ADJ
elif [ "$v_workspace_id" = "WS9_ADJ" ]
then
    next_ws=WS7_ADJ
elif [ "$v_workspace_id" = "WS29_ADJ" ]
then
    next_ws=WS36_ADJ
elif [ "$v_workspace_id" = "WS36_ADJ" ]
then
    next_ws=WS29_ADJ
elif [ "$v_workspace_id" = "WS36" ]
then
    next_ws=WS29
elif [ "$v_workspace_id" = "WS29" ]
then
    next_ws=WS36
elif [ "$v_workspace_id" = "WS42" ]
then
    next_ws=WS43
elif [ "$v_workspace_id" = "WS43" ]
then
    next_ws=WS42
else
     next_ws=null
fi

echo $next_ws


v_out_status=$(hive -e "select b.out_status from sgmiprdetl.jbs_batch b where b.batch_id in (select max(a.batch_id) from sgmiprdetl.jbs_batch a where a.regime ='"$next_ws"'  and a.reporting_dt ='"$v_reporting_date"');")

echo "$next_ws output job is " $v_out_status


if [ "$v_out_status" == 'RUNNING' ];
then		  	
                v_loop_counter=0
		#loop starts to check the .go file exist in the $STAGING path or not.
while [ $v_no_of_loops != $v_loop_counter ] 
	do
			echo "Loop Number $v_loop_counter at "`date`
		
	   v_loop_counter=`expr $v_loop_counter + 1`
	if [ $v_loop_counter -ne $v_no_of_loops ]
	   then
	   echo "sleeping on"
           echo "Waiting for $next_ws output to complete......"
	   sleep $v_sleep_time
	fi
	echo "sleep off"
v_out_status=$(hive -e "select b.out_status from sgmiprdetl.jbs_batch b where b.batch_id in (select max(a.batch_id) from sgmiprdetl.jbs_batch a where a.regime ='"$next_ws"'  and a.reporting_dt ='"$v_reporting_date"');")
echo "$next_ws output job is " $v_out_status
if [ "$v_out_status" == 'SUCCESS' ] || [ "$v_out_status" == 'FAILED' ]; then
echo "starting out for $v_workspace_id............"
#calling Tableau refresh script

sh /CTRLFW/sgmi/prd/appl/bin/jbs_tableu_refresh.sh $1 $2 $3 $4 $5


 if [[ $? == 0 ]]
       then
       echo "Output completed successfully for"  $v_workspace_id
        exit 0   
        else
        echo "Error with Output referesh for " $v_workspace_id
        exit 1
       fi
fi

done        
else echo "$next_ws out job not running...starting the out job for $v_workspace_id............"
sh /CTRLFW/sgmi/prd/appl/bin/jbs_tableu_refresh.sh $1 $2 $3 $4 $5
 if [[ $? == 0 ]]
       then
       echo "Output completed successfully for"  $v_workspace_id
        exit 0   
        else
        echo "Error with Output referesh for " $v_workspace_id
        exit 1
       fi
fi


