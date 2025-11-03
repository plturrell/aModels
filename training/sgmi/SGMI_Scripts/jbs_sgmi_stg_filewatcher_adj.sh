# Purpose: File watcher script checks the respective file exist in the $STAGING path. 
# Script parameters:
# $1: daily_monthly_flag
# $2: STG
# $3: System
# $4: Country
# $5: Workspaceid
# $6: Next Schedule date


source /CTRLFW/sgmi/prd/appl/bin/jbs_env.properties

#Checking the input parameters started

	v_workspace_id=$5
	v_total_time=50400
	v_sleep_time=3600


DT=`date +%d%m%Y%H%M%S`


v_reporting_date1=$(hive -e "select curr_rpt_month from sgmiprdetl.jbs_current_reporting_period;") 

v_reporting_date=$(echo $v_reporting_date1 | rev | cut -d"|" -f2 | rev )


v_year=$(echo $v_reporting_date |cut -b1-4)
v_month=$(echo $v_reporting_date |cut -b6-7)
v_date=$(echo $v_reporting_date |cut -b9-10)



v_filename=$(echo SGMI_MASTER_FILE_EXTRACT_"$v_workspace_id"_ADJ_"$v_year""$v_month""$v_date".txt)


echo "reporting date: $v_reporting_date " >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log

echo "filename: $v_filename " >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log

echo "Param: $1 $2 $3 $4 $5 $6" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log




#echo "Input Parameters are $1  $2 and $3 " >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
v_no_of_loops=`expr $v_total_time / $v_sleep_time` # How many times loop should iterate
v_no_of_loops=`expr $v_no_of_loops + 1`
echo "Number of loops are $v_no_of_loops " >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log



v_loop_counter=0
#loop starts to check the .go file exist in the $STAGING path or not.
while [ $v_no_of_loops -ne $v_loop_counter ] 
do
	echo "Loop Number $v_loop_counter at "`date` >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log

       if [ "$3" == 'BCRS_PRA' ]
        then
	if [ -r $BCRS_LOCATION/PRA/SGMI_MASTER_FILE_EXTRACT_"$v_workspace_id"_ADJ_"$v_year""$v_month""$v_date".txt ]
        then
           echo "PRA file available for " $v_workspace_id "_ADJ" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
           sh $JBS_APPHOME/bin/jbs_sgmi_gunzip.sh
	
        sleep 30
	sh $JBS_APPHOME/bin/jbs_sgmi_stg_file_copy.sh $1 $2 $3 $4 $5

       if [[ $? == 0 ]]
       then
       echo "PRA files Loaded successfully for " $v_workspace_id "_ADJ" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        exit 0   
        else
        echo "Error with PRA file loading for " $v_workspace_id "_ADJ" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        exit 1
       fi

	fi
        fi

        if [ "$3" == 'BCRS_ND' ]
        then
        if [ -r $BCRS_LOCATION/ND/SGMI_MASTER_FILE_EXTRACT_"$v_workspace_id"_ADJ_"$v_year""$v_month""$v_date".txt ]
        then
        echo "ND file available for " $v_workspace_id "_ADJ" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        sleep 30
		sh $JBS_APPHOME/bin/jbs_sgmi_gunzip.sh
		sleep 30
	sh $JBS_APPHOME/bin/jbs_sgmi_stg_file_copy.sh $1 $2 $3 $4 $5

       if [[ $? == 0 ]]
       then
       echo "ND filed Loaded successfully for " $v_workspace_id "_ADJ">> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        exit 0   
        else
        echo "Error with ND file loading for " $v_workspace_id "_ADJ">> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        exit 1
       fi

	fi
        fi


  if [ "$3" == 'BCRS_RC' ]
        then
        if [ -r $BCRS_LOCATION/RC/SGMI_MASTER_FILE_EXTRACT_"$v_workspace_id"_ADJ_"$v_year""$v_month""$v_date".txt -o -r $BCRS_LOCATION/RC/SGMI_MASTER_FILE_EXTRACT_"$v_workspace_id"_"$v_year""$v_month""$v_date".txt ]
        then
        echo "RC file available for " $v_workspace_id >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        sleep 30
	sh $JBS_APPHOME/bin/SGMI_Adj_File_Rename.sh

       if [[ $? == 0 ]]
       then
       echo "RC filed Loaded successfully for " $v_workspace_id >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        exit 0   
        else
        echo "Error with RC file loading for " $v_workspace_id >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
        exit 1
       fi

	fi
        fi


	v_loop_counter=`expr $v_loop_counter + 1`
	if [ $v_loop_counter -ne $v_no_of_loops ]
	then
	echo "sleeping on"
		sleep $v_sleep_time
	fi
	echo "sleep off"


done


echo "Time up .. We didn't receive any file so exiting the process "`date` >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log

sch_date=`expr substr $6 5 2`


today_mt=`date +"%m"`

#Close job as successfull on last schedule if job ran successfully on first day

if [ $sch_date -ne $today_mt ] 
then
v_stg_status1=$(hive -e "select b.stg_status from sgmiprdetl.jbs_batch b where b.batch_id in (select max(a.batch_id) from sgmiprdetl.jbs_batch a where reporting_dt = '"$v_reporting_date"' and regime='"$v_workspace_id"');")
echo "Staging status for $v_workspace_id reporting $v_reporting_date is $v_stg_status" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log


v_stg_status = $(echo v_stg_status1 | rev | cut -d"|" -f2 | rev)


if [ "$v_stg_status" == 'SUCCESS' ]
then
echo $v_workspace_id "Staging job successfull!!! Closing job as successfull!!!" >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log
exit 2
fi
echo 'Last day of schedule....No files recieved...Ending current job as failed....!!!' >> $JBS_LOG_DIR/File_Watcher_STG_"$3"_"$v_workspace_id"_"$DT".log

exit 1
fi	

echo "Time out ..No file for today"	
exit 2
