#!/bin/ksh

# Purpose: File watcher script checks the respective file for RMI exist in the $STAGING path. 
# Script parameters:
# $1: daily_monthly_flag
# $2: STG
# $3: System
# $4: Country
# $5: Next Schedule date


daily_monthly_flag=$1

source /CTRLFW/sgmi/prd/appl/bin/jbs_env.properties

#Checking the input parameters started

	v_total_time=50400
	v_sleep_time=3600


DT=`date +%d%m%Y%H%M%S`


v_reporting_date1=$(hive -e "select curr_rpt_month from sgmiprdetl.jbs_current_reporting_period;") 

v_reporting_date=$(echo $v_reporting_date1 | rev | cut -d"|" -f2 | rev )

v_year=$(echo $v_reporting_date |cut -b1-4)
v_month=$(echo $v_reporting_date |cut -b6-7)
v_date=$(echo $v_reporting_date |cut -b9-10)


echo "rep : $v_reporting_date $v_year $v_month $v_date"



v_filename=$(echo SGMI_MASTER_FILE_EXTRACT_"$1"_"$v_year""$v_month""$v_date".txt)


echo "reporting date: $v_reporting_date " >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log

echo "filename: $v_filename " >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log




#echo "Input Parameters are $1  $2 $3 $4 and $5" >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log
v_no_of_loops=`expr $v_total_time / $v_sleep_time` # How many times loop should iterate
v_no_of_loops=`expr $v_no_of_loops + 1`
echo "Number of loops are $v_no_of_loops " >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log



v_loop_counter=0
#loop starts to check the .go file exist in the $STAGING path or not.
while [ $v_no_of_loops -ne $v_loop_counter ] 
do
	echo "Loop Number $v_loop_counter at "`date` >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log

	if [ -r $RMI_LOCATION/SGMI_RMI_MASTER_FILE_EXTRACT_"$daily_monthly_flag"_"$v_year""$v_month""$v_date".txt ]
        then
        echo "RMI file available for " $1 >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log
        sleep 30
	sh $JBS_APPHOME/bin/jbs_rmi_stg_ingestion.sh $1 $2 $3 $4
       
	   if [[ $? == 0 ]]
       then
       echo "RMI files Loaded successfully for " $daily_monthly_flag >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log
        exit 0   
        else
        echo "Error with RMI file loading for " $daily_monthly_flag >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log
        exit 1
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


echo "Time up .. We didn't receive any file... so exiting the process "`date` >> $JBS_LOG_DIR/File_Watcher_RMI_"$daily_monthly_flag"_"$DT".log

sch_date=`expr substr $5 5 2`

today_mt=`date +"%m"`

#Close job as successfull on last schedule if job ran successfully on first day

if [ $sch_date -ne $today_mt ] 
then
v_stg_status1=$(hive -e "select b.stg_status from sgmiprdetl.jbs_batch b where b.batch_id in (select max(a.batch_id) from sgmiprdetl.jbs_batch a where reporting_dt = '"$v_reporting_date"' and src_sys_nm='RMI' and daily_monthly_flag='"$1"');")
echo "Staging status for RMI $1 reporting date $v_reporting_date is $v_stg_status"

v_stg_status = $(echo v_stg_status1 | rev | cut -d"|" -f2 | rev)

if [ "$v_stg_status" == 'SUCCESS' ]
then
echo "RMI $1 Staging job successfull!!! Closing job as successfull!!!"
exit 2
fi
echo 'Last day of schedule....No files recieved...Ending current job as failed....!!!'
exit 1
fi

echo "Time Out..No RMI files for today"	
exit 2
