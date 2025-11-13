#! /bin/sh 

########################################################################
# Description: Script to handle SGMI Tableau refresh for 23 MONTHS data source
# Created By: SGMI-DEV TEam
# Created On: 2-Jan-2022
# Last Changed On: 2-Jan-2022
######################################################################## 

TABLEAU_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $TABLEAU_PATH/../appl/bin/jbs_env.properties

main_data_source="$1"
inter_data_source=$main_data_source"_inter_new"


Workspace=(WS7 WS9 WS29 WS36 WS42 WS43)


# Log File Creation
curr_date=$(date +"%Y%m%d")
curr_time=$(date +"%H%M%S")
file_name="sgmi_tbl_24m_refresh_"$main_data_source"_"$curr_date"_"$curr_time
logfile_name=$TABLEAU_LOG_DIR"/"$file_name".log"
mkdir -p $TABLEAU_LOG_DIR
mkdir -p $TABLEAU_TMP


rm $TABLEAU_TMP/$main_data_source.zip
rm -r $TABLEAU_TMP/$main_data_source
rm $TABLEAU_TMP/$inter_data_source.zip
rm -r $TABLEAU_TMP/$inter_data_source
rm $TABLEAU_TMP/$inter_data_source.hyper
rm $TABLEAU_PATH/Python_error_logFile.log

# Getting tableau password
set +x
tableau_pwd="$(${TABLEAU_PROJECT_PATH}/encrypt_tableau.sh D)"

if ! [ $? -eq 0 ]; then
		echo "Can not decrypt the tableau password. Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
		exit 1 
fi
set -x
 
if [ -z $TABLEAU_SITE ]; then  
	TABLEAU_SITE=Default 
fi

export JAVA_HOME=/CTRLFW/sgmi/prd/tableau/java17
export PATH=$JAVA_HOME/bin:$PATH:/CTRLFW/sgmi/prd/tableau/tableau/tabcmd/bin
 
# login 
echo " Login to Tableau Server " $TABLEAU_SERVER $TABLEAU_USERNAME >> $logfile_name
set +x
tabcmd login  --no-certcheck  -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie --no-certcheck 2>&1
set -x 

#echo "Triggering empty refresh for consolidated current month datasources..." >> $logfile_name
cm_data_source_ws=$main_data_source"_currentmonth"

set +x
tabcmd  refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_ws -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
if ! [ $? -eq 0 ]; then
		echo "Can not trigger the refresh for current month " $CW ". Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
		exit 1 
fi
set -x
echo " " >> $logfile_name


echo "Job execution completed. "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name

exit 0
