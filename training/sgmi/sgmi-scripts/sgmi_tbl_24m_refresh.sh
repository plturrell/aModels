#! /bin/sh 

####################################################################################################################
# Description: Script to handle SGMI Tableau refresh for 13 MONTHS data source.Recreated for run time optimization
# Created By: SGMI Team
# Created On: 2-Jan-2023
# Last Changed On: 3-Sep-2025
#################################################################################################################### 

#COMMONAPPAREA=$(cd "$(dirname "$0")";  pwd)
#cd $COMMONAPPAREA
#. ./tableau.env 

TABLEAU_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $TABLEAU_PATH/../appl/bin/jbs_env.properties

# Set up variabels
main_data_source="$1"
job_group="$2"
workspace="$3"

input_date1=$(hive -e "select curr_rpt_month from sgmiprdetl.jbs_current_reporting_period;")
input_date=$(echo $input_date1 | rev | cut -d"|" -f2 | rev | xargs )

echo "Input_date" $input_date

current_month=$(date -d "$input_date" +"%Y-%m")


echo $input_date


cm_data_source="SingaporeMI_currentmonth"
cm_data_source_WS7=$main_data_source"_currentmonth_WS7"
cm_data_source_WS9=$main_data_source"_currentmonth_WS9"
cm_data_source_WS29=$main_data_source"_currentmonth_WS29"
cm_data_source_WS36=$main_data_source"_currentmonth_WS36"
cm_data_source_WS42=$main_data_source"_currentmonth_WS42"
cm_data_source_WS43=$main_data_source"_currentmonth_WS43"
twenty3_data_source=$main_data_source"_23months"
inter_data_source=$main_data_source"_inter"
 
# Log File Creation
curr_date=$(date +"%Y%m%d")
curr_time=$(date +"%H%M%S")
file_name="sgmi_tbl_24m_refresh_"$main_data_source"_"$curr_date"_"$curr_time
logfile_name=$TABLEAU_LOG_DIR"/"$file_name".log"
file_status="sgmi_tbl_currmonth_refresh_"$input_date
refresh_status=$TABLEAU_LOG_DIR"/"$file_status".conf"
mkdir -p $TABLEAU_LOG_DIR
mkdir -p $TABLEAU_TMP

echo "see file" $file_status

if [[ $main_data_source == "SingaporeMI" || $main_data_source == "SingaporeMI_Limit" || $main_data_source == "SingaporeMI_Collateral" ]];then

export JAVA_HOME=/CTRLFW/sgmi/prd/tableau/java17
export PATH=$JAVA_HOME/bin:$PATH:/CTRLFW/sgmi/prd/tableau/tableau/tabcmd/bin
echo $JAVA_HOME
echo $(which tabcmd)

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
 
# login 
echo " Login to Tableau Server " $TABLEAU_SERVER $TABLEAU_USERNAME >> $logfile_name
echo "Workspace" $workspace >> $logfile_name

echo " " >> $logfile_name

set +x
tabcmd login -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie --no-certcheck 2>&1
set -x 

if [ $job_group = "OUT_LOAD" ]; then 


   if [ "$workspace" = "WS7" ]; then
      # trigger refresh current month WS7 data source
      echo "Trigger refresh for "$cm_data_source_WS7" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
       
      set +x
      tabcmd    refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_WS7 -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
      if ! [ $? -eq 0 ]; then
      		echo "Can not trigger the refresh for "$cm_data_source_WS7" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
      		exit 1 
      fi
      set -x
   elif [ "$workspace" = "WS9" ]
      then
      # trigger refresh current month WS9 data source
      echo "Trigger refresh for "$cm_data_source_WS9" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
       
      set +x
      tabcmd    refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_WS9 -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
      if ! [ $? -eq 0 ]; then
      		echo "Can not trigger the refresh for "$cm_data_source_WS9". Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
      		exit 1 
      fi
      set -x
   elif [ "$workspace" = "WS29" ]
      then
      # trigger refresh current month WS29 data source
      echo "Trigger refresh for "$cm_data_source_WS29" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
       
      set +x
      tabcmd    refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_WS29 -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
      if ! [ $? -eq 0 ]; then
      		echo "Can not trigger the refresh for "$cm_data_source_WS29". Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
      		exit 1 
      fi
      set -x
   elif [ "$workspace" = "WS36" ]
      then
      # trigger refresh current month WS36 data source
      echo "Trigger refresh for "$cm_data_source_WS36" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
       
      set +x
      tabcmd    refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_WS36 -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
      if ! [ $? -eq 0 ]; then
      		echo "Can not trigger the refresh for "$cm_data_source_WS36". Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
      		exit 1 
      fi
      set -x
   elif [ "$workspace" = "WS42" ]
      then
      # trigger refresh current month WS42 data source
      echo "Trigger refresh for "$cm_data_source_WS42" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
       
      set +x
      tabcmd    refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_WS42 -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
      if ! [ $? -eq 0 ]; then
      		echo "Can not trigger the refresh for "$cm_data_source_WS42". Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
      		exit 1 
      fi
      set -x
   elif [ "$workspace" = "WS43" ]
      then
      # trigger refresh current month WS43 data source
      echo "Trigger refresh for "$cm_data_source_WS43" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
       
      set +x
      tabcmd    refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource $cm_data_source_WS43 -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
      if ! [ $? -eq 0 ]; then
      		echo "Can not trigger the refresh for "$cm_data_source_WS43". Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
      		exit 1 
      fi
      set -x
   else
   echo "Workspace not provided!!!" >> $logfile_name
   		exit 1 
   fi
   
   
   
   echo " " >> $logfile_name
   
 # download all current month WS 
   if [ "$workspace" = "WS7" ]; then
		# download current month  WS7 data source
		echo "Download datasource for "$cm_data_source_WS7" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			
		set +x
		tabcmd   get "/datasources/$cm_data_source_WS7.tdsx" --no-certcheck -f $TABLEAU_TMP/$cm_data_source_WS7.zip -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE  --no-cookie >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not download "$cm_data_source_WS7 " . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
		set -x
		
		echo " " >> $logfile_name
   elif [ "$workspace" = "WS9" ]
      then
		# download current month  WS9 data source
		echo "Download datasource for "$cm_data_source_WS9" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			
		set +x
		tabcmd  get "/datasources/$cm_data_source_WS9.tdsx" --no-certcheck -f $TABLEAU_TMP/$cm_data_source_WS9.zip -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE  --no-cookie >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not download "$cm_data_source_WS9 " . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
		set -x
		
		echo " " >> $logfile_name
   elif [ "$workspace" = "WS29" ]
      then	
		# download current month  WS29 data source
		echo "Download datasource for "$cm_data_source_WS29" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			
		set +x
		tabcmd   get "/datasources/$cm_data_source_WS29.tdsx" --no-certcheck -f $TABLEAU_TMP/$cm_data_source_WS29.zip -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE  --no-cookie >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not download "$cm_data_source_WS29 " . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
		set -x
		
		echo " " >> $logfile_name
   elif [ "$workspace" = "WS36" ]
      then	
		# download current month  WS36 data source
		echo "Download datasource for "$cm_data_source_WS36" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			
		set +x
		tabcmd   get "/datasources/$cm_data_source_WS36.tdsx" --no-certcheck -f $TABLEAU_TMP/$cm_data_source_WS36.zip -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE  --no-cookie >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not download "$cm_data_source_WS36 " . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
		set -x
		
		echo " " >> $logfile_name
		
   elif [ "$workspace" = "WS42" ]
      then	
		
		#download current month for WS42 data source
		echo "Download datasource for "$cm_data_source_WS42" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			
		set +x
	    tabcmd get "/datasources/$cm_data_source_WS42.tdsx" --no-certcheck -f $TABLEAU_TMP/$cm_data_source_WS42.zip -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE  --no-cookie >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not download "$cm_data_source_WS42" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
		set -x
		
		echo " " >> $logfile_name
		
   elif [ "$workspace" = "WS43" ]
      then	
		#download current month for WS43 data source
		echo "Download datasource for "$cm_data_source_WS43" "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			
		set +x
		tabcmd get "/datasources/$cm_data_source_WS43.tdsx" --no-certcheck -f $TABLEAU_TMP/$cm_data_source_WS43.zip -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE  --no-cookie >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not download "$cm_data_source_WS43" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
		set -x
		
		echo " " >> $logfile_name
   else
        echo "Workspace not provided!!!" >> $logfile_name
   		exit 1 
   fi  

   # unzip current month data source
   echo "Unzip current months workspace  sources and append to current month  "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
   if [ "$workspace" = "WS7" ]; then
		unzip -o -q  $TABLEAU_TMP/$cm_data_source_WS7.zip  -d $TABLEAU_TMP/$cm_data_source_WS7  >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
			echo "Can not unzip "$cm_data_source_WS7" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
			exit 1 
		fi 
   
   		echo "$cm_data_source_WS7=complete"  >> $refresh_status 2>&1
      		# remove space in hyper name if exists
		find $TABLEAU_TMP/$cm_data_source_WS7 -type f -name "* *.hyper" -exec bash -c 'mv -f "$0" "${0// /_}"' {} \; >> $logfile_name
		cm_hyper_name_WS7=$(basename $TABLEAU_TMP/$cm_data_source_WS7"/Data/Extracts/"*) 
		cm_hyper_path_WS7=$TABLEAU_TMP/$cm_data_source_WS7"/Data/Extracts/"$cm_hyper_name_WS7
		echo $cm_hyper_path_WS7 >> $logfile_name
		
   
		echo " " >> $logfile_name
   elif [ "$workspace" = "WS9" ]
      then
		unzip -o -q  $TABLEAU_TMP/$cm_data_source_WS9.zip  -d $TABLEAU_TMP/$cm_data_source_WS9  >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not unzip "$cm_data_source_WS9" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
                echo "$cm_data_source_WS9=complete"  >> $refresh_status 2>&1
      
		# remove space in hyper name if exists
		find $TABLEAU_TMP/$cm_data_source_WS9 -type f -name "* *.hyper" -exec bash -c 'mv -f "$0" "${0// /_}"' {} \; >> $logfile_name
		cm_hyper_name_WS9=$(basename $TABLEAU_TMP/$cm_data_source_WS9"/Data/Extracts/"*) 
		cm_hyper_path_WS9=$TABLEAU_TMP/$cm_data_source_WS9"/Data/Extracts/"$cm_hyper_name_WS9
		echo $cm_hyper_path_WS9 >> $logfile_name
				
		echo " " >> $logfile_name
   
   elif [ "$workspace" = "WS29" ]
      then
		unzip -o -q  $TABLEAU_TMP/$cm_data_source_WS29.zip  -d $TABLEAU_TMP/$cm_data_source_WS29  >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not unzip "$cm_data_source_WS29" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
                echo "$cm_data_source_WS29=complete"  >> $refresh_status 2>&1		
		# remove space in hyper name if exists
		find $TABLEAU_TMP/$cm_data_source_WS29 -type f -name "* *.hyper" -exec bash -c 'mv -f "$0" "${0// /_}"' {} \; >> $logfile_name
		cm_hyper_name_WS29=$(basename $TABLEAU_TMP/$cm_data_source_WS29"/Data/Extracts/"*) 
		cm_hyper_path_WS29=$TABLEAU_TMP/$cm_data_source_WS29"/Data/Extracts/"$cm_hyper_name_WS29
		echo $cm_hyper_path_WS29 >> $logfile_name
		
   		echo " " >> $logfile_name
   
   elif [ "$workspace" = "WS36" ]
      then
		unzip -o -q  $TABLEAU_TMP/$cm_data_source_WS36.zip  -d $TABLEAU_TMP/$cm_data_source_WS36  >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not unzip "$cm_data_source_WS36" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
                echo "$cm_data_source_WS36=complete"  >> $refresh_status 2>&1		
		# remove space in hyper name if exists
		find $TABLEAU_TMP/$cm_data_source_WS36 -type f -name "* *.hyper" -exec bash -c 'mv -f "$0" "${0// /_}"' {} \; >> $logfile_name
		cm_hyper_name_WS36=$(basename $TABLEAU_TMP/$cm_data_source_WS36"/Data/Extracts/"*) 
		cm_hyper_path_WS36=$TABLEAU_TMP/$cm_data_source_WS36"/Data/Extracts/"$cm_hyper_name_WS36
		echo $cm_hyper_path_WS36 >> $logfile_name
  
		echo " " >> $logfile_name
   elif [ "$workspace" = "WS42" ]
      then
		unzip -o -q  $TABLEAU_TMP/$cm_data_source_WS42.zip  -d $TABLEAU_TMP/$cm_data_source_WS42 >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not unzip "$cm_data_source_WS42" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
                echo "$cm_data_source_WS42=complete"  >> $refresh_status 2>&1		
		# remove space in hyper name if exists
		find $TABLEAU_TMP/$cm_data_source_WS42 -type f -name "* *.hyper" -exec bash -c 'mv -f "$0" "${0// /_}"' {} \; >> $logfile_name
		cm_hyper_name_WS42=$(basename $TABLEAU_TMP/$cm_data_source_WS42"/Data/Extracts/"*) 
		cm_hyper_path_WS42=$TABLEAU_TMP/$cm_data_source_WS42"/Data/Extracts/"$cm_hyper_name_WS42
		echo $cm_hyper_path_WS42 >> $logfile_name
   
		echo " " >> $logfile_name
   
   elif [ "$workspace" = "WS43" ]
      then
		unzip -o -q  $TABLEAU_TMP/$cm_data_source_WS43.zip  -d $TABLEAU_TMP/$cm_data_source_WS43  >> $logfile_name 2>&1
		if ! [ $? -eq 0 ]; then
				echo "Can not unzip "$cm_data_source_WS43" . Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
				exit 1 
		fi
        echo "$cm_data_source_WS43=complete"  >> $refresh_status 2>&1	
		# remove space in hyper name if exists
		find $TABLEAU_TMP/$cm_data_source_WS43 -type f -name "* *.hyper" -exec bash -c 'mv -f "$0" "${0// /_}"' {} \; >> $logfile_name
		cm_hyper_name_WS43=$(basename $TABLEAU_TMP/$cm_data_source_WS43"/Data/Extracts/"*) 
		cm_hyper_path_WS43=$TABLEAU_TMP/$cm_data_source_WS43"/Data/Extracts/"$cm_hyper_name_WS43
		echo $cm_hyper_path_WS43 >> $logfile_name

   
		echo " " >> $logfile_name
   else
        echo "Workspace not provided!!!" >> $logfile_name
   		exit 1 
   fi 
   
   
   # publish 13 months to SingaporeMI
   
   Workspace=(WS7 WS9 WS29 WS36 WS42 WS43)
   ALL_COMPLETE="Yes"
  
   
   for i in {0..5}
   do
		job=${Workspace[i]}
                status=$job"=complete"
                if grep -q $status $refresh_status; then
                     echo "$job workspace completed" >> $logfile_name
		else
                        echo $job " is pending" >> $logfile_name
			ALL_COMPLETE="No"
			
		fi
   done

   if [ $ALL_COMPLETE == "Yes" ];then
	echo "All workspace refresh completed. Start with SingaporeMI Refresh" >> $logfile_name        
        if grep -q 'SGMI Full Refresh Started' $refresh_status; then
            WS=grep 'SGMI Full Refresh Started:' $refresh_status | cut -d':' -f2-
            echo "Full refresh already started as part of "$WS".Exiting the script" >> $logfile_name
         else
            sh $TABLEAU_PROJECT_PATH/sgmi_full_refresh.sh $main_data_source $workspace
        fi
   else	
    echo "Pending Workspaces refresh." >> $logfile_name
    tabcmd logout >> $logfile_name
    exit 0   
   fi	
   
else # job_group apart from OUT_LOAD

echo "****************************** START FULL REFRESH PROCESS FOR " $main_data_source " AT " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
 
# login 
echo "Login to Tableau Server " $TABLEAU_SERVER $TABLEAU_USERNAME >> $logfile_name
set +x
tabcmd login -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie --no-certcheck 2>&1
set -x 

# trigger refresh current month data source
echo "Trigger refresh for " "$main_data_source" " "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name

if [[ $main_data_source == "SingaporeMI_Limit" || $main_data_source == "SingaporeMI_Collateral" ]];then
   set +x
   tabcmd  refreshextracts --no-certcheck --project $TABLEAU_PROJECT --synchronous  --datasource "$main_data_source" -s $TABLEAU_SERVER -u $TABLEAU_USERNAME -p $tableau_pwd -t $TABLEAU_SITE --no-cookie >> $logfile_name 2>&1
   if ! [ $? -eq 0 ]; then
   		echo "Can not trigger the refresh. Exit at " $(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
   		exit 1 
   else
   echo $main_data_source " has been refreshed successfully. "$(date +"%Y%m%d") $(date +"%H%M%S") >> $logfile_name
   fi
   set -x

fi

# logout

tabcmd logout >> $logfile_name
exit 0 
fi
fi
