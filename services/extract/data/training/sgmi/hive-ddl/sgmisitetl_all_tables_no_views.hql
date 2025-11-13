hive.strict.checks.no.partition.filter=false
create database if not exists sgmisitetl LOCATION '/sit/sgmi/hdata/sgmisitetl';

use sgmisitetl;

CREATE EXTERNAL TABLE `jbs_additional_reporting_period`(
  `reporting_period` string, 
  `updated_ts` timestamp)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_additional_reporting_period'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619391', 
  'numFiles'='0', 
  'numRows'='0', 
  'rawDataSize'='0', 
  'totalSize'='0', 
  'transient_lastDdlTime'='1689619391');

CREATE EXTERNAL TABLE `jbs_batch`(
  `batch_id` string COMMENT '', 
  `ctry_cd` string COMMENT '', 
  `src_sys_nm` string COMMENT '', 
  `reporting_dt` string COMMENT '', 
  `prev_reporting_dt` string COMMENT '', 
  `daily_monthly_flag` string COMMENT '', 
  `regime` string COMMENT '', 
  `status` string COMMENT '', 
  `pstg_status` string COMMENT '', 
  `stg_status` string COMMENT '', 
  `fnd_status` string COMMENT '', 
  `plp_status` string COMMENT '', 
  `out_status` string COMMENT '', 
  `outbound_status` string COMMENT '', 
  `start_ts` timestamp COMMENT '', 
  `end_ts` timestamp COMMENT '', 
  `pstg_start_ts` timestamp COMMENT '', 
  `pstg_end_ts` timestamp COMMENT '', 
  `stg_start_ts` timestamp COMMENT '', 
  `stg_end_ts` timestamp COMMENT '', 
  `fnd_start_ts` timestamp COMMENT '', 
  `fnd_end_ts` timestamp COMMENT '', 
  `plp_start_ts` timestamp COMMENT '', 
  `plp_end_ts` timestamp COMMENT '', 
  `out_start_ts` timestamp COMMENT '', 
  `out_end_ts` timestamp COMMENT '', 
  `outbound_start_ts` timestamp COMMENT '', 
  `outbound_end_ts` timestamp COMMENT '', 
  `message` string COMMENT '')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.hbase.HBaseSerDe' 
STORED BY 
  'org.apache.hadoop.hive.hbase.HBaseStorageHandler' 
WITH SERDEPROPERTIES ( 
  'hbase.columns.mapping'=':key,batch_details:ctry_cd,batch_details:src_sys_nm,batch_details:reporting_dt,batch_details:prev_reporting_dt,batch_details:daily_monthly_flag,batch_details:regime,batch_details:status,batch_details:pstg_status,batch_details:stg_status,batch_details:fnd_status,batch_details:plp_status,batch_details:out_status,batch_details:outbound_status,batch_details:start_ts,batch_details:end_ts,batch_details:pstg_start_ts,batch_details:pstg_end_ts,batch_details:stg_start_ts,batch_details:stg_end_ts,batch_details:fnd_start_ts,batch_details:fnd_end_ts,batch_details:plp_start_ts,batch_details:plp_end_ts,batch_details:out_start_ts,batch_details:out_end_ts,batch_details:outbound_start_ts,batch_details:outbound_end_ts,batch_details:message', 
  'serialization.format'='1')
TBLPROPERTIES (
  'hbase.mapred.output.outputtable'='sgmisitetl:jbs_batch', 
  'hbase.table.name'='sgmisitetl:jbs_batch', 
  'last_modified_by'='g.sgmidev.001', 
  'last_modified_time'='1737740057', 
  'numFiles'='0', 
  'numRows'='0', 
  'rawDataSize'='0', 
  'totalSize'='0', 
  'transient_lastDdlTime'='1737740057');

CREATE EXTERNAL TABLE `jbs_current_reporting_period`(
  `key` int COMMENT '', 
  `curr_rpt_month` string COMMENT '', 
  `prev_rpt_month` string COMMENT '', 
  `next_rpt_month` string COMMENT '', 
  `updated_ts` timestamp COMMENT '')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.hbase.HBaseSerDe' 
STORED BY 
  'org.apache.hadoop.hive.hbase.HBaseStorageHandler' 
WITH SERDEPROPERTIES ( 
  'hbase.columns.mapping'=':key,details:curr_rpt_month,details:prev_rpt_month,details:next_rpt_month,details:updated_ts', 
  'serialization.format'='1')
TBLPROPERTIES (
  'hbase.mapred.output.outputtable'='sgmisitetl:jbs_current_reporting_period', 
  'hbase.table.name'='sgmisitetl:jbs_current_reporting_period', 
  'last_modified_by'='sgmisitapp', 
  'last_modified_time'='1632128944', 
  'numFiles'='0', 
  'numRows'='0', 
  'rawDataSize'='0', 
  'totalSize'='0', 
  'transient_lastDdlTime'='1632128944');

CREATE EXTERNAL TABLE `jbs_data_load_util_config`(
  `job_grp_sequence` bigint, 
  `data_source` string, 
  `source_sql` string, 
  `source_view` string, 
  `source_table` string, 
  `source_filter` string, 
  `target_table` string, 
  `target_filter` string, 
  `insert_mode` string, 
  `partition_cols` string, 
  `id_cols` string, 
  `index_flag` string, 
  `gather_stats_flag` string, 
  `daily_monthly_flag` string, 
  `execution_engine` string, 
  `status` string, 
  `truncate_partition` string, 
  `excluded_context_cols` string, 
  `partial_merge_cols` string)
PARTITIONED BY ( 
  `data_load_id` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_data_load_util_config'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619393', 
  'transient_lastDdlTime'='1689619393');

CREATE EXTERNAL TABLE `jbs_data_load_util_config_bak`(
  `job_grp_sequence` bigint, 
  `data_source` string, 
  `source_sql` string, 
  `source_view` string, 
  `source_table` string, 
  `source_filter` string, 
  `target_table` string, 
  `target_filter` string, 
  `insert_mode` string, 
  `partition_cols` string, 
  `id_cols` string, 
  `index_flag` string, 
  `gather_stats_flag` string, 
  `daily_monthly_flag` string, 
  `execution_engine` string, 
  `status` string, 
  `truncate_partition` string, 
  `excluded_context_cols` string, 
  `partial_merge_cols` string, 
  `data_load_id` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_data_load_util_config_bak'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619413', 
  'numFiles'='1', 
  'numRows'='230', 
  'rawDataSize'='1023572', 
  'totalSize'='1023802', 
  'transient_lastDdlTime'='1689619413');

CREATE EXTERNAL TABLE `jbs_data_trace_util_config`(
  `job_grp_sequence` int, 
  `source_table` string, 
  `source_sql` string, 
  `source_filter` string, 
  `target_table` string, 
  `target_sql` string, 
  `target_filter` string, 
  `metric_cols` string, 
  `context_cols` string, 
  `label` string, 
  `status` string, 
  `data_source` string)
PARTITIONED BY ( 
  `data_load_id` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_data_trace_util_config'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619397', 
  'transient_lastDdlTime'='1689619397');

CREATE EXTERNAL TABLE `jbs_date_param`(
  `param_id` string COMMENT '', 
  `ctry_cd` string COMMENT '', 
  `src_sys_nm` string COMMENT '', 
  `from_date` string COMMENT '', 
  `to_date` string COMMENT '')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.hbase.HBaseSerDe' 
STORED BY 
  'org.apache.hadoop.hive.hbase.HBaseStorageHandler' 
WITH SERDEPROPERTIES ( 
  'hbase.columns.mapping'=':key,details:ctry_cd,details:src_sys_nm,details:from_date,details:to_date', 
  'serialization.format'='1')
TBLPROPERTIES (
  'COLUMN_STATS_ACCURATE'='{\"BASIC_STATS\":\"true\"}', 
  'hbase.mapred.output.outputtable'='sgmisitetl:jbs_date_param', 
  'hbase.table.name'='sgmisitetl:jbs_date_param', 
  'numFiles'='0', 
  'numRows'='0', 
  'rawDataSize'='0', 
  'totalSize'='0', 
  'transient_lastDdlTime'='1531276956');

CREATE EXTERNAL TABLE `jbs_job_group_config`(
  `data_load_id` string)
PARTITIONED BY ( 
  `job_group` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_job_group_config'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619394', 
  'transient_lastDdlTime'='1689619394');

CREATE EXTERNAL TABLE `jbs_job_group_config_bak`(
  `data_load_id` string, 
  `job_group` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_job_group_config_bak'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619409', 
  'numFiles'='1', 
  'numRows'='585', 
  'rawDataSize'='25631', 
  'totalSize'='26216', 
  'transient_lastDdlTime'='1689619409');

CREATE EXTERNAL TABLE `jbs_job_group_trace_config`(
  `data_load_id` string)
PARTITIONED BY ( 
  `job_group` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_job_group_trace_config'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619399', 
  'transient_lastDdlTime'='1689619399');

CREATE EXTERNAL TABLE `jbs_job_instance`(
  `job_instance_id` string COMMENT '', 
  `job_group` string COMMENT '', 
  `batch_id` bigint COMMENT '', 
  `src_sys_nm` string COMMENT '', 
  `ctry_cd` string COMMENT '', 
  `data_load_config_id` string COMMENT '', 
  `status` string COMMENT '', 
  `start_ts` timestamp COMMENT '', 
  `end_ts` timestamp COMMENT '', 
  `message` string COMMENT '', 
  `custom1` string COMMENT '', 
  `custom2` string COMMENT '', 
  `custom3` string COMMENT '')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.hbase.HBaseSerDe' 
STORED BY 
  'org.apache.hadoop.hive.hbase.HBaseStorageHandler' 
WITH SERDEPROPERTIES ( 
  'hbase.columns.mapping'=':key,batch_details:job_group,batch_details:batch_id,batch_details:src_sys_nm,batch_details:ctry_cd,batch_details:data_load_config_id,batch_details:status,batch_details:start_ts,batch_details:end_ts,batch_details:message,batch_details:custom1,batch_details:custom2,batch_details:custom3', 
  'serialization.format'='1')
TBLPROPERTIES (
  'COLUMN_STATS_ACCURATE'='{\"BASIC_STATS\":\"true\"}', 
  'hbase.mapred.output.outputtable'=' sgmisitetl:jbs_job_instance', 
  'hbase.table.name'='sgmisitetl:jbs_job_instance', 
  'numFiles'='0', 
  'numRows'='0', 
  'rawDataSize'='0', 
  'totalSize'='0', 
  'transient_lastDdlTime'='1542983679');

CREATE EXTERNAL TABLE `jbs_job_instance_23nov2018`(
  `job_instance_id` string, 
  `job_group` string, 
  `batch_id` bigint, 
  `src_sys_nm` string, 
  `ctry_cd` string, 
  `data_load_config_id` string, 
  `status` string, 
  `start_ts` timestamp, 
  `end_ts` timestamp, 
  `message` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/jbs_job_instance_23nov2018'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619411', 
  'numFiles'='1', 
  'numRows'='1136', 
  'rawDataSize'='189597', 
  'totalSize'='190733', 
  'transient_lastDdlTime'='1689619411');

CREATE EXTERNAL TABLE `jbs_trace`(
  `id` bigint COMMENT 'from deserializer', 
  `job_id` bigint COMMENT 'from deserializer', 
  `data_load_id` string COMMENT 'from deserializer', 
  `tbl_name` string COMMENT 'from deserializer', 
  `stage` string COMMENT 'from deserializer', 
  `type` string COMMENT 'from deserializer', 
  `val` map<string,string> COMMENT 'from deserializer')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.hbase.HBaseSerDe' 
STORED BY 
  'org.apache.hadoop.hive.hbase.HBaseStorageHandler' 
WITH SERDEPROPERTIES ( 
  'colelction.delim'=',', 
  'hbase.columns.mapping'=':key,trace_details:parent_id,trace_details:data_load_id,trace_details:table_name,trace_details:stage,trace_details:table,trace_details:val', 
  'mapkey.delim'=':', 
  'serialization.format'='1')
TBLPROPERTIES (
  'COLUMN_STATS_ACCURATE'='{\"BASIC_STATS\":\"true\"}', 
  'hbase.table.name'='sgmisitetl:jbs_trace', 
  'numFiles'='0', 
  'numRows'='0', 
  'rawDataSize'='0', 
  'totalSize'='0', 
  'transient_lastDdlTime'='1531276980');

CREATE EXTERNAL TABLE `sgmi_sg_gm_mandate`(
  `gm_id` string, 
  `approved_date` string, 
  `cust_crid` string, 
  `cr_cust_le_id` string, 
  `lmp_long_nm` string, 
  `lmp_sub_sgmnt_cd_num` string, 
  `lmp_sgmnt_cd_num` string, 
  `rev_grp_id` string, 
  `rev_grp_name` string, 
  `grp_scb_cg` string, 
  `orig_cntry` string, 
  `credit_risk_cntry` string, 
  `prop_lmt_cat1_stlending` string, 
  `prop_lmt_cat1_ltlending` string, 
  `prop_lmt_cat1_trade` string, 
  `prop_lmt_cat1_others` string, 
  `prop_lmt_cat2_st` string, 
  `prop_lmt_cat2_lt` string, 
  `prop_lmt_cat3` string, 
  `prop_lmt_ctrllmt` string, 
  `prop_lmt_cat5` string, 
  `sap_limits` string, 
  `grp_prim_isic_cd` string, 
  `grp_sec_isic_cd` string, 
  `modified_date` string)
PARTITIONED BY ( 
  `ods` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/sgmi_sg_gm_mandate'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619402', 
  'transient_lastDdlTime'='1689619402');

CREATE EXTERNAL TABLE `sgmi_sg_sci`(
  `lmt_le_id` string, 
  `lmt_lsp_id` string, 
  `lmp_long_nm` string, 
  `lmp_sub_sgmnt_cd_val` string, 
  `lmp_sgmnt_cd_val` string, 
  `doc_crm_id` string, 
  `llp_bca_ref_num` string, 
  `llp_bca_ref_appr_dt` string, 
  `cntry_crd_risk_res` string, 
  `cntry_lmt_bkg_loctn_id` string, 
  `lmt_bkg_cd` string, 
  `bca_org_loc` string, 
  `bca_org_bkg_cd` string, 
  `bca_ccr` string, 
  `llp_extd_next_rvw_dt` string, 
  `llp_next_intrm_rvw_dt` string, 
  `llp_cntry_risk_appr_req_ind` string, 
  `lmt_sys_gen_id` string, 
  `facility_id` string, 
  `lmt_outer_lmt_id` string, 
  `lmt_tp_val` string, 
  `lmt_synd_loan_val` string, 
  `lmt_tenor` string, 
  `lmt_tenor_basis_val` string, 
  `lmt_bkg_loctn_id` string, 
  `lmt_prd_tp_val` string, 
  `lmt_amt` string, 
  `lmt_crrncy_iso_cd` string, 
  `lmt_share_ind` string, 
  `lmt_cmmtd_ind` string, 
  `lmt_advise_ind` string, 
  `lmt_sec_tp_val` string, 
  `lmt_expry_dt` string, 
  `lmt_rvlvg_flag_val` string, 
  `bca_gm_id` string, 
  `last_upd_dt` string, 
  `llp_client_lgd` string, 
  `llp_holdco_cd_val` string, 
  `llp_non_gam_cd_val` string)
PARTITIONED BY ( 
  `ods` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/sgmi_sg_sci'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619407', 
  'transient_lastDdlTime'='1689619407');

CREATE EXTERNAL TABLE `sgmi_sg_scorecard`(
  `cust_id` string, 
  `cust_le_id` string, 
  `cust_name` string, 
  `cust_sub_seg` string, 
  `cust_seg` string, 
  `cust_cmp_id` string, 
  `sales_rev` string, 
  `supply_chain_flag` string, 
  `is_large_corp` string, 
  `sugg_scorecard` string, 
  `override_flag` string, 
  `override_comments` string, 
  `scorecard_type` string, 
  `scorecard_id` string, 
  `rm_name` string, 
  `fin_stmt_date` string, 
  `appr_name` string, 
  `date_of_appr` string, 
  `parent_sup_applied` string, 
  `parent_cap_applied` string, 
  `parent_cap_crg` string, 
  `parent_sci_id` string, 
  `parent_cust_id` string, 
  `parent_rating` string, 
  `parent_pd` string, 
  `parent_crg_just` string, 
  `rating_aft_parent_sup` string, 
  `govt_sup_flag` string, 
  `crg_aft_govt_sup` string, 
  `govt_sup_comm` string, 
  `govt_crg_aft_ctry_ceiling` string, 
  `govt_ctry_ceiling` string, 
  `govt_sup_crg` string, 
  `govt_sup_ctry_code` string, 
  `govt_sup_ctry_flag` string, 
  `standalone_rating` string, 
  `standalone_rating_pd` string, 
  `ctry_lcy_ceiling` string, 
  `final_recommended_rating` string, 
  `final_scorecard_crg` string, 
  `final_approved_rating` string, 
  `final_approved_pd` string, 
  `scorecard_pd` string, 
  `recommended_pd` string, 
  `override_status_flag` string, 
  `reason_for_crg_override` string, 
  `justification_for_override` string, 
  `approval_override_comments` string, 
  `approval_comments` string, 
  `crg_of_domicile_country` string, 
  `crg_aft_sover_ceiling_factor` string, 
  `parent_regulatory_country` string, 
  `client_regulatory_country` string, 
  `status` string, 
  `gm_orig_cntry` string, 
  `gm_gam_loc` string, 
  `cob_bca_ref_no` string, 
  `cob_bca_ccr` string, 
  `cob_bca_loc` string, 
  `parent_incorp_country` string, 
  `origin_country` string, 
  `parent_domicile_country` string, 
  `parent_gov_supp_country` string, 
  `bor_bca_ref_no` string, 
  `bor_bca_ccr` string, 
  `bor_bca_loc` string, 
  `pledgor_bca_ref_no` string, 
  `pledgor_bca_ccr` string, 
  `pledgor_bca_loc` string, 
  `fila_bca_ref_no` string, 
  `fila_bca_loc` string, 
  `modified_date` string)
PARTITIONED BY ( 
  `ods` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/sgmi_sg_scorecard'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619403', 
  'transient_lastDdlTime'='1689619403');

CREATE EXTERNAL TABLE `sgmi_sg_sharholding`(
  `sci_id` string, 
  `shareholder_prcnt` string, 
  `shareholder_name` string, 
  `shareholder_id` string, 
  `bca_ref_no` string, 
  `approval_date` string, 
  `update_date` string)
PARTITIONED BY ( 
  `ods` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
LOCATION
  'hdfs://haasbatchreg.hbrscb.dev.net/sit/sgmi/hdata/sgmisitetl/sgmi_sg_sharholding'
TBLPROPERTIES (
  'external.table.purge'='true', 
  'last_modified_by'='hive', 
  'last_modified_time'='1689619405', 
  'transient_lastDdlTime'='1689619405');