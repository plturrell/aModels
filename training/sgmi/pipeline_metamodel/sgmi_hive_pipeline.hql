CREATE TABLE sgmi_sample.sales (
  order_id BIGINT,
  customer_id BIGINT,
  amount DECIMAL(10,2),
  ordered_at TIMESTAMP
)
PARTITIONED BY (business_date STRING)
STORED AS PARQUET;
