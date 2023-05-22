CREATE EXTERNAL TABLE IF NOT EXISTS fraud(
    user_id DOUBLE
    ,monto_count_aceptada DOUBLE
    ,diff_hours_max DOUBLE
    ,cashback_min_en_proceso DOUBLE
    ,num_ciudades DOUBLE
    ,monto_mean_aceptada DOUBLE
    ,dispositivo_ano_2018 DOUBLE
    ,cashback_max_aceptada DOUBLE
    ,establecimiento_supermercado DOUBLE
    ,interes_tc DOUBLE
    ,establecimiento_tienda_departamental DOUBLE
    ,establecimiento_compra_en_linea DOUBLE
    ,cashback_mean_rechazada DOUBLE
    ,cashback_max_en_proceso DOUBLE
    ,created_at TIMESTAMP
)
STORED AS PARQUET
LOCATION 's3://bucket-ums/athena/predicteds/'
tblproperties ("parquet.compression"="GZIP")