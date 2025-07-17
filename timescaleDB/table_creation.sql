-- Create stock trading data table for TimescaleDB
CREATE TABLE IF NOT EXISTS stock_data (
    date DATE NOT NULL,
    close DECIMAL(20, 8) NOT NULL,  -- Changed to handle small decimal values with high precision
    high DECIMAL(20, 8) NOT NULL,   -- 20 total digits, 8 decimal places
    low DECIMAL(20, 8) NOT NULL,    -- Can handle values like 0.05959763 accurately
    open DECIMAL(20, 8) NOT NULL,
    volume BIGINT NOT NULL
);

-- Convert to TimescaleDB hypertable (makes it a time-series database)
SELECT create_hypertable('stock_data', 'date', 
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 month');

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data (date);
CREATE INDEX IF NOT EXISTS idx_stock_data_volume ON stock_data (volume DESC);
CREATE INDEX IF NOT EXISTS idx_stock_data_close ON stock_data (close);

-- Optional: Create continuous aggregate for monthly statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS stock_data_monthly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 month', date) AS month,
    COUNT(*) as trading_days,
    AVG(close) as avg_close_price,
    MAX(high) as month_high,
    MIN(low) as month_low,
    SUM(volume) as total_volume,
    -- Calculate monthly return
    (LAST(close, date) - FIRST(close, date)) / FIRST(close, date) * 100 as monthly_return_pct
FROM stock_data
GROUP BY month;
--WITH NO DATA;

-- Enable continuous aggregate policy (optional - auto-refreshes the view)
-- TimescaleDB needs at least 2 time buckets to work properly with continuous aggregates.

SELECT add_continuous_aggregate_policy('stock_data_monthly',
                                      start_offset => INTERVAL '3 months',
                                      end_offset => INTERVAL '1 day',
                                      schedule_interval => INTERVAL '1 day');

-- Enable compression for storage optimization
ALTER TABLE stock_data SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'date DESC'
);

-- Add compression policy - compress data older than 3 months (aligned with continuous aggregate)
SELECT add_compression_policy('stock_data', INTERVAL '3 months');

-- Add retention policy - automatically delete data older than 7 years
SELECT add_retention_policy('stock_data', INTERVAL '7 years');

select * from  timescaledb_information.jobs ; 


select pg_size_pretty(before_compression_total_bytes) as "before comporession", 
pg_size_pretty(after_compression_total_bytes) as "after compression"
from hypertable_compression_stats('stock_data'); 

SELECT Date , close from stock_data
WHERE Date > '2023-01-01' and Date < now() ; 




SELECT Date , close from stock_data
WHERE Date > now() - interval '1 month'; 


select date , volume from stock_data 
order by volume desc  
limit 10 ; 

-- avg close price for the last month 
select 
avg(close) 
from stock_data s 
WHERE Date > now() - interval '1 month'; 


select first(close , date), last(close , date)
from stock_data s 
where Date > now() - interval '2 months' ; 


SELECT 
    time_bucket('1 month', date) as month,
    FIRST(close, date) as month_first_close,
    LAST(close, date) as month_last_close
FROM stock_data 
WHERE date > now() - interval '2 months'
GROUP BY month
ORDER BY month;


select 
time_bucket('1 month', date) as month, 
avg(close) as close_avg, 
avg(volume) as volume_avg
from stock_data s
group by month
order by month ; 


