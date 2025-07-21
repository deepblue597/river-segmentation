# Grafana Dashboard Queries for River Segmentation Pipeline

This document contains all the Grafana dashboard queries for monitoring the river segmentation pipeline with TimescaleDB as the data source.

## 📊 Dashboard Structure

### 1. **River Monitoring Overview**

### 2. **Model Performance Metrics**

### 3. **System Health & Alerts**

### 4. **Historical Analysis**

### 5. **Real-time Image Monitoring**

### 6. **📈 Microsoft Stock Data Dashboard**

---

## 🌊 1. River Monitoring Overview

### Water Coverage Percentage (Time Series)

```sql
SELECT
  time_bucket('5m', timestamp) AS time,
  AVG(water_coverage_percent) as avg_water_coverage,
  MAX(water_coverage_percent) as max_water_coverage,
  MIN(water_coverage_percent) as min_water_coverage
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('5m', timestamp)
ORDER BY time;
```

### Current Water Level Status (Stat Panel)

```sql
SELECT
  water_coverage_percent as "Current Water Coverage %"
FROM river_predictions
ORDER BY timestamp DESC
LIMIT 1;
```

### Water Level Trend (Stat Panel with Sparkline)

```sql
SELECT
  timestamp as time,
  water_coverage_percent as value
FROM river_predictions
WHERE $__timeFilter(timestamp)
ORDER BY timestamp;
```

### Overflow Alert Status (Stat Panel)

```sql
SELECT
  CASE
    WHEN water_coverage_percent > 80 THEN 'CRITICAL'
    WHEN water_coverage_percent > 60 THEN 'WARNING'
    ELSE 'NORMAL'
  END as alert_status,
  COUNT(*) as count
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY alert_status
ORDER BY count DESC
LIMIT 1;
```

---

## 🤖 2. Model Performance Metrics

### Model Confidence Score (Time Series)

```sql
SELECT
  time_bucket('5m', timestamp) AS time,
  AVG(confidence_score) as avg_confidence,
  MIN(confidence_score) as min_confidence,
  MAX(confidence_score) as max_confidence
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('5m', timestamp)
ORDER BY time;
```

### Prediction Accuracy Distribution (Histogram)

```sql
SELECT
  CASE
    WHEN confidence_score >= 0.9 THEN '90-100%'
    WHEN confidence_score >= 0.8 THEN '80-90%'
    WHEN confidence_score >= 0.7 THEN '70-80%'
    WHEN confidence_score >= 0.6 THEN '60-70%'
    ELSE 'Below 60%'
  END as confidence_range,
  COUNT(*) as prediction_count
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY confidence_range
ORDER BY confidence_range;
```

### Model Processing Time (Time Series)

```sql
SELECT
  time_bucket('5m', timestamp) AS time,
  AVG(processing_time_ms) as avg_processing_time,
  MAX(processing_time_ms) as max_processing_time
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('5m', timestamp)
ORDER BY time;
```

### Low Confidence Predictions (Table)

```sql
SELECT
  timestamp,
  image_id,
  water_coverage_percent,
  confidence_score,
  processing_time_ms
FROM river_predictions
WHERE confidence_score < 0.7
  AND $__timeFilter(timestamp)
ORDER BY timestamp DESC
LIMIT 50;
```

---

## 🚨 3. System Health & Alerts

### Images Processed Per Hour (Time Series)

```sql
SELECT
  time_bucket('1h', timestamp) AS time,
  COUNT(*) as images_processed
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('1h', timestamp)
ORDER BY time;
```

### System Uptime & Data Gaps (Time Series)

```sql
SELECT
  time_bucket('10m', timestamp) AS time,
  COUNT(*) as data_points,
  CASE
    WHEN COUNT(*) = 0 THEN 0
    ELSE 1
  END as system_status
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('10m', timestamp)
ORDER BY time;
```

### Alert Summary (Stat Panels)

```sql
-- Critical Alerts (Last 24h)
SELECT COUNT(*) as critical_alerts
FROM river_predictions
WHERE water_coverage_percent > 80
  AND timestamp > NOW() - INTERVAL '24 hours';

-- Warning Alerts (Last 24h)
SELECT COUNT(*) as warning_alerts
FROM river_predictions
WHERE water_coverage_percent BETWEEN 60 AND 80
  AND timestamp > NOW() - INTERVAL '24 hours';

-- Low Confidence Predictions (Last 24h)
SELECT COUNT(*) as low_confidence_count
FROM river_predictions
WHERE confidence_score < 0.7
  AND timestamp > NOW() - INTERVAL '24 hours';
```

### Error Rate (Time Series)

```sql
SELECT
  time_bucket('1h', timestamp) AS time,
  COUNT(CASE WHEN confidence_score < 0.5 THEN 1 END) * 100.0 / COUNT(*) as error_rate
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('1h', timestamp)
ORDER BY time;
```

---

## 📈 4. Historical Analysis

### Daily Water Level Summary (Time Series)

```sql
SELECT
  time_bucket('1 day', timestamp) AS time,
  AVG(water_coverage_percent) as avg_daily_coverage,
  MAX(water_coverage_percent) as max_daily_coverage,
  MIN(water_coverage_percent) as min_daily_coverage,
  STDDEV(water_coverage_percent) as coverage_stddev
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('1 day', timestamp)
ORDER BY time;
```

### Weekly Trend Analysis (Time Series)

```sql
SELECT
  time_bucket('1 week', timestamp) AS time,
  AVG(water_coverage_percent) as avg_weekly_coverage,
  COUNT(*) as total_predictions,
  AVG(confidence_score) as avg_weekly_confidence
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('1 week', timestamp)
ORDER BY time;
```

### Seasonal Patterns (Bar Chart)

```sql
SELECT
  EXTRACT(HOUR FROM timestamp) as hour_of_day,
  AVG(water_coverage_percent) as avg_water_coverage
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY EXTRACT(HOUR FROM timestamp)
ORDER BY hour_of_day;
```

### Monthly Statistics (Table)

```sql
SELECT
  DATE_TRUNC('month', timestamp) as month,
  COUNT(*) as total_predictions,
  AVG(water_coverage_percent) as avg_water_coverage,
  MAX(water_coverage_percent) as max_water_coverage,
  MIN(water_coverage_percent) as min_water_coverage,
  AVG(confidence_score) as avg_confidence,
  COUNT(CASE WHEN water_coverage_percent > 80 THEN 1 END) as critical_events
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month DESC;
```

---

## 🖼️ 5. Real-time Image Monitoring

### Latest Prediction Details (Table)

```sql
SELECT
  timestamp,
  image_id,
  image_url,
  water_coverage_percent,
  confidence_score,
  processing_time_ms,
  model_version
FROM river_predictions
ORDER BY timestamp DESC
LIMIT 20;
```

### Image Processing Rate (Gauge)

```sql
SELECT
  COUNT(*) * 60.0 / EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) as images_per_minute
FROM river_predictions
WHERE timestamp > NOW() - INTERVAL '10 minutes';
```

### Recent High Water Events (Table)

```sql
SELECT
  timestamp,
  image_id,
  water_coverage_percent,
  confidence_score,
  image_url
FROM river_predictions
WHERE water_coverage_percent > 70
  AND $__timeFilter(timestamp)
ORDER BY timestamp DESC
LIMIT 10;
```

---

## 🔧 Advanced Queries

### Water Level Change Rate (Time Series)

```sql
SELECT
  timestamp,
  water_coverage_percent,
  water_coverage_percent - LAG(water_coverage_percent) OVER (ORDER BY timestamp) as coverage_change
FROM river_predictions
WHERE $__timeFilter(timestamp)
ORDER BY timestamp;
```

### Prediction Correlation Analysis

```sql
SELECT
  time_bucket('1h', timestamp) AS time,
  CORR(water_coverage_percent, confidence_score) as coverage_confidence_correlation
FROM river_predictions
WHERE $__timeFilter(timestamp)
GROUP BY time_bucket('1h', timestamp)
ORDER BY time;
```

### Anomaly Detection (Outliers)

```sql
WITH stats AS (
  SELECT
    AVG(water_coverage_percent) as mean_coverage,
    STDDEV(water_coverage_percent) as stddev_coverage
  FROM river_predictions
  WHERE $__timeFilter(timestamp)
)
SELECT
  timestamp,
  image_id,
  water_coverage_percent,
  confidence_score,
  ABS(water_coverage_percent - stats.mean_coverage) / stats.stddev_coverage as z_score
FROM river_predictions, stats
WHERE $__timeFilter(timestamp)
  AND ABS(water_coverage_percent - stats.mean_coverage) / stats.stddev_coverage > 2
ORDER BY z_score DESC
LIMIT 20;
```

---

## 🎯 Alert Rules

### Critical Water Level Alert

```sql
SELECT
  timestamp,
  water_coverage_percent
FROM river_predictions
WHERE water_coverage_percent > 80
  AND timestamp > NOW() - INTERVAL '5 minutes'
LIMIT 1;
```

### Model Performance Degradation Alert

```sql
SELECT
  AVG(confidence_score) as avg_confidence
FROM river_predictions
WHERE timestamp > NOW() - INTERVAL '30 minutes'
HAVING AVG(confidence_score) < 0.7;
```

### Data Pipeline Health Check

```sql
SELECT
  COUNT(*) as recent_predictions
FROM river_predictions
WHERE timestamp > NOW() - INTERVAL '10 minutes'
HAVING COUNT(*) < 5;
```

---

## 📋 Dashboard Variables

### Time Range Variables

```sql
-- For dropdown selection of common time ranges
SELECT DISTINCT
  '1h' as text, '1h' as value
UNION ALL
SELECT '6h' as text, '6h' as value
UNION ALL
SELECT '24h' as text, '24h' as value
UNION ALL
SELECT '7d' as text, '7d' as value;
```

### Model Version Filter

```sql
SELECT DISTINCT model_version
FROM river_predictions
WHERE $__timeFilter(timestamp)
ORDER BY model_version;
```

### Water Level Threshold Variable

```sql
SELECT DISTINCT
  '60' as text, 60 as value
UNION ALL
SELECT '70' as text, 70 as value
UNION ALL
SELECT '80' as text, 80 as value
UNION ALL
SELECT '90' as text, 90 as value;
```

---

## 💡 Dashboard Tips

1. **Use Time Buckets**: For better performance with large datasets, always use `time_bucket()` for time series queries
2. **Index Strategy**: Ensure indexes on `timestamp`, `water_coverage_percent`, and `confidence_score` columns
3. **Variable Usage**: Use dashboard variables like `$threshold` in queries: `WHERE water_coverage_percent > $threshold`
4. **Refresh Rates**: Set appropriate refresh rates (30s for real-time, 5m for historical)
5. **Alert Thresholds**: Customize alert thresholds based on your specific river conditions

## 🚀 Quick Setup

1. Import these queries into Grafana panels
2. Configure TimescaleDB as data source: `localhost:5432`
3. Set up alerts for critical water levels
4. Create notification channels (email, Slack, etc.)
5. Organize panels into logical dashboard sections

---

## 📈 6. Microsoft Stock Data Dashboard

### Stock Price Overview (Time Series)

```sql
SELECT
  date as time,
  open as "Open Price",
  high as "High Price",
  low as "Low Price",
  close as "Close Price"
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY date;
```

### Current Stock Price (Stat Panel)

```sql
SELECT
  close as "Current Price"
FROM stock_data
ORDER BY date DESC
LIMIT 1;
```

### Daily Price Change (Stat Panel)

```sql
SELECT
  close - LAG(close) OVER (ORDER BY date) as price_change,
  (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 as price_change_pct
FROM stock_data
ORDER BY date DESC
LIMIT 1;
```

### Trading Volume (Time Series)

```sql
SELECT
  date as time,
  volume as "Trading Volume"
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY date;
```

### Stock Price Candlestick Chart

```sql
SELECT
  date as time,
  open as "Open",
  high as "High",
  low as "Low",
  close as "Close"
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY date;
```

### Monthly Stock Performance (Time Series)

```sql
SELECT
  time_bucket('1 month', date) as time,
  AVG(close) as avg_monthly_price,
  MAX(high) as monthly_high,
  MIN(low) as monthly_low,
  SUM(volume) as total_monthly_volume
FROM stock_data
WHERE $__timeFilter(date)
GROUP BY time_bucket('1 month', date)
ORDER BY time;
```

### Weekly Returns (Time Series)

```sql
SELECT
  time_bucket('1 week', date) as time,
  (LAST(close, date) - FIRST(close, date)) / FIRST(close, date) * 100 as weekly_return_pct
FROM stock_data
WHERE $__timeFilter(date)
GROUP BY time_bucket('1 week', date)
ORDER BY time;
```

### Price Movement Analysis (Time Series)

```sql
SELECT
  date as time,
  close,
  close - LAG(close) OVER (ORDER BY date) as daily_change,
  (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 as daily_change_pct
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY date;
```

### Volume Analysis (Bar Chart)

```sql
SELECT
  date,
  volume,
  CASE
    WHEN volume > AVG(volume) OVER () THEN 'Above Average'
    ELSE 'Below Average'
  END as volume_category
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY date DESC
LIMIT 30;
```

### Top Trading Days by Volume (Table)

```sql
SELECT
  date,
  close as "Close Price",
  volume as "Trading Volume",
  (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 as "Daily Change %"
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY volume DESC
LIMIT 20;
```

### Price Volatility (Time Series)

```sql
SELECT
  time_bucket('1 week', date) as time,
  STDDEV(close) as price_volatility,
  (MAX(high) - MIN(low)) / AVG(close) * 100 as weekly_range_pct
FROM stock_data
WHERE $__timeFilter(date)
GROUP BY time_bucket('1 week', date)
ORDER BY time;
```

### Moving Averages (Time Series)

```sql
SELECT
  date as time,
  close,
  AVG(close) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as "7-day MA",
  AVG(close) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as "30-day MA",
  AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as "50-day MA"
FROM stock_data
WHERE $__timeFilter(date)
ORDER BY date;
```

### Support and Resistance Levels (Stat Panels)

```sql
-- Recent High (Resistance)
SELECT MAX(high) as "52-Week High"
FROM stock_data
WHERE date > NOW() - INTERVAL '52 weeks';

-- Recent Low (Support)
SELECT MIN(low) as "52-Week Low"
FROM stock_data
WHERE date > NOW() - INTERVAL '52 weeks';

-- Average Price
SELECT AVG(close) as "52-Week Average"
FROM stock_data
WHERE date > NOW() - INTERVAL '52 weeks';
```

### Monthly Statistics (Table)

```sql
SELECT
  DATE_TRUNC('month', date) as month,
  COUNT(*) as trading_days,
  ROUND(AVG(close)::numeric, 2) as avg_price,
  ROUND(MAX(high)::numeric, 2) as month_high,
  ROUND(MIN(low)::numeric, 2) as month_low,
  ROUND(SUM(volume)::numeric, 0) as total_volume,
  ROUND(((LAST(close, date) - FIRST(close, date)) / FIRST(close, date) * 100)::numeric, 2) as monthly_return_pct
FROM stock_data
WHERE $__timeFilter(date)
GROUP BY DATE_TRUNC('month', date)
ORDER BY month DESC;
```

### Price Distribution (Histogram)

```sql
SELECT
  CASE
    WHEN close < 100 THEN 'Under $100'
    WHEN close < 200 THEN '$100-200'
    WHEN close < 300 THEN '$200-300'
    WHEN close < 400 THEN '$300-400'
    ELSE 'Over $400'
  END as price_range,
  COUNT(*) as frequency
FROM stock_data
WHERE $__timeFilter(date)
GROUP BY price_range
ORDER BY price_range;
```

### Gap Analysis (Table)

```sql
SELECT
  date,
  open,
  LAG(close) OVER (ORDER BY date) as prev_close,
  open - LAG(close) OVER (ORDER BY date) as gap,
  (open - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 as gap_pct
FROM stock_data
WHERE $__timeFilter(date)
  AND ABS(open - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) > 0.02
ORDER BY ABS(gap_pct) DESC
LIMIT 20;
```

### Performance Metrics (Stat Panels)

```sql
-- YTD Return
SELECT
  (LAST(close, date) - FIRST(close, date)) / FIRST(close, date) * 100 as ytd_return
FROM stock_data
WHERE EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE);

-- Best Day
SELECT
  MAX((close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100) as best_day_return
FROM stock_data
WHERE $__timeFilter(date);

-- Worst Day
SELECT
  MIN((close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100) as worst_day_return
FROM stock_data
WHERE $__timeFilter(date);
```

### Stock Alerts (Alert Rules)

```sql
-- Price Drop Alert (>5% in one day)
SELECT
  date,
  close,
  (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 as daily_change_pct
FROM stock_data
WHERE date = CURRENT_DATE - INTERVAL '1 day'
  AND (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 < -5;

-- High Volume Alert
SELECT
  date,
  volume,
  AVG(volume) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND 1 PRECEDING) as avg_volume_30d
FROM stock_data
WHERE date = CURRENT_DATE - INTERVAL '1 day'
  AND volume > (AVG(volume) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND 1 PRECEDING) * 2);
```

### Dashboard Variables for Stock Data

```sql
-- Time Period Variable
SELECT DISTINCT
  '1 week' as text, '1 week' as value
UNION ALL
SELECT '1 month' as text, '1 month' as value
UNION ALL
SELECT '3 months' as text, '3 months' as value
UNION ALL
SELECT '1 year' as text, '1 year' as value;

-- Price Threshold Variable
SELECT DISTINCT
  '100' as text, 100 as value
UNION ALL
SELECT '200' as text, 200 as value
UNION ALL
SELECT '300' as text, 300 as value
UNION ALL
SELECT '400' as text, 400 as value;
```
