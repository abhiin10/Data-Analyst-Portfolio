
-- =====================================================
-- PROJECT: RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION
-- PURPOSE: RETAIL SALES DATA ANALYSIS
-- DATABASE: retail_analytics
-- =====================================================

USE retail_analytics;

-- 1. View sample records
SELECT *
FROM train_sample_100k
LIMIT 10;

-- 2. Total number of records
SELECT COUNT(*) AS total_records
FROM train_sample_100k;

-- 3. Total and average sales
SELECT
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(AVG(sales), 2) AS average_sales
FROM train_sample_100k;

-- 4. Sales by product family
SELECT
    family,
    ROUND(SUM(sales), 2) AS total_sales
FROM train_sample_100k
GROUP BY family
ORDER BY total_sales DESC;

-- 5. Average sales by product family
SELECT
    family,
    ROUND(AVG(sales), 2) AS average_sales
FROM train_sample_100k
GROUP BY family
ORDER BY average_sales DESC;

-- 6. Total sales by store
SELECT
    store_nbr,
    ROUND(SUM(sales), 2) AS total_sales
FROM train_sample_100k
GROUP BY store_nbr
ORDER BY total_sales DESC;

-- 7. Top 10 product families by sales
SELECT
    family,
    ROUND(SUM(sales), 2) AS total_sales
FROM train_sample_100k
GROUP BY family
ORDER BY total_sales DESC
LIMIT 10;

-- 8. Sales by promotion status
SELECT
    CASE
        WHEN onpromotion > 0 THEN 'Promotion'
        ELSE 'No Promotion'
    END AS promotion_status,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(AVG(sales), 2) AS average_sales
FROM train_sample_100k
GROUP BY promotion_status;

-- 9. Product families with sales above 1 million
SELECT
    family,
    ROUND(SUM(sales), 2) AS total_sales
FROM train_sample_100k
GROUP BY family
HAVING SUM(sales) > 1000000
ORDER BY total_sales DESC;

-- 10. Sales by date
SELECT
    date,
    ROUND(SUM(sales), 2) AS daily_sales
FROM train_sample_100k
GROUP BY date
ORDER BY date;
