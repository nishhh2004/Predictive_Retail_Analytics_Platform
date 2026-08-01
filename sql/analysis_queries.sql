-- Q1: Total sales by product family (Top 10)
SELECT family,
       ROUND(SUM(sales), 2) AS total_sales,
       COUNT(*) AS record_count
FROM sales
GROUP BY family
ORDER BY total_sales DESC
LIMIT 10;

-- Q2: Monthly sales trend
SELECT DATE_FORMAT(date, '%Y-%m') AS month,
       ROUND(SUM(sales), 2) AS monthly_sales
FROM sales
GROUP BY month
ORDER BY month;

-- Q3: Sales distribution by store type
SELECT s.type,
       COUNT(DISTINCT s.store_nbr) AS num_stores,
       ROUND(SUM(sa.sales), 2) AS total_sales,
       ROUND(AVG(sa.sales), 2) AS avg_daily_sales
FROM sales sa
JOIN stores s ON sa.store_nbr = s.store_nbr
GROUP BY s.type
ORDER BY total_sales DESC;

-- Q4: Sales with store info + oil price + transactions (Master JOIN)
SELECT sa.date,
       sa.store_nbr,
       s.city,
       s.state,
       s.type,
       sa.family,
       sa.sales,
       sa.onpromotion,
       o.dcoilwtico AS oil_price,
       t.transactions
FROM sales sa
JOIN stores s ON sa.store_nbr = s.store_nbr
LEFT JOIN oil o ON sa.date = o.date
LEFT JOIN transactions t ON sa.date = t.date
                         AND sa.store_nbr = t.store_nbr
WHERE sa.date BETWEEN '2017-01-01' AND '2017-08-15'
ORDER BY sa.sales DESC
LIMIT 100;

-- Q5: Holiday impact on sales
SELECT h.type,
       h.description,
       ROUND(AVG(sa.sales), 2) AS avg_sales_on_holiday
FROM sales sa
JOIN holidays h ON sa.date = h.date
GROUP BY h.type, h.description
ORDER BY avg_sales_on_holiday DESC
LIMIT 15;

-- Q6: Sales performance by state with store count
SELECT s.state,
       COUNT(DISTINCT s.store_nbr) AS num_stores,
       ROUND(SUM(sa.sales), 2) AS total_sales,
       ROUND(SUM(sa.sales) / COUNT(DISTINCT s.store_nbr), 2) AS sales_per_store
FROM sales sa
JOIN stores s ON sa.store_nbr = s.store_nbr
GROUP BY s.state
ORDER BY sales_per_store DESC;

-- Q7: Rolling 7-day and 30-day average sales (Window Functions)
WITH daily_sales AS (
    SELECT date,
           ROUND(SUM(sales), 2) AS total_daily_sales
    FROM sales
    GROUP BY date
)
SELECT date,
       total_daily_sales,
       ROUND(AVG(total_daily_sales) OVER (
           ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_7d_avg,
       ROUND(AVG(total_daily_sales) OVER (
           ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_30d_avg
FROM daily_sales
ORDER BY date;

-- Q8: Year-over-Year growth by product family
WITH yearly_sales AS (
    SELECT family,
           YEAR(date) AS yr,
           ROUND(SUM(sales), 2) AS annual_sales
    FROM sales
    GROUP BY family, YEAR(date)
)
SELECT family,
       yr,
       annual_sales,
       LAG(annual_sales) OVER (PARTITION BY family ORDER BY yr) AS prev_year_sales,
       ROUND(
           (annual_sales - LAG(annual_sales) OVER (PARTITION BY family ORDER BY yr))
           / NULLIF(LAG(annual_sales) OVER (PARTITION BY family ORDER BY yr), 0) * 100
       , 2) AS yoy_growth_pct
FROM yearly_sales
ORDER BY family, yr;

-- Q9: Top 5 stores per state by total sales (RANK)
WITH store_sales AS (
    SELECT s.state,
           sa.store_nbr,
           s.city,
           ROUND(SUM(sa.sales), 2) AS total_sales,
           RANK() OVER (PARTITION BY s.state ORDER BY SUM(sa.sales) DESC) AS rnk
    FROM sales sa
    JOIN stores s ON sa.store_nbr = s.store_nbr
    GROUP BY s.state, sa.store_nbr, s.city
)
SELECT state, store_nbr, city, total_sales, rnk
FROM store_sales
WHERE rnk <= 5
ORDER BY state, rnk;

-- Q10: Promotion effectiveness by product family
SELECT family,
       ROUND(AVG(CASE WHEN onpromotion > 0 THEN sales END), 2) AS avg_sales_with_promo,
       ROUND(AVG(CASE WHEN onpromotion = 0 THEN sales END), 2) AS avg_sales_without_promo,
       ROUND(
           (AVG(CASE WHEN onpromotion > 0 THEN sales END) -
            AVG(CASE WHEN onpromotion = 0 THEN sales END))
           / NULLIF(AVG(CASE WHEN onpromotion = 0 THEN sales END), 0) * 100
       , 2) AS promo_lift_pct
FROM sales
GROUP BY family
HAVING avg_sales_with_promo IS NOT NULL
ORDER BY promo_lift_pct DESC;
