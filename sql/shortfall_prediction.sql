-- Shortfall Prediction Query
WITH Trial_Supply AS (
    SELECT 
        "Trial Name" as trial_id,
        SUM("Quantity Available") as total_stock
    FROM available_inventory_report
    GROUP BY "Trial Name"
),
Trial_Demand AS (
    SELECT 
        "trial_alias" as trial_id,
        "enrollment_rate_monthly_actual" as monthly_rate
    FROM study_level_enrollment_report
)
SELECT 
    s.trial_id,
    s.total_stock,
    d.monthly_rate,
    ROUND(d.monthly_rate / 4, 2) as weekly_burn_rate,
    ROUND(s.total_stock / NULLIF((d.monthly_rate / 4), 0), 1) as weeks_of_coverage
FROM Trial_Supply s
INNER JOIN Trial_Demand d ON s.trial_id = d.trial_id
WHERE (s.total_stock / NULLIF((d.monthly_rate / 4), 0)) < 8
ORDER BY weeks_of_coverage ASC;


