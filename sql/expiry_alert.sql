-- Expiry Alert Query
SELECT 
    am."Material Number" as batch_id,
    am."Trial Name" as trial,
    am."Expiry Date" as expiry_date,
    am."Quantity Reserved" as quantity_at_risk,
    EXTRACT(DAY FROM (am."Expiry Date" - CURRENT_DATE)) as days_remaining,
    CASE 
        WHEN am."Expiry Date" <= CURRENT_DATE + INTERVAL '30 days' THEN 'Critical'
        WHEN am."Expiry Date" <= CURRENT_DATE + INTERVAL '60 days' THEN 'High'
        ELSE 'Medium'
    END as risk_level
FROM allocated_materials am
WHERE am."Expiry Date" <= CURRENT_DATE + INTERVAL '90 days'
  AND am."Quantity Reserved" > 0
ORDER BY am."Expiry Date" ASC;


