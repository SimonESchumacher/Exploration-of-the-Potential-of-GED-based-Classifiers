USE `iota_opt`;

CREATE OR REPLACE VIEW completed_trials_per_study AS
SELECT s.study_id,
       s.study_name,
       min(t.datetime_start) as earliest_trial_start,
       COUNT(t.trial_id) AS num_completed_trials
FROM studies s
         JOIN trials t ON s.study_id = t.study_id
WHERE t.state = 'COMPLETE'
GROUP BY s.study_name
ORDER BY num_completed_trials;
