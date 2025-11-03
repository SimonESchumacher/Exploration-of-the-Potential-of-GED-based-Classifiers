use `2025_02_04_experiments`;
-- N.B.: Also replace the optuna database name in the view definition below.

create or replace view results_per_fold as
select e.id                             as experiment_id,
       s.optuna_study_id                AS optuna_study_id,
       e.dataset_name                   AS dataset_name,
       e.model_name                     AS model_name,
       e.method_name                    AS method_name,
       count(r.test_acc)                AS num_results,
       avg(r.test_acc)                  AS avg_test_acc_fold,
       stddev_samp(r.test_acc)          AS std_test_acc_fold,
       stddev_samp(r.test_acc) /
       sqrt(count(r.test_acc))          AS std_err_test_acc_fold,
       s.best_val_acc                   AS best_val_acc_fold,
       avg(r.test_acc) - s.best_val_acc AS avg_acc_diff_fold,
       s.fold                           AS fold,
       e.num_folds                      AS required_folds,
       ctps.num_completed_trials        AS num_completed_trials_fold,
       s.best_trial_number              AS best_trial_number
from experiments e
         join studies s on e.id = s.experiment_id
         join results r on s.id = r.study_id
         join `2025_02_04_optuna`.completed_trials_per_study ctps
              on s.optuna_study_id = ctps.study_id
group by e.id,
         e.model_name,
         e.dataset_name,
         e.method_name,
         s.optuna_study_id,
         s.fold,
         ctps.num_completed_trials,
         s.best_trial_number
order by e.id, s.fold;

create or replace view results_overall as
select results_per_fold.experiment_id,
       results_per_fold.dataset_name                   AS dataset_name,
       results_per_fold.model_name                     AS model_name,
       results_per_fold.method_name                    AS method_name,
       avg(results_per_fold.avg_test_acc_fold)         AS avg_test_acc,
       stddev_samp(results_per_fold.avg_test_acc_fold) AS std_test_acc,
       stddev_samp(results_per_fold.avg_test_acc_fold) /
       sqrt(count(results_per_fold.fold))              AS std_err_test_acc,
       avg(results_per_fold.best_val_acc_fold)         AS avg_best_val_acc,
       stddev_samp(results_per_fold.best_val_acc_fold) AS std_best_val_acc,
       stddev_samp(results_per_fold.best_val_acc_fold) /
       sqrt(count(results_per_fold.fold))              AS std_err_best_val_acc,
       avg(results_per_fold.avg_test_acc_fold) -
       avg(results_per_fold.best_val_acc_fold)         as avg_acc_diff,
       count(results_per_fold.fold)                    AS num_folds,
       sum(results_per_fold.num_results)               AS num_results,
       e.num_folds * e.num_test_rounds                 AS expected_num_results,
       min(results_per_fold.num_completed_trials_fold) AS min_completed_trials,
       max(results_per_fold.num_completed_trials_fold) AS max_completed_trials,
       sum(results_per_fold.num_completed_trials_fold) AS sum_completed_trials,
       avg(results_per_fold.best_trial_number)         AS avg_best_trial_number,
       stddev_samp(results_per_fold.best_trial_number) AS std_best_trial_number,
       min(results_per_fold.best_trial_number)         AS min_best_trial_number,
       max(results_per_fold.best_trial_number)         AS max_best_trial_number
from results_per_fold
         join experiments e on results_per_fold.experiment_id = e.id
group by results_per_fold.experiment_id,
         results_per_fold.model_name,
         results_per_fold.dataset_name,
         results_per_fold.method_name
order by expected_num_results,
         results_per_fold.dataset_name,
         results_per_fold.model_name,
         results_per_fold.method_name;


create or replace view mixup_hyperparams_per_fold as
select e.id        as experiment_id,
       s.optuna_study_id,
       e.dataset_name,
       e.model_name,
       e.method_name,
       e.num_folds as required_folds,
       s.fold,
       bsp.`key`,
       bsp.value
from best_study_params bsp
         join studies s on s.id = bsp.study_id
         join experiments e on s.experiment_id = e.id
where bsp.`key` IN ('use_vanilla', 'mixup_alpha', 'augmented_ratio')
order by bsp.`key`, bsp.value;

create or replace view mixup_hparams_overall as
select e.id                 as experiment_id,
       e.dataset_name,
       e.model_name,
       e.method_name,
       `key`,
       avg(`value`)         as avg,
       stddev_samp(`value`) as std,
       min(`value`)         as min,
       max(`value`)         as max,
       required_folds,
       count(`value`)       as cnt
from mixup_hyperparams_per_fold mhpf
         join experiments e on mhpf.experiment_id = e.id
group by experiment_id, model_name, dataset_name, method_name, required_folds,
         `key`
order by required_folds, `key`, avg(`value`);
