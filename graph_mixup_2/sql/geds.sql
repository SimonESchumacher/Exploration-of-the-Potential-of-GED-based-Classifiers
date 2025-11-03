use gamma_graphs;

create or replace view ged_computations as
select d.name,
       count(*)                                    as total,
       count(case when geds.value > -1 then 1 end) as positives,
       count(case when geds.value = -1 then 1 end) as negatives_total,
       count(case
                 when geds.value = -1 and geds.time = 0
                     then 1 end)                   as negatives_lower_bound,
       count(case
                 when geds.value = -1 and geds.time > 0
                     then 1 end)                   as negatives_timeout
from geds
         join graphs g0 on g0.graph_id = geds.graph_0_id
         join graphs g1 on g1.graph_id = geds.graph_1_id
         join datasets d on g0.dataset_id = d.dataset_id
where g0.dataset_id = g1.dataset_id
-- exclude geds of mixup graphs
  and not exists (select 1
                  from mixup_attrs ma
                  where ma.graph_id = g0.graph_id)
  and not exists(select 1
                 from mixup_attrs ma
                 where ma.graph_id = g1.graph_id)
group by d.name;

-- no. of mixup graphs
create or replace view mixup_graphs as
select d.name,
       ma.mixup_method,
       count(1)
from mixup_attrs ma
         join graphs g on g.graph_id = ma.graph_id
         join datasets d on d.dataset_id = g.dataset_id
group by d.name, ma.mixup_method;


create or replace view mixup_geds as
select `ma`.`mixup_attr_id`                     AS `mixup_attr_id`,
       `ma`.`mixup_lambda`                      AS `mixup_lambda`,
       `ma`.`mixup_method`                      AS `mixup_method`,
       `ma`.`mixup_hyperparameters`             AS `mixup_hyperparameters`,
       `d`.`name`                               AS `dataset_name`,
       `ma`.`graph_id`                          AS `mixup_graph_id`,
       `ma`.`parent_0_id`                       AS `parent_0_id`,
       `ma`.`parent_1_id`                       AS `parent_1_id`,
       coalesce(`gp1`.`value`, `gp2`.`value`)   AS `ged_parents`,
       coalesce(`g0p1`.`value`, `g0p2`.`value`) AS `ged_mixup_parent_0`,
       coalesce(`g1p1`.`value`, `g1p2`.`value`) AS `ged_mixup_parent_1`
from ((((((((`gamma_graphs`.`mixup_attrs` `ma` join `gamma_graphs`.`graphs` `g`
             on ((`ma`.`graph_id` = `g`.`graph_id`))) join `gamma_graphs`.`datasets` `d`
            on ((`d`.`dataset_id` = `g`.`dataset_id`))) left join `gamma_graphs`.`geds` `gp1`
           on (((`gp1`.`graph_0_id` = `ma`.`parent_0_id`) and
                (`gp1`.`graph_1_id` = `ma`.`parent_1_id`)))) left join `gamma_graphs`.`geds` `gp2`
          on (((`gp2`.`graph_0_id` = `ma`.`parent_1_id`) and
               (`gp2`.`graph_1_id` = `ma`.`parent_0_id`)))) left join `gamma_graphs`.`geds` `g0p1`
         on (((`g0p1`.`graph_0_id` = `ma`.`graph_id`) and
              (`g0p1`.`graph_1_id` = `ma`.`parent_0_id`)))) left join `gamma_graphs`.`geds` `g0p2`
        on (((`g0p2`.`graph_0_id` = `ma`.`parent_0_id`) and
             (`g0p2`.`graph_1_id` = `ma`.`graph_id`)))) left join `gamma_graphs`.`geds` `g1p1`
       on (((`g1p1`.`graph_0_id` = `ma`.`graph_id`) and
            (`g1p1`.`graph_1_id` = `ma`.`parent_1_id`)))) left join `gamma_graphs`.`geds` `g1p2`
      on (((`g1p2`.`graph_0_id` = `ma`.`parent_1_id`) and
           (`g1p2`.`graph_1_id` = `ma`.`graph_id`))));


CREATE or replace VIEW mpd_values AS
(
SELECT ma.mixup_attr_id,
       ma.mixup_lambda,
       d.name                                                        AS dataset_name,
       ma.mixup_method                                               as method_name,
       COALESCE(g1.value, g1_rev.value)                              AS mixup_parent_0_ged,
       COALESCE(g2.value, g2_rev.value)                              AS mixup_parent_1_ged,
       COALESCE(g3.value, g3_rev.value)                              AS parents_ged,

       -- apd1 + apd2 = |d(G_1,G_m) - λ d(G1,G2)| + |d(G_2,G_m) - (1-λ) d(G1,G2)|
       ABS(COALESCE(g1.value, g1_rev.value) -
           ma.mixup_lambda * COALESCE(g3.value, g3_rev.value)) +
       ABS(COALESCE(g2.value, g2_rev.value) -
           (1 - ma.mixup_lambda) * COALESCE(g3.value, g3_rev.value)) as apd,

       ABS(COALESCE(g1.value, g1_rev.value) / COALESCE(g3.value, g3_rev.value) -
           ma.mixup_lambda) +
       ABS(COALESCE(g2.value, g2_rev.value) / COALESCE(g3.value, g3_rev.value) -
           1 + ma.mixup_lambda)                                      as mpd,

       -- apd1_rev + apd2_rev = |d(G_2,G_m) - (1-λ) d(G1,G2)| + |d(G_1,G_m) - λ d(G1,G2)|
       ABS(COALESCE(g2.value, g2_rev.value) -
           ma.mixup_lambda * COALESCE(g3.value, g3_rev.value)) +
       ABS(COALESCE(g1.value, g1_rev.value) -
           (1 - ma.mixup_lambda) * COALESCE(g3.value, g3_rev.value)) as apd_rev,

       ABS(COALESCE(g2.value, g2_rev.value) / COALESCE(g3.value, g3_rev.value) -
           ma.mixup_lambda) +
       ABS(COALESCE(g1.value, g1_rev.value) / COALESCE(g3.value, g3_rev.value) -
           1 + ma.mixup_lambda)                                      as mpd_rev
FROM mixup_attrs ma
         JOIN
     graphs g ON g.graph_id = ma.graph_id
         JOIN
     datasets d ON d.dataset_id = g.dataset_id
         LEFT JOIN
     geds g1
     ON ma.graph_id = g1.graph_0_id AND
        ma.parent_0_id = g1.graph_1_id
         LEFT JOIN
     geds g1_rev ON ma.graph_id = g1_rev.graph_1_id AND
                    ma.parent_0_id = g1_rev.graph_0_id
         LEFT JOIN
     geds g2
     ON ma.graph_id = g2.graph_0_id AND
        ma.parent_1_id = g2.graph_1_id
         LEFT JOIN
     geds g2_rev ON ma.graph_id = g2_rev.graph_1_id AND
                    ma.parent_1_id = g2_rev.graph_0_id
         LEFT JOIN
     geds g3
     ON ma.parent_0_id = g3.graph_0_id AND
        ma.parent_1_id = g3.graph_1_id
         LEFT JOIN
     geds g3_rev ON ma.parent_0_id = g3_rev.graph_1_id AND
                    ma.parent_1_id = g3_rev.graph_0_id
WHERE COALESCE(g1.value, g1_rev.value) IS NOT NULL
  AND COALESCE(g2.value, g2_rev.value) IS NOT NULL
  AND COALESCE(g3.value, g3_rev.value) IS NOT NULL
  AND COALESCE(g1.value, g1_rev.value) > -1
  AND COALESCE(g2.value, g2_rev.value) > -1
  AND COALESCE(g3.value, g3_rev.value) > -1
    );

# ===
# All methods besides SubMix (max. 500 values).
# ===
USE gamma_graphs;
WITH ranked AS (SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY dataset_name, method_name, mixup_lambda
                           ORDER BY mixup_attr_id
                           ) AS rn
                FROM gamma_graphs.mpd_values),
     limited AS (SELECT *
                 FROM ranked
                 WHERE rn <= 500)
SELECT dataset_name,
       method_name,
       mixup_lambda,
       AVG(apd),
       STDDEV_SAMP(apd),
       AVG(mpd),
       STDDEV_SAMP(mpd),
       AVG(apd_rev),
       STDDEV_SAMP(apd_rev),
       AVG(mpd_rev),
       STDDEV_SAMP(mpd_rev),
       COUNT(*) AS `count(1)`
FROM limited
GROUP BY dataset_name, method_name, mixup_lambda
ORDER BY dataset_name, method_name, mixup_lambda;

# ===
# SubMix Values
# ===
use alpha_graphs;
CREATE OR REPLACE VIEW alpha_graphs.aggr_mpd AS
SELECT dataset_name,
       method_name,
       CASE
           WHEN mixup_lambda BETWEEN 0.5 - 0.025 AND 0.5 + 0.025 THEN 0.5
           WHEN mixup_lambda BETWEEN 0.8 - 0.025 AND 0.8 + 0.025 THEN 0.8
           WHEN mixup_lambda BETWEEN 0.9 - 0.025 AND 0.9 + 0.025 THEN 0.9
           END AS lambda_bucket,
       avg(apd_rev),
       stddev_samp(apd_rev),
       avg(mpd_rev),
       stddev_samp(mpd_rev),
       count(1)
from mpd_values
where method_name = 'submix'
group by dataset_name, method_name, lambda_bucket
order by dataset_name, method_name, lambda_bucket;


# ===
# SubMix Aggregates (<= 500)
# ===

CREATE OR REPLACE VIEW alpha_graphs.aggr_mpd AS
WITH bucketed AS (SELECT mixup_attr_id,
                         dataset_name,
                         method_name,
                         CASE
                             WHEN mixup_lambda BETWEEN 0.5 - 0.025 AND 0.5 + 0.025
                                 THEN 0.5
                             WHEN mixup_lambda BETWEEN 0.8 - 0.025 AND 0.8 + 0.025
                                 THEN 0.8
                             WHEN mixup_lambda BETWEEN 0.9 - 0.025 AND 0.9 + 0.025
                                 THEN 0.9
                             END AS lambda_bucket,
                         apd,
                         mpd,
                         apd_rev,
                         mpd_rev
                  FROM mpd_values
                  WHERE method_name = 'submix'),
     ranked AS (SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY dataset_name, method_name, lambda_bucket
                           ORDER BY mixup_attr_id
                           ) AS rn
                FROM bucketed),
     limited as (SELECT *
                 FROM ranked
                 WHERE rn <= 500)
SELECT dataset_name,
       method_name,
       lambda_bucket,
       AVG(apd),
       STDDEV_SAMP(apd),
       AVG(mpd),
       STDDEV_SAMP(mpd),
       AVG(apd_rev),
       STDDEV_SAMP(apd_rev),
       AVG(mpd_rev),
       STDDEV_SAMP(mpd_rev),
       COUNT(*)
FROM limited
WHERE rn <= 500
GROUP BY dataset_name, method_name, lambda_bucket
ORDER BY dataset_name, method_name, lambda_bucket;

