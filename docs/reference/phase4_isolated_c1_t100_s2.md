# Stage 4.1: the isolated condition

Each institution alone: trained on its own visible transactions,
scoring its own visible transactions, forming clusters inside its
own view. No model averaging, no boundary-account sharing. This is
the floor `fedavg` and `fedavg_exchange` have to beat.

**An institution is restricted by its scorer, not by its data.**
The transactions, campaigns, deadlines, detection grid and horizons
are the global ones Phase 3 used; the scorer returns 0 outside the
institution's view. Visibility restricts what can be scored, never
what is true.

**Each institution has its own ceiling**, the oracle through the
same mask. The centralized ceiling is never reused for an
institution's number: an institution that sees a quarter of a
campaign's edges could not reach it.

**Two pooled numbers, never averaged together.** The
per-institution mean is reported twice - over all institutions and
over trained institutions only - because one mean conflates 'cannot
see the campaign' with 'had no labels to learn from'. The union is a
**supervisory** number: a regulator or FIU receiving all K alert
streams, not a bank. No bank can compute it or benefit from the
deduplication.

**The LEVEL of the per-institution mean is not interpretable on
its own - only differences against 4.2 and 4.3, paired per
institution, are.** K = 8 is a construction, not a sample of
banks, so no number here may be quoted as what a bank achieves
alone. In a paired difference the hub institution sits on both
sides and cancels; in a level it does not. No standard error is
ever taken across the eight institutions.

**The eight institutions are not exchangeable, so neither mean is
an estimate of what a typical bank achieves.** Banks are walked
into institutions largest-volume-first, into the least loaded, with
ties going to the lowest index - so the dataset's single largest
bank lands in institution 0 every time. Total volume is balanced to
within 3%, but institution 0 is one hub bank plus small ones while
the other seven are many small ones, and a hub is an endpoint of
disproportionately many edges. On the full data institution 0 held
the most training positives in 12 of 12 cells, 44% above the
cross-institution mean, on 2% more rows. Each pooled mean therefore
carries its median, min and max with the institutions that produced
them: the mean is the mean over this fixed panel, and the spread is
what stops it being read as a typical bank.

A booster loaded from disk carries no training detail, so the row
counts, positive counts and class weights are in
`train_isolated.csv` rather than here. The taus are identical
either way: a booster trained in process and the same booster
saved and reloaded choose the same threshold.

## The two pooled numbers (test, campaign recall)

| condition | model | variant | visibility_target | lookback_h | seed | operating_point | eval_split | test_subset | metric | mean_over_all_institutions | n_institutions_in_mean | mean_over_trained_institutions | n_trained_in_mean | median_over_institutions | min_over_institutions | min_institution | max_over_institutions | max_institution | untrained_institutions | undefined_institutions | n_cells_with_a_budget | n_binding_budgets | binding_fraction | non_binding_institutions | n_cells_with_an_f1_curve | n_f1_degenerate | f1_degenerate_fraction | f1_degenerate_epsilon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | f1 | test | all | campaign_recall | 0.09438775510204081 | 7 | 0.09438775510204081 | 7 | 0.07142857142857142 | 0.0 | 6 | 0.23214285714285715 | 1 |  | 2 | 0 | 0 | nan |  | 7 | 1 | 0.14285714285714285 | 0.01 |
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | per_inst_10 | test | all | campaign_recall | 0.05803571428571429 | 8 | 0.05803571428571429 | 8 | 0.05357142857142857 | 0.0 | 7 | 0.14285714285714285 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 2 | 0.25 | 0.01 |
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | per_inst_50 | test | all | campaign_recall | 0.08258928571428571 | 8 | 0.08258928571428571 | 8 | 0.08035714285714285 | 0.017857142857142856 | 6 | 0.16071428571428573 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 2 | 0.25 | 0.01 |
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | per_inst_100 | test | all | campaign_recall | 0.09598214285714285 | 8 | 0.09598214285714285 | 8 | 0.08035714285714285 | 0.03571428571428571 | 7 | 0.19642857142857142 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 2 | 0.25 | 0.01 |
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | total_10 | test | all | campaign_recall | 0.04241071428571429 | 8 | 0.04241071428571429 | 8 | 0.03571428571428571 | 0.0 | 6 | 0.125 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 2 | 0.25 | 0.01 |
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | total_50 | test | all | campaign_recall | 0.05133928571428571 | 8 | 0.05133928571428571 | 8 | 0.05357142857142857 | 0.0 | 6 | 0.125 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 2 | 0.25 | 0.01 |
| isolated | xgboost | xgboost_centralized | 1.0 | 48 | 2 | total_100 | test | all | campaign_recall | 0.05803571428571429 | 8 | 0.05803571428571429 | 8 | 0.05357142857142857 | 0.0 | 7 | 0.14285714285714285 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 2 | 0.25 | 0.01 |

## The union (test, subset all)

`total_*` points give each institution B / K so the summed volume
meets B; `per_inst_*` points give each institution the full B, and
their summed volume is printed beside them. `per_inst_*` is not a
budgeted point for an isolated institution - the binding count
beside it says how often the budget set the threshold at all.

| visibility_target | seed | operating_point | campaign_recall | oracle_recall | recall_relative_to_ceiling | false_alarms_per_day | summed_false_alarms_per_day | duplicate_clusters | n_institutions_operating |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 2 | f1 | 0.5714285714285714 | 0.8928571428571429 | 0.6399999999999999 | 2119.16 | 2119.37 | 1 | 7 |
| 1.0 | 2 | per_inst_10 | 0.42857142857142855 | 0.8928571428571429 | 0.4799999999999999 | 29.47 | 29.47 | 1 | 8 |
| 1.0 | 2 | per_inst_50 | 0.5714285714285714 | 0.8928571428571429 | 0.6399999999999999 | 162.11 | 162.11 | 6 | 8 |
| 1.0 | 2 | per_inst_100 | 0.625 | 0.8928571428571429 | 0.7 | 499.58 | 500.21 | 12 | 8 |
| 1.0 | 2 | total_10 | 0.3392857142857143 | 0.8928571428571429 | 0.38 | 2.95 | 2.95 | 0 | 8 |
| 1.0 | 2 | total_50 | 0.4107142857142857 | 0.8928571428571429 | 0.45999999999999996 | 10.32 | 10.32 | 0 | 8 |
| 1.0 | 2 | total_100 | 0.42857142857142855 | 0.8928571428571429 | 0.4799999999999999 | 29.47 | 29.47 | 1 | 8 |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 35.6 | 3,169 MB | yes |
| assignment t100_s2 | 0.0 | 3,169 MB | yes |
| evaluate t100_s2 | 2,977.8 | 4,357 MB | yes |
