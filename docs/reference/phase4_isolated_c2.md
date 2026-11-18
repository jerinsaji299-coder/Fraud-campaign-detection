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
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | f1 | test | all | campaign_recall | 0.13392857142857142 | 8 | 0.13392857142857142 | 8 | 0.1517857142857143 | 0.05357142857142857 | 3 | 0.19642857142857142 | 4 |  |  | 0 | 0 | nan |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | per_inst_10 | test | all | campaign_recall | 0.12723214285714285 | 8 | 0.12723214285714285 | 8 | 0.125 | 0.05357142857142857 | 3 | 0.19642857142857142 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | per_inst_50 | test | all | campaign_recall | 0.14732142857142855 | 8 | 0.14732142857142855 | 8 | 0.1607142857142857 | 0.05357142857142857 | 3 | 0.23214285714285715 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | per_inst_100 | test | all | campaign_recall | 0.171875 | 8 | 0.171875 | 8 | 0.1875 | 0.10714285714285714 | 3 | 0.23214285714285715 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | total_10 | test | all | campaign_recall | 0.07589285714285715 | 8 | 0.07589285714285715 | 8 | 0.0625 | 0.03571428571428571 | 3 | 0.19642857142857142 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | total_50 | test | all | campaign_recall | 0.10267857142857142 | 8 | 0.10267857142857142 | 8 | 0.08928571428571427 | 0.03571428571428571 | 6 | 0.19642857142857142 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 0 | total_100 | test | all | campaign_recall | 0.13392857142857142 | 8 | 0.13392857142857142 | 8 | 0.1517857142857143 | 0.05357142857142857 | 3 | 0.19642857142857142 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | f1 | test | all | campaign_recall | 0.109375 | 8 | 0.109375 | 8 | 0.10714285714285715 | 0.05357142857142857 | 1 | 0.19642857142857142 | 2 |  |  | 0 | 0 | nan |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | per_inst_10 | test | all | campaign_recall | 0.10714285714285715 | 8 | 0.10714285714285715 | 8 | 0.08928571428571429 | 0.05357142857142857 | 3 | 0.19642857142857142 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | per_inst_50 | test | all | campaign_recall | 0.1361607142857143 | 8 | 0.1361607142857143 | 8 | 0.125 | 0.07142857142857142 | 3 | 0.23214285714285715 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | per_inst_100 | test | all | campaign_recall | 0.16294642857142858 | 8 | 0.16294642857142858 | 8 | 0.16071428571428573 | 0.10714285714285714 | 3 | 0.26785714285714285 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | total_10 | test | all | campaign_recall | 0.05803571428571429 | 8 | 0.05803571428571429 | 8 | 0.05357142857142857 | 0.0 | 2 | 0.125 | 5 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | total_50 | test | all | campaign_recall | 0.08928571428571427 | 8 | 0.08928571428571427 | 8 | 0.08035714285714285 | 0.03571428571428571 | 7 | 0.16071428571428573 | 5 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 1 | total_100 | test | all | campaign_recall | 0.12276785714285714 | 8 | 0.12276785714285714 | 8 | 0.125 | 0.07142857142857142 | 0 | 0.19642857142857142 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | f1 | test | all | campaign_recall | 0.125 | 8 | 0.125 | 8 | 0.14285714285714285 | 0.03571428571428571 | 2 | 0.17857142857142858 | 0 |  |  | 0 | 0 | nan |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | per_inst_10 | test | all | campaign_recall | 0.109375 | 8 | 0.109375 | 8 | 0.125 | 0.03571428571428571 | 2 | 0.17857142857142858 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | per_inst_50 | test | all | campaign_recall | 0.13839285714285715 | 8 | 0.13839285714285715 | 8 | 0.1517857142857143 | 0.03571428571428571 | 3 | 0.19642857142857142 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | per_inst_100 | test | all | campaign_recall | 0.1607142857142857 | 8 | 0.1607142857142857 | 8 | 0.1875 | 0.03571428571428571 | 3 | 0.23214285714285715 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | total_10 | test | all | campaign_recall | 0.05357142857142857 | 8 | 0.05357142857142857 | 8 | 0.0625 | 0.0 | 2 | 0.125 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | total_50 | test | all | campaign_recall | 0.10044642857142858 | 8 | 0.10044642857142858 | 8 | 0.10714285714285715 | 0.03571428571428571 | 2 | 0.17857142857142858 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.75 | 48 | 2 | total_100 | test | all | campaign_recall | 0.12053571428571429 | 8 | 0.12053571428571429 | 8 | 0.13392857142857142 | 0.03571428571428571 | 2 | 0.17857142857142858 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | f1 | test | all | campaign_recall | 0.12276785714285714 | 8 | 0.12276785714285714 | 8 | 0.125 | 0.017857142857142856 | 0 | 0.21428571428571427 | 6 |  |  | 0 | 0 | nan |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | per_inst_10 | test | all | campaign_recall | 0.11830357142857142 | 8 | 0.11830357142857142 | 8 | 0.125 | 0.017857142857142856 | 2 | 0.25 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | per_inst_50 | test | all | campaign_recall | 0.16964285714285712 | 8 | 0.16964285714285712 | 8 | 0.20535714285714285 | 0.05357142857142857 | 2 | 0.2857142857142857 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | per_inst_100 | test | all | campaign_recall | 0.1808035714285714 | 8 | 0.1808035714285714 | 8 | 0.20535714285714285 | 0.07142857142857142 | 2 | 0.2857142857142857 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | total_10 | test | all | campaign_recall | 0.07589285714285714 | 8 | 0.07589285714285714 | 8 | 0.04464285714285714 | 0.0 | 2 | 0.25 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | total_50 | test | all | campaign_recall | 0.09821428571428571 | 8 | 0.09821428571428571 | 8 | 0.08035714285714285 | 0.017857142857142856 | 2 | 0.25 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 0 | total_100 | test | all | campaign_recall | 0.125 | 8 | 0.125 | 8 | 0.125 | 0.017857142857142856 | 2 | 0.25 | 4 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | f1 | test | all | campaign_recall | 0.11160714285714285 | 8 | 0.11160714285714285 | 8 | 0.09821428571428571 | 0.017857142857142856 | 6 | 0.21428571428571427 | 5 |  |  | 0 | 0 | nan |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | per_inst_10 | test | all | campaign_recall | 0.12053571428571429 | 8 | 0.12053571428571429 | 8 | 0.11607142857142858 | 0.05357142857142857 | 3 | 0.21428571428571427 | 5 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | per_inst_50 | test | all | campaign_recall | 0.1450892857142857 | 8 | 0.1450892857142857 | 8 | 0.1517857142857143 | 0.07142857142857142 | 0 | 0.23214285714285715 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | per_inst_100 | test | all | campaign_recall | 0.171875 | 8 | 0.171875 | 8 | 0.17857142857142858 | 0.08928571428571429 | 0 | 0.23214285714285715 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | total_10 | test | all | campaign_recall | 0.04241071428571429 | 8 | 0.04241071428571429 | 8 | 0.05357142857142857 | 0.0 | 1 | 0.07142857142857142 | 0 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | total_50 | test | all | campaign_recall | 0.10267857142857142 | 8 | 0.10267857142857142 | 8 | 0.10714285714285714 | 0.05357142857142857 | 3 | 0.16071428571428573 | 1 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 1 | total_100 | test | all | campaign_recall | 0.125 | 8 | 0.125 | 8 | 0.11607142857142858 | 0.05357142857142857 | 3 | 0.21428571428571427 | 2 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | f1 | test | all | campaign_recall | 0.11160714285714286 | 8 | 0.11160714285714286 | 8 | 0.11607142857142858 | 0.017857142857142856 | 1 | 0.23214285714285715 | 7 |  |  | 0 | 0 | nan |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | per_inst_10 | test | all | campaign_recall | 0.09821428571428571 | 8 | 0.09821428571428571 | 8 | 0.09821428571428571 | 0.017857142857142856 | 1 | 0.17857142857142858 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | per_inst_50 | test | all | campaign_recall | 0.1361607142857143 | 8 | 0.1361607142857143 | 8 | 0.13392857142857142 | 0.05357142857142857 | 2 | 0.23214285714285715 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | per_inst_100 | test | all | campaign_recall | 0.16964285714285715 | 8 | 0.16964285714285715 | 8 | 0.16964285714285715 | 0.10714285714285714 | 3 | 0.2857142857142857 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | total_10 | test | all | campaign_recall | 0.040178571428571425 | 8 | 0.040178571428571425 | 8 | 0.04464285714285714 | 0.0 | 3 | 0.07142857142857142 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | total_50 | test | all | campaign_recall | 0.08705357142857142 | 8 | 0.08705357142857142 | 8 | 0.08035714285714285 | 0.017857142857142856 | 1 | 0.17857142857142858 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |
| isolated | xgboost | xgboost_centralized | 0.5 | 48 | 2 | total_100 | test | all | campaign_recall | 0.10491071428571429 | 8 | 0.10491071428571429 | 8 | 0.09821428571428571 | 0.03571428571428571 | 2 | 0.17857142857142858 | 7 |  |  | 8 | 8 | 1.0 |  | 8 | 0 | 0.0 | 0.01 |

## The union (test, subset all)

`total_*` points give each institution B / K so the summed volume
meets B; `per_inst_*` points give each institution the full B, and
their summed volume is printed beside them. `per_inst_*` is not a
budgeted point for an isolated institution - the binding count
beside it says how often the budget set the threshold at all.

| visibility_target | seed | operating_point | campaign_recall | oracle_recall | recall_relative_to_ceiling | false_alarms_per_day | summed_false_alarms_per_day | duplicate_clusters | n_institutions_operating |
|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 0 | f1 | 0.5892857142857143 | 0.8928571428571429 | 0.66 | 57.47 | 58.32 | 28 | 8 |
| 0.75 | 0 | per_inst_10 | 0.5714285714285714 | 0.8928571428571429 | 0.6399999999999999 | 36.21 | 37.05 | 28 | 8 |
| 0.75 | 0 | per_inst_50 | 0.6428571428571429 | 0.8928571428571429 | 0.72 | 213.47 | 213.47 | 45 | 8 |
| 0.75 | 0 | per_inst_100 | 0.6785714285714286 | 0.8928571428571429 | 0.76 | 422.11 | 422.11 | 51 | 8 |
| 0.75 | 0 | total_10 | 0.44642857142857145 | 0.8928571428571429 | 0.5 | 3.58 | 3.58 | 12 | 8 |
| 0.75 | 0 | total_50 | 0.5178571428571429 | 0.8928571428571429 | 0.5800000000000001 | 20.84 | 20.84 | 21 | 8 |
| 0.75 | 0 | total_100 | 0.5892857142857143 | 0.8928571428571429 | 0.66 | 64.63 | 65.47 | 38 | 8 |
| 0.75 | 1 | f1 | 0.5 | 0.8928571428571429 | 0.5599999999999999 | 38.95 | 38.95 | 12 | 8 |
| 0.75 | 1 | per_inst_10 | 0.5178571428571429 | 0.8928571428571429 | 0.5800000000000001 | 31.16 | 31.16 | 11 | 8 |
| 0.75 | 1 | per_inst_50 | 0.5714285714285714 | 0.8928571428571429 | 0.6399999999999999 | 169.89 | 169.89 | 23 | 8 |
| 0.75 | 1 | per_inst_100 | 0.6785714285714286 | 0.8928571428571429 | 0.76 | 438.95 | 438.95 | 31 | 8 |
| 0.75 | 1 | total_10 | 0.39285714285714285 | 0.8928571428571429 | 0.43999999999999995 | 3.37 | 3.37 | 1 | 8 |
| 0.75 | 1 | total_50 | 0.48214285714285715 | 0.8928571428571429 | 0.54 | 17.89 | 17.89 | 8 | 8 |
| 0.75 | 1 | total_100 | 0.5178571428571429 | 0.8928571428571429 | 0.5800000000000001 | 55.79 | 55.79 | 9 | 8 |
| 0.75 | 2 | f1 | 0.625 | 0.8928571428571429 | 0.7 | 74.53 | 74.53 | 15 | 8 |
| 0.75 | 2 | per_inst_10 | 0.6071428571428571 | 0.8928571428571429 | 0.6799999999999999 | 38.74 | 38.74 | 10 | 8 |
| 0.75 | 2 | per_inst_50 | 0.625 | 0.8928571428571429 | 0.7 | 196.0 | 196.42 | 18 | 8 |
| 0.75 | 2 | per_inst_100 | 0.6785714285714286 | 0.8928571428571429 | 0.76 | 500.63 | 501.89 | 33 | 8 |
| 0.75 | 2 | total_10 | 0.39285714285714285 | 0.8928571428571429 | 0.43999999999999995 | 1.89 | 1.89 | 4 | 8 |
| 0.75 | 2 | total_50 | 0.5892857142857143 | 0.8928571428571429 | 0.66 | 26.11 | 26.11 | 11 | 8 |
| 0.75 | 2 | total_100 | 0.6071428571428571 | 0.8928571428571429 | 0.6799999999999999 | 45.68 | 45.68 | 13 | 8 |
| 0.5 | 0 | f1 | 0.5535714285714286 | 0.8928571428571429 | 0.62 | 86.74 | 86.74 | 10 | 8 |
| 0.5 | 0 | per_inst_10 | 0.5 | 0.8928571428571429 | 0.5599999999999999 | 45.89 | 45.89 | 12 | 8 |
| 0.5 | 0 | per_inst_50 | 0.6071428571428571 | 0.8928571428571429 | 0.6799999999999999 | 218.32 | 218.32 | 30 | 8 |
| 0.5 | 0 | per_inst_100 | 0.6785714285714286 | 0.8928571428571429 | 0.76 | 438.95 | 438.95 | 36 | 8 |
| 0.5 | 0 | total_10 | 0.4107142857142857 | 0.8928571428571429 | 0.45999999999999996 | 2.32 | 2.32 | 5 | 8 |
| 0.5 | 0 | total_50 | 0.4642857142857143 | 0.8928571428571429 | 0.52 | 26.53 | 26.53 | 7 | 8 |
| 0.5 | 0 | total_100 | 0.5178571428571429 | 0.8928571428571429 | 0.5800000000000001 | 53.05 | 53.05 | 14 | 8 |
| 0.5 | 1 | f1 | 0.6071428571428571 | 0.8928571428571429 | 0.6799999999999999 | 61.05 | 61.05 | 1 | 8 |
| 0.5 | 1 | per_inst_10 | 0.5714285714285714 | 0.8928571428571429 | 0.6399999999999999 | 27.58 | 27.79 | 3 | 8 |
| 0.5 | 1 | per_inst_50 | 0.6428571428571429 | 0.8928571428571429 | 0.72 | 131.37 | 131.79 | 4 | 8 |
| 0.5 | 1 | per_inst_100 | 0.7321428571428571 | 0.8928571428571429 | 0.82 | 506.95 | 508.21 | 15 | 8 |
| 0.5 | 1 | total_10 | 0.2857142857142857 | 0.8928571428571429 | 0.31999999999999995 | 1.89 | 1.89 | 0 | 8 |
| 0.5 | 1 | total_50 | 0.5 | 0.8928571428571429 | 0.5599999999999999 | 20.84 | 20.84 | 0 | 8 |
| 0.5 | 1 | total_100 | 0.5892857142857143 | 0.8928571428571429 | 0.66 | 35.37 | 35.58 | 3 | 8 |
| 0.5 | 2 | f1 | 0.5178571428571429 | 0.8928571428571429 | 0.5800000000000001 | 76.42 | 76.42 | 15 | 8 |
| 0.5 | 2 | per_inst_10 | 0.48214285714285715 | 0.8928571428571429 | 0.54 | 27.37 | 27.37 | 13 | 8 |
| 0.5 | 2 | per_inst_50 | 0.5714285714285714 | 0.8928571428571429 | 0.6399999999999999 | 148.21 | 148.42 | 17 | 8 |
| 0.5 | 2 | per_inst_100 | 0.6607142857142857 | 0.8928571428571429 | 0.74 | 471.37 | 473.05 | 29 | 8 |
| 0.5 | 2 | total_10 | 0.26785714285714285 | 0.8928571428571429 | 0.3 | 4.0 | 4.0 | 0 | 8 |
| 0.5 | 2 | total_50 | 0.48214285714285715 | 0.8928571428571429 | 0.54 | 22.74 | 22.74 | 8 | 8 |
| 0.5 | 2 | total_100 | 0.5 | 0.8928571428571429 | 0.5599999999999999 | 43.58 | 43.79 | 14 | 8 |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 32.7 | 3,168 MB | yes |
| assignment t075_s0 | 1.0 | 3,168 MB | yes |
| evaluate t075_s0 | 2,755.0 | 4,426 MB | yes |
| assignment t075_s1 | 0.9 | 4,426 MB | yes |
| evaluate t075_s1 | 2,655.7 | 4,426 MB | yes |
| assignment t075_s2 | 0.9 | 4,426 MB | yes |
| evaluate t075_s2 | 2,610.6 | 4,450 MB | yes |
| assignment t050_s0 | 0.9 | 4,450 MB | yes |
| evaluate t050_s0 | 2,603.9 | 4,491 MB | yes |
| assignment t050_s1 | 0.9 | 4,491 MB | yes |
| evaluate t050_s1 | 2,636.4 | 4,521 MB | yes |
| assignment t050_s2 | 0.9 | 4,521 MB | yes |
| evaluate t050_s2 | 2,627.2 | 4,521 MB | yes |
