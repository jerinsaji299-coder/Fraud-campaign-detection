# Phase 4 chain on the 1% sample: 4.1 -> 4.2 -> 4.3 -> 4.4 -> compare

> **DEVELOPMENT ONLY - not a result.** This is the committed copy of the
> Phase 4 dry run, kept as evidence that the chain runs end to end and that
> its checks hold - **not** as a measurement of anything. Reproduce with
> `python scripts/run_phase4_chain.py --config configs/local.yaml`.
>
> The run exited **3**, for 12 F1 taus at the LOW edge of the grid. On the 1%
> sample that is expected and the sample runners record it; on the full data
> the same event is a stop condition and the Kaggle runner refuses to record
> the cell.
>
> 1% sample. The positive rate is ~3.3% against the full data's 0.102%, so precision, false alarms per day, budget-chosen taus and binding counts are properties of the thinned sample. Not a result; not comparable to Phase 3's full-data centralized 0.679 / 0.760.

## Scope

```json
{
  "targets": [
    1.0,
    0.75,
    0.5,
    0.25
  ],
  "seeds": [
    0
  ],
  "lookback_h": 48,
  "splits": [
    "val",
    "test"
  ],
  "conditions": [
    "isolated",
    "fedavg",
    "evidence",
    "fedavg_evidence"
  ],
  "centralized_policy": "matched_seed0",
  "centralized_model_seeds": [
    0
  ],
  "centralized_retrained_here": true
}
```

## Checks that must hold before any difference is read

```json
{
  "ceilings_agree_across_conditions": {
    "n_ceiling_cells": {
      "evidence": 256,
      "fedavg": 256,
      "fedavg_evidence": 256,
      "isolated": 256
    },
    "max_disagreement": 0.0
  },
  "union_ceiling_equals_centralized": {
    "compared": 32,
    "max_difference": 0.0,
    "agrees": true
  }
}
```

## The seed-noise floor

Phase 4's seeds vary visibility only - every booster is model seed 0 -
so model-seed noise is held fixed and invisible here. Phase 3 measures
it at centralized across five model seeds: 0.661-0.696 recall,
0.74-0.78 relative. **A difference smaller than ~0.04 relative recall
(~2 campaigns of 56) must not be reported as an effect**, and every row
below carries `below_seed_noise_floor` saying whether it is.

## Primary points: `f1` and `total_10`, test, subset `all`

| contrast | unit | visibility_target | seed | operating_point | n_pairs | mean_difference | median_difference | min_difference | max_difference | n_positive | n_negative | below_seed_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| centralized - evidence | union_vs_centralized | 1.0 | 0 | f1 | 1 | 0.22000000000000008 | 0.22000000000000008 | 0.22000000000000008 | 0.22000000000000008 | 1 | 0 | False |
| centralized - evidence | union_vs_centralized | 1.0 | 0 | total_10 | 1 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 1 | 0 | False |
| centralized - evidence | union_vs_centralized | 0.75 | 0 | f1 | 1 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | 0 | 1 | True |
| centralized - evidence | union_vs_centralized | 0.75 | 0 | total_10 | 1 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 1 | 0 | False |
| centralized - evidence | union_vs_centralized | 0.5 | 0 | f1 | 1 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | 0 | 1 | True |
| centralized - evidence | union_vs_centralized | 0.5 | 0 | total_10 | 1 | 0.12 | 0.12 | 0.12 | 0.12 | 1 | 0 | False |
| centralized - evidence | union_vs_centralized | 0.25 | 0 | f1 | 1 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | 0 | 1 | True |
| centralized - evidence | union_vs_centralized | 0.25 | 0 | total_10 | 1 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 1 | 0 | False |
| centralized - fedavg | union_vs_centralized | 1.0 | 0 | f1 | 1 | 0.14 | 0.14 | 0.14 | 0.14 | 1 | 0 | False |
| centralized - fedavg | union_vs_centralized | 1.0 | 0 | total_10 | 1 | 0.12 | 0.12 | 0.12 | 0.12 | 1 | 0 | False |
| centralized - fedavg | union_vs_centralized | 0.75 | 0 | f1 | 1 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | 0 | 1 | True |
| centralized - fedavg | union_vs_centralized | 0.75 | 0 | total_10 | 1 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 1 | 0 | False |
| centralized - fedavg | union_vs_centralized | 0.5 | 0 | f1 | 1 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | 0 | 1 | False |
| centralized - fedavg | union_vs_centralized | 0.5 | 0 | total_10 | 1 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 1 | 0 | False |
| centralized - fedavg | union_vs_centralized | 0.25 | 0 | f1 | 1 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | 0 | 1 | True |
| centralized - fedavg | union_vs_centralized | 0.25 | 0 | total_10 | 1 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 1 | 0 | False |
| centralized - fedavg_evidence | union_vs_centralized | 1.0 | 0 | f1 | 1 | 0.14 | 0.14 | 0.14 | 0.14 | 1 | 0 | False |
| centralized - fedavg_evidence | union_vs_centralized | 1.0 | 0 | total_10 | 1 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 1 | 0 | False |
| centralized - fedavg_evidence | union_vs_centralized | 0.75 | 0 | f1 | 1 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | 0 | 1 | False |
| centralized - fedavg_evidence | union_vs_centralized | 0.75 | 0 | total_10 | 1 | 0.02000000000000013 | 0.02000000000000013 | 0.02000000000000013 | 0.02000000000000013 | 1 | 0 | True |
| centralized - fedavg_evidence | union_vs_centralized | 0.5 | 0 | f1 | 1 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | 0 | 1 | False |
| centralized - fedavg_evidence | union_vs_centralized | 0.5 | 0 | total_10 | 1 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 1 | 0 | False |
| centralized - fedavg_evidence | union_vs_centralized | 0.25 | 0 | f1 | 1 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | 0 | 1 | True |
| centralized - fedavg_evidence | union_vs_centralized | 0.25 | 0 | total_10 | 1 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 1 | 0 | False |
| centralized - isolated | union_vs_centralized | 1.0 | 0 | f1 | 1 | 0.14 | 0.14 | 0.14 | 0.14 | 1 | 0 | False |
| centralized - isolated | union_vs_centralized | 1.0 | 0 | total_10 | 1 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 1 | 0 | False |
| centralized - isolated | union_vs_centralized | 0.75 | 0 | f1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| centralized - isolated | union_vs_centralized | 0.75 | 0 | total_10 | 1 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 0.10000000000000009 | 1 | 0 | False |
| centralized - isolated | union_vs_centralized | 0.5 | 0 | f1 | 1 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | 0 | 1 | True |
| centralized - isolated | union_vs_centralized | 0.5 | 0 | total_10 | 1 | 0.07999999999999996 | 0.07999999999999996 | 0.07999999999999996 | 0.07999999999999996 | 1 | 0 | False |
| centralized - isolated | union_vs_centralized | 0.25 | 0 | f1 | 1 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | -0.05999999999999983 | 0 | 1 | False |
| centralized - isolated | union_vs_centralized | 0.25 | 0 | total_10 | 1 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 1 | 0 | False |
| evidence - isolated | per_institution | 1.0 | 0 | f1 | 8 | 0.015625 | 0.038461538461538436 | -0.25 | 0.25 | 4 | 3 | True |
| evidence - isolated | per_institution | 1.0 | 0 | total_10 | 8 | 0.0548878205128205 | 0.07692307692307687 | 0.0 | 0.125 | 5 | 0 | False |
| evidence - isolated | per_institution | 0.75 | 0 | f1 | 8 | 0.09385488176964149 | 0.13157894736842102 | -0.33333333333333337 | 0.2666666666666666 | 7 | 1 | False |
| evidence - isolated | per_institution | 0.75 | 0 | total_10 | 8 | 0.22317092550216122 | 0.08187134502923982 | 0.0 | 0.7368421052631577 | 7 | 0 | False |
| evidence - isolated | per_institution | 0.5 | 0 | f1 | 8 | 0.09166666666666667 | 0.09999999999999998 | 0.0 | 0.19999999999999996 | 5 | 0 | False |
| evidence - isolated | per_institution | 0.5 | 0 | total_10 | 8 | 0.007648601398601393 | 0.0 | -0.2272727272727273 | 0.30000000000000004 | 3 | 2 | True |
| evidence - isolated | per_institution | 0.25 | 0 | f1 | 8 | 0.0007558480375196991 | 0.003591954022988453 | -0.15999999999999992 | 0.11538461538461542 | 4 | 4 | True |
| evidence - isolated | per_institution | 0.25 | 0 | total_10 | 8 | 0.029623914004536178 | 0.04166666666666663 | -0.17391304347826075 | 0.2068965517241379 | 4 | 3 | True |
| evidence - isolated | union | 1.0 | 0 | f1 | 1 | -0.08000000000000007 | -0.08000000000000007 | -0.08000000000000007 | -0.08000000000000007 | 0 | 1 | False |
| evidence - isolated | union | 1.0 | 0 | total_10 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| evidence - isolated | union | 0.75 | 0 | f1 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| evidence - isolated | union | 0.75 | 0 | total_10 | 1 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 1 | 0 | False |
| evidence - isolated | union | 0.5 | 0 | f1 | 1 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | 0 | 1 | True |
| evidence - isolated | union | 0.5 | 0 | total_10 | 1 | -0.040000000000000036 | -0.040000000000000036 | -0.040000000000000036 | -0.040000000000000036 | 0 | 1 | False |
| evidence - isolated | union | 0.25 | 0 | f1 | 1 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | 0 | 1 | True |
| evidence - isolated | union | 0.25 | 0 | total_10 | 1 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | 0 | 1 | True |
| fedavg - isolated | per_institution | 1.0 | 0 | f1 | 8 | 0.048878205128205114 | 0.0 | -0.4166666666666668 | 0.5 | 3 | 1 | False |
| fedavg - isolated | per_institution | 1.0 | 0 | total_10 | 8 | -0.04967948717948722 | 0.0 | -0.4166666666666668 | 0.25 | 1 | 3 | False |
| fedavg - isolated | per_institution | 0.75 | 0 | f1 | 8 | 0.18969457157386221 | 0.1836956521739131 | 0.052631578947368474 | 0.34782608695652173 | 8 | 0 | False |
| fedavg - isolated | per_institution | 0.75 | 0 | total_10 | 8 | 0.12449783879989831 | 0.11521739130434788 | -0.36842105263157887 | 0.6842105263157894 | 6 | 1 | False |
| fedavg - isolated | per_institution | 0.5 | 0 | f1 | 8 | 0.09924242424242426 | 0.04999999999999999 | -0.045454545454545414 | 0.30000000000000004 | 4 | 1 | False |
| fedavg - isolated | per_institution | 0.5 | 0 | total_10 | 8 | -0.02712703962703966 | 0.007575757575757541 | -0.2272727272727273 | 0.15 | 4 | 4 | True |
| fedavg - isolated | per_institution | 0.25 | 0 | f1 | 8 | -0.014803247414754153 | 0.02083333333333337 | -0.4799999999999999 | 0.34782608695652173 | 4 | 3 | True |
| fedavg - isolated | per_institution | 0.25 | 0 | total_10 | 8 | -0.02142918925152809 | 0.0 | -0.3478260869565217 | 0.20833333333333331 | 3 | 3 | True |
| fedavg - isolated | union | 1.0 | 0 | f1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg - isolated | union | 1.0 | 0 | total_10 | 1 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | 0 | 1 | True |
| fedavg - isolated | union | 0.75 | 0 | f1 | 1 | 0.039999999999999925 | 0.039999999999999925 | 0.039999999999999925 | 0.039999999999999925 | 1 | 0 | True |
| fedavg - isolated | union | 0.75 | 0 | total_10 | 1 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 0.06000000000000005 | 1 | 0 | False |
| fedavg - isolated | union | 0.5 | 0 | f1 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| fedavg - isolated | union | 0.5 | 0 | total_10 | 1 | -0.02000000000000013 | -0.02000000000000013 | -0.02000000000000013 | -0.02000000000000013 | 0 | 1 | True |
| fedavg - isolated | union | 0.25 | 0 | f1 | 1 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | -0.039999999999999925 | 0 | 1 | True |
| fedavg - isolated | union | 0.25 | 0 | total_10 | 1 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | 0 | 1 | True |
| fedavg_evidence - evidence | per_institution | 1.0 | 0 | f1 | 8 | 0.12900641025641024 | 0.04166666666666663 | -0.16666666666666674 | 0.75 | 4 | 3 | False |
| fedavg_evidence - evidence | per_institution | 1.0 | 0 | total_10 | 8 | 0.04086538461538461 | 0.0 | 0.0 | 0.25 | 2 | 0 | False |
| fedavg_evidence - evidence | per_institution | 0.75 | 0 | f1 | 8 | 0.11377447241291636 | 0.04347826086956519 | -0.10526315789473684 | 0.6111111111111112 | 4 | 1 | False |
| fedavg_evidence - evidence | per_institution | 0.75 | 0 | total_10 | 8 | 0.018182684973302804 | 0.0 | -0.10526315789473684 | 0.1333333333333333 | 3 | 2 | True |
| fedavg_evidence - evidence | per_institution | 0.5 | 0 | f1 | 8 | 0.042191142191142186 | 0.047727272727272674 | -0.045454545454545414 | 0.15000000000000013 | 5 | 1 | False |
| fedavg_evidence - evidence | per_institution | 0.5 | 0 | total_10 | 8 | 0.03352272727272726 | 0.06818181818181815 | -0.35000000000000003 | 0.20000000000000007 | 5 | 1 | True |
| fedavg_evidence - evidence | per_institution | 0.25 | 0 | f1 | 8 | 0.05161762868565718 | 0.020833333333333315 | -0.07999999999999996 | 0.26086956521739135 | 4 | 1 | False |
| fedavg_evidence - evidence | per_institution | 0.25 | 0 | total_10 | 8 | -0.04814059316495599 | 0.0 | -0.4166666666666667 | 0.10344827586206906 | 2 | 2 | False |
| fedavg_evidence - evidence | union | 1.0 | 0 | f1 | 1 | 0.08000000000000007 | 0.08000000000000007 | 0.08000000000000007 | 0.08000000000000007 | 1 | 0 | False |
| fedavg_evidence - evidence | union | 1.0 | 0 | total_10 | 1 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 1 | 0 | False |
| fedavg_evidence - evidence | union | 0.75 | 0 | f1 | 1 | 0.039999999999999925 | 0.039999999999999925 | 0.039999999999999925 | 0.039999999999999925 | 1 | 0 | True |
| fedavg_evidence - evidence | union | 0.75 | 0 | total_10 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| fedavg_evidence - evidence | union | 0.5 | 0 | f1 | 1 | 0.039999999999999925 | 0.039999999999999925 | 0.039999999999999925 | 0.039999999999999925 | 1 | 0 | True |
| fedavg_evidence - evidence | union | 0.5 | 0 | total_10 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| fedavg_evidence - evidence | union | 0.25 | 0 | f1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - evidence | union | 0.25 | 0 | total_10 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - fedavg | per_institution | 1.0 | 0 | f1 | 8 | 0.09575320512820512 | 0.0625 | 0.0 | 0.3333333333333333 | 4 | 0 | False |
| fedavg_evidence - fedavg | per_institution | 1.0 | 0 | total_10 | 8 | 0.14543269230769232 | 0.10096153846153849 | 0.0 | 0.5 | 5 | 0 | False |
| fedavg_evidence - fedavg | per_institution | 0.75 | 0 | f1 | 8 | 0.017934782608695646 | 0.0 | -0.052631578947368474 | 0.10000000000000009 | 3 | 1 | True |
| fedavg_evidence - fedavg | per_institution | 0.75 | 0 | total_10 | 8 | 0.1168557716755657 | 0.05409356725146203 | 0.0 | 0.3157894736842104 | 7 | 0 | False |
| fedavg_evidence - fedavg | per_institution | 0.5 | 0 | f1 | 8 | 0.0346153846153846 | 0.024999999999999967 | 0.0 | 0.09999999999999998 | 4 | 0 | True |
| fedavg_evidence - fedavg | per_institution | 0.5 | 0 | total_10 | 8 | 0.0682983682983683 | 0.1515151515151515 | -0.35000000000000003 | 0.25 | 5 | 2 | False |
| fedavg_evidence - fedavg | per_institution | 0.25 | 0 | f1 | 8 | 0.06717672413793105 | 0.020833333333333315 | 0.0 | 0.24000000000000005 | 4 | 0 | False |
| fedavg_evidence - fedavg | per_institution | 0.25 | 0 | total_10 | 8 | 0.0029125100911082877 | 0.02243589743589741 | -0.24999999999999994 | 0.2173913043478261 | 4 | 4 | True |
| fedavg_evidence - fedavg | union | 1.0 | 0 | f1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - fedavg | union | 1.0 | 0 | total_10 | 1 | 0.05999999999999994 | 0.05999999999999994 | 0.05999999999999994 | 0.05999999999999994 | 1 | 0 | False |
| fedavg_evidence - fedavg | union | 0.75 | 0 | f1 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| fedavg_evidence - fedavg | union | 0.75 | 0 | total_10 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| fedavg_evidence - fedavg | union | 0.5 | 0 | f1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - fedavg | union | 0.5 | 0 | total_10 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - fedavg | union | 0.25 | 0 | f1 | 1 | 0.020000000000000018 | 0.020000000000000018 | 0.020000000000000018 | 0.020000000000000018 | 1 | 0 | True |
| fedavg_evidence - fedavg | union | 0.25 | 0 | total_10 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - isolated | per_institution | 1.0 | 0 | f1 | 8 | 0.14463141025641024 | 0.10096153846153844 | -0.08333333333333348 | 0.5 | 5 | 1 | False |
| fedavg_evidence - isolated | per_institution | 1.0 | 0 | total_10 | 8 | 0.09575320512820511 | 0.08012820512820512 | 0.0 | 0.25 | 6 | 0 | False |
| fedavg_evidence - isolated | per_institution | 0.75 | 0 | f1 | 8 | 0.20762935418255785 | 0.2304347826086957 | 0.05263157894736836 | 0.34782608695652173 | 8 | 0 | False |
| fedavg_evidence - isolated | per_institution | 0.75 | 0 | total_10 | 8 | 0.241353610475464 | 0.15362318840579714 | -0.052631578947368474 | 0.7368421052631577 | 7 | 1 | False |
| fedavg_evidence - isolated | per_institution | 0.5 | 0 | f1 | 8 | 0.13385780885780885 | 0.12499999999999994 | -0.045454545454545414 | 0.3500000000000001 | 6 | 1 | False |
| fedavg_evidence - isolated | per_institution | 0.5 | 0 | total_10 | 8 | 0.04117132867132866 | 0.041958041958041925 | -0.5 | 0.4 | 5 | 2 | False |
| fedavg_evidence - isolated | per_institution | 0.25 | 0 | f1 | 8 | 0.05237347672317688 | 0.0761494252873563 | -0.23999999999999988 | 0.34782608695652173 | 6 | 2 | False |
| fedavg_evidence - isolated | per_institution | 0.25 | 0 | total_10 | 8 | -0.018516679160419804 | -0.020000000000000073 | -0.3333333333333334 | 0.31034482758620696 | 2 | 4 | True |
| fedavg_evidence - isolated | union | 1.0 | 0 | f1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| fedavg_evidence - isolated | union | 1.0 | 0 | total_10 | 1 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 0.040000000000000036 | 1 | 0 | False |
| fedavg_evidence - isolated | union | 0.75 | 0 | f1 | 1 | 0.05999999999999983 | 0.05999999999999983 | 0.05999999999999983 | 0.05999999999999983 | 1 | 0 | False |
| fedavg_evidence - isolated | union | 0.75 | 0 | total_10 | 1 | 0.07999999999999996 | 0.07999999999999996 | 0.07999999999999996 | 0.07999999999999996 | 1 | 0 | False |
| fedavg_evidence - isolated | union | 0.5 | 0 | f1 | 1 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 0.019999999999999907 | 1 | 0 | True |
| fedavg_evidence - isolated | union | 0.5 | 0 | total_10 | 1 | -0.02000000000000013 | -0.02000000000000013 | -0.02000000000000013 | -0.02000000000000013 | 0 | 1 | True |
| fedavg_evidence - isolated | union | 0.25 | 0 | f1 | 1 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | -0.019999999999999907 | 0 | 1 | True |
| fedavg_evidence - isolated | union | 0.25 | 0 | total_10 | 1 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | -0.020000000000000018 | 0 | 1 | True |

## Contrasts that were skipped

```json
[]
```

## Centralized reproduction check

```json
{
  "per_seed": [
    {
      "model_seed": 0,
      "measured": {
        "campaign_recall": 0.8035714285714286,
        "relative": 0.9,
        "oracle_recall": 0.8928571428571429,
        "tau": 0.8
      },
      "phase3_recorded": {
        "campaign_recall": 0.6964285714285714,
        "relative": 0.78,
        "n_detected": 39
      },
      "recall_matches": false,
      "ceiling_matches": true
    }
  ],
  "measured_mean_recall": 0.8035714285714286,
  "phase3_mean_over_these_seeds": {
    "model_seeds": [
      0
    ],
    "campaign_recall": 0.6964285714285714,
    "relative": 0.78,
    "oracle_recall": 0.8928571428571429
  },
  "phase3_headline": {
    "campaign_recall": 0.6785714285714286,
    "relative": 0.76
  },
  "headline_is_five_seed_mean": true
}
```

## Timing

| Step | Seconds |
|---|---:|
| build pipeline | 1.3 |
| t100_s0 train isolated | 10.2 |
| t100_s0 isolated | 173.6 |
| t100_s0 fedavg | 208.6 |
| t100_s0 train evidence | 28.3 |
| t100_s0 evidence | 100.0 |
| t100_s0 fedavg_evidence | 137.8 |
| t075_s0 train isolated | 24.4 |
| t075_s0 isolated | 151.1 |
| t075_s0 fedavg | 164.7 |
| t075_s0 train evidence | 27.2 |
| t075_s0 evidence | 155.6 |
| t075_s0 fedavg_evidence | 162.1 |
| t050_s0 train isolated | 16.6 |
| t050_s0 isolated | 112.7 |
| t050_s0 fedavg | 166.5 |
| t050_s0 train evidence | 27.1 |
| t050_s0 evidence | 70.4 |
| t050_s0 fedavg_evidence | 150.3 |
| t025_s0 train isolated | 24.8 |
| t025_s0 isolated | 137.2 |
| t025_s0 fedavg | 158.2 |
| t025_s0 train evidence | 28.1 |
| t025_s0 evidence | 162.9 |
| t025_s0 fedavg_evidence | 158.3 |
| train centralized s0 | 7.4 |
| evaluate centralized | 24.3 |

**Total 2608.6s.**
