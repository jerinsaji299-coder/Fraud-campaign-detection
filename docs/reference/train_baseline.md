# XGBoost baseline: training

The non-graph control. Same windows, same causal bounds and the
same features as the GNN will see; each transaction is an
independent row. No threshold is chosen here.

Two variants, identical in every respect except the columns the
second drops: `xgboost_centralized` and
`xgboost_centralized_no_format`, the latter without the payment
format one-hots. In this synthetic data laundering is strongly
linked to format, which is an allowed input but a shortcut a
model can take instead of learning transactional structure, so
every model is reported with and without it.

## Hyperparameters

| parameter | value |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 1.0 |
| reg_lambda | 1.0 |
| tree_method | hist |
| max_bin | 256 |
| seed | 0 |

## Models

| variant | lookback_h | seed | model_dir | n_features | n_windows | detection_times | n_rows | n_positive | positive_rate | matrix_memory_mb | xgboost_version | n_dropped_columns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost_centralized | 24 | 0 | /kaggle/working/models/xgboost_centralized_L24_s0 | 66 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1852.1 | 3.4.1 | 0 |
| xgboost_centralized | 24 | 1 | /kaggle/working/models/xgboost_centralized_L24_s1 | 66 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1852.1 | 3.4.1 | 0 |
| xgboost_centralized | 24 | 2 | /kaggle/working/models/xgboost_centralized_L24_s2 | 66 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1852.1 | 3.4.1 | 0 |
| xgboost_centralized | 24 | 3 | /kaggle/working/models/xgboost_centralized_L24_s3 | 66 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1852.1 | 3.4.1 | 0 |
| xgboost_centralized | 24 | 4 | /kaggle/working/models/xgboost_centralized_L24_s4 | 66 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1852.1 | 3.4.1 | 0 |
| xgboost_centralized_no_format | 24 | 0 | /kaggle/working/models/xgboost_centralized_no_format_L24_s0 | 58 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1637.4 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 24 | 1 | /kaggle/working/models/xgboost_centralized_no_format_L24_s1 | 58 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1637.4 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 24 | 2 | /kaggle/working/models/xgboost_centralized_no_format_L24_s2 | 58 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1637.4 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 24 | 3 | /kaggle/working/models/xgboost_centralized_no_format_L24_s3 | 58 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1637.4 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 24 | 4 | /kaggle/working/models/xgboost_centralized_no_format_L24_s4 | 58 | 13 | ['2022-09-02 00:00:00', '2022-09-02 06:00:00', '2022-09-02 12:00:00', '2022-09-02 18:00:00', '2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 6710488 | 5033 | 0.00075 | 1637.4 | 3.4.1 | 8 |
| xgboost_centralized | 48 | 0 | /kaggle/working/models/xgboost_centralized_L48_s0 | 66 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2514.7 | 3.4.1 | 0 |
| xgboost_centralized | 48 | 1 | /kaggle/working/models/xgboost_centralized_L48_s1 | 66 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2514.7 | 3.4.1 | 0 |
| xgboost_centralized | 48 | 2 | /kaggle/working/models/xgboost_centralized_L48_s2 | 66 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2514.7 | 3.4.1 | 0 |
| xgboost_centralized | 48 | 3 | /kaggle/working/models/xgboost_centralized_L48_s3 | 66 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2514.7 | 3.4.1 | 0 |
| xgboost_centralized | 48 | 4 | /kaggle/working/models/xgboost_centralized_L48_s4 | 66 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2514.7 | 3.4.1 | 0 |
| xgboost_centralized_no_format | 48 | 0 | /kaggle/working/models/xgboost_centralized_no_format_L48_s0 | 58 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2223.2 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 48 | 1 | /kaggle/working/models/xgboost_centralized_no_format_L48_s1 | 58 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2223.2 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 48 | 2 | /kaggle/working/models/xgboost_centralized_no_format_L48_s2 | 58 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2223.2 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 48 | 3 | /kaggle/working/models/xgboost_centralized_no_format_L48_s3 | 58 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2223.2 | 3.4.1 | 8 |
| xgboost_centralized_no_format | 48 | 4 | /kaggle/working/models/xgboost_centralized_no_format_L48_s4 | 58 | 9 | ['2022-09-03 00:00:00', '2022-09-03 06:00:00', '2022-09-03 12:00:00', '2022-09-03 18:00:00', '2022-09-04 00:00:00', '2022-09-04 06:00:00', '2022-09-04 12:00:00', '2022-09-04 18:00:00', '2022-09-05 00:00:00'] | 9111385 | 7070 | 0.000776 | 2223.2 | 3.4.1 | 8 |

## Training windows

| lookback_h | t | n_edges | n_positive | positive_rate | class_weight |
|---|---|---|---|---|---|
| 24 | 2022-09-02 00:00:00 | 1114921 | 322 | 0.000289 | 3461.49 |
| 24 | 2022-09-02 06:00:00 | 803684 | 323 | 0.000402 | 2487.19 |
| 24 | 2022-09-02 12:00:00 | 787448 | 341 | 0.000433 | 2308.23 |
| 24 | 2022-09-02 18:00:00 | 772119 | 395 | 0.000512 | 1953.73 |
| 24 | 2022-09-03 00:00:00 | 754449 | 408 | 0.000541 | 1848.14 |
| 24 | 2022-09-03 06:00:00 | 616514 | 412 | 0.000668 | 1495.39 |
| 24 | 2022-09-03 12:00:00 | 479911 | 423 | 0.000881 | 1133.54 |
| 24 | 2022-09-03 18:00:00 | 342641 | 403 | 0.001176 | 849.23 |
| 24 | 2022-09-04 00:00:00 | 207382 | 391 | 0.001885 | 529.39 |
| 24 | 2022-09-04 06:00:00 | 208240 | 394 | 0.001892 | 527.53 |
| 24 | 2022-09-04 12:00:00 | 207655 | 395 | 0.001902 | 524.71 |
| 24 | 2022-09-04 18:00:00 | 208094 | 419 | 0.002014 | 495.64 |
| 24 | 2022-09-05 00:00:00 | 207430 | 407 | 0.001962 | 508.66 |
| 48 | 2022-09-03 00:00:00 | 1869370 | 730 | 0.000391 | 2559.78 |
| 48 | 2022-09-03 06:00:00 | 1420198 | 735 | 0.000518 | 1931.24 |
| 48 | 2022-09-03 12:00:00 | 1267359 | 764 | 0.000603 | 1657.85 |
| 48 | 2022-09-03 18:00:00 | 1114760 | 798 | 0.000716 | 1395.94 |
| 48 | 2022-09-04 00:00:00 | 961831 | 799 | 0.000831 | 1202.79 |
| 48 | 2022-09-04 06:00:00 | 824754 | 806 | 0.000977 | 1022.27 |
| 48 | 2022-09-04 12:00:00 | 687566 | 818 | 0.00119 | 839.55 |
| 48 | 2022-09-04 18:00:00 | 550735 | 822 | 0.001493 | 668.99 |
| 48 | 2022-09-05 00:00:00 | 414812 | 798 | 0.001924 | 518.81 |

## Feature importance (gain, one seed, report only)

Never used to select features: that would make the variants'
feature sets differ by something other than the ablation.

| variant | lookback_h | seed | feature | gain | share |
|---|---|---|---|---|---|
| xgboost_centralized | 24 | 0 | fmt_ACH | 75106.88 | 0.2849 |
| xgboost_centralized | 24 | 0 | self_transfer | 22392.22 | 0.085 |
| xgboost_centralized | 24 | 0 | same_currency | 19261.58 | 0.0731 |
| xgboost_centralized | 24 | 0 | src_out_degree | 13122.51 | 0.0498 |
| xgboost_centralized | 24 | 0 | fmt_Credit Card | 11656.78 | 0.0442 |
| xgboost_centralized | 24 | 0 | fmt_Wire | 10009.79 | 0.038 |
| xgboost_centralized | 24 | 0 | fmt_Cheque | 7498.96 | 0.0285 |
| xgboost_centralized | 24 | 0 | src_n_counterparties | 7006.34 | 0.0266 |
| xgboost_centralized | 24 | 0 | recv_cur_Bitcoin | 5250.58 | 0.0199 |
| xgboost_centralized | 24 | 0 | src_n_currencies | 4605.08 | 0.0175 |
| xgboost_centralized | 24 | 0 | pay_cur_Bitcoin | 3927.44 | 0.0149 |
| xgboost_centralized | 24 | 0 | pay_cur_Saudi Riyal | 3404.39 | 0.0129 |
| xgboost_centralized | 24 | 0 | log_amount_received | 3312.15 | 0.0126 |
| xgboost_centralized | 24 | 0 | recv_cur_Saudi Riyal | 3288.3 | 0.0125 |
| xgboost_centralized | 24 | 0 | pay_cur_Brazil Real | 3259.49 | 0.0124 |
| xgboost_centralized | 24 | 0 | fmt_Cash | 3132.1 | 0.0119 |
| xgboost_centralized | 24 | 0 | recv_cur_Brazil Real | 2902.59 | 0.011 |
| xgboost_centralized | 24 | 0 | dst_n_counterparties | 2704.44 | 0.0103 |
| xgboost_centralized | 24 | 0 | dst_in_degree | 2679.33 | 0.0102 |
| xgboost_centralized | 24 | 0 | cross_bank | 2541.96 | 0.0096 |
| xgboost_centralized_no_format | 24 | 0 | self_transfer | 36406.14 | 0.1444 |
| xgboost_centralized_no_format | 24 | 0 | same_currency | 27290.56 | 0.1082 |
| xgboost_centralized_no_format | 24 | 0 | cross_bank | 27272.59 | 0.1081 |
| xgboost_centralized_no_format | 24 | 0 | src_n_currencies | 14145.69 | 0.0561 |
| xgboost_centralized_no_format | 24 | 0 | src_out_degree | 9315.66 | 0.0369 |
| xgboost_centralized_no_format | 24 | 0 | pay_cur_Saudi Riyal | 8800.49 | 0.0349 |
| xgboost_centralized_no_format | 24 | 0 | pay_cur_Bitcoin | 8103.28 | 0.0321 |
| xgboost_centralized_no_format | 24 | 0 | dst_n_counterparties | 6488.2 | 0.0257 |
| xgboost_centralized_no_format | 24 | 0 | dst_in_degree | 5079.87 | 0.0201 |
| xgboost_centralized_no_format | 24 | 0 | log_amount_received | 4617.35 | 0.0183 |
| xgboost_centralized_no_format | 24 | 0 | recv_cur_Saudi Riyal | 4346.05 | 0.0172 |
| xgboost_centralized_no_format | 24 | 0 | recv_cur_Bitcoin | 4265.73 | 0.0169 |
| xgboost_centralized_no_format | 24 | 0 | src_n_counterparties | 4041.0 | 0.016 |
| xgboost_centralized_no_format | 24 | 0 | log_amount_paid | 3879.85 | 0.0154 |
| xgboost_centralized_no_format | 24 | 0 | recv_cur_Yen | 3622.9 | 0.0144 |
| xgboost_centralized_no_format | 24 | 0 | pay_cur_Rupee | 3261.78 | 0.0129 |
| xgboost_centralized_no_format | 24 | 0 | recv_cur_Shekel | 3134.26 | 0.0124 |
| xgboost_centralized_no_format | 24 | 0 | dst_out_degree | 3032.76 | 0.012 |
| xgboost_centralized_no_format | 24 | 0 | recv_cur_Ruble | 2928.4 | 0.0116 |
| xgboost_centralized_no_format | 24 | 0 | pay_cur_Ruble | 2906.38 | 0.0115 |
| xgboost_centralized | 48 | 0 | fmt_ACH | 122570.98 | 0.3258 |
| xgboost_centralized | 48 | 0 | same_currency | 36664.09 | 0.0974 |
| xgboost_centralized | 48 | 0 | self_transfer | 19873.39 | 0.0528 |
| xgboost_centralized | 48 | 0 | src_out_degree | 16410.17 | 0.0436 |
| xgboost_centralized | 48 | 0 | fmt_Credit Card | 14193.63 | 0.0377 |
| xgboost_centralized | 48 | 0 | fmt_Bitcoin | 11632.73 | 0.0309 |
| xgboost_centralized | 48 | 0 | fmt_Cheque | 11168.08 | 0.0297 |
| xgboost_centralized | 48 | 0 | recv_cur_Bitcoin | 10589.21 | 0.0281 |
| xgboost_centralized | 48 | 0 | src_n_counterparties | 9148.62 | 0.0243 |
| xgboost_centralized | 48 | 0 | src_n_currencies | 6969.81 | 0.0185 |
| xgboost_centralized | 48 | 0 | fmt_Cash | 6665.81 | 0.0177 |
| xgboost_centralized | 48 | 0 | pay_cur_Saudi Riyal | 6625.61 | 0.0176 |
| xgboost_centralized | 48 | 0 | recv_cur_Brazil Real | 5863.92 | 0.0156 |
| xgboost_centralized | 48 | 0 | src_in_degree | 4767.99 | 0.0127 |
| xgboost_centralized | 48 | 0 | log_amount_received | 4307.36 | 0.0114 |
| xgboost_centralized | 48 | 0 | dst_n_counterparties | 3418.45 | 0.0091 |
| xgboost_centralized | 48 | 0 | pay_cur_Bitcoin | 3396.97 | 0.009 |
| xgboost_centralized | 48 | 0 | dst_in_degree | 3125.21 | 0.0083 |
| xgboost_centralized | 48 | 0 | fmt_Wire | 2954.55 | 0.0079 |
| xgboost_centralized | 48 | 0 | cross_bank | 2853.51 | 0.0076 |
| xgboost_centralized_no_format | 48 | 0 | cross_bank | 34978.99 | 0.1107 |
| xgboost_centralized_no_format | 48 | 0 | self_transfer | 33752.02 | 0.1068 |
| xgboost_centralized_no_format | 48 | 0 | same_currency | 26449.42 | 0.0837 |
| xgboost_centralized_no_format | 48 | 0 | src_n_currencies | 22947.93 | 0.0726 |
| xgboost_centralized_no_format | 48 | 0 | pay_cur_Saudi Riyal | 16503.22 | 0.0522 |
| xgboost_centralized_no_format | 48 | 0 | pay_cur_Bitcoin | 12669.67 | 0.0401 |
| xgboost_centralized_no_format | 48 | 0 | src_n_counterparties | 10293.8 | 0.0326 |
| xgboost_centralized_no_format | 48 | 0 | src_out_degree | 8946.79 | 0.0283 |
| xgboost_centralized_no_format | 48 | 0 | dst_n_counterparties | 8399.71 | 0.0266 |
| xgboost_centralized_no_format | 48 | 0 | src_in_degree | 8059.06 | 0.0255 |
| xgboost_centralized_no_format | 48 | 0 | dst_in_degree | 6007.49 | 0.019 |
| xgboost_centralized_no_format | 48 | 0 | log_amount_paid | 5737.21 | 0.0182 |
| xgboost_centralized_no_format | 48 | 0 | log_amount_received | 5206.47 | 0.0165 |
| xgboost_centralized_no_format | 48 | 0 | dow_4 | 5053.78 | 0.016 |
| xgboost_centralized_no_format | 48 | 0 | recv_cur_Saudi Riyal | 4540.01 | 0.0144 |
| xgboost_centralized_no_format | 48 | 0 | recv_cur_Rupee | 4357.12 | 0.0138 |
| xgboost_centralized_no_format | 48 | 0 | dst_log_out_amount | 3924.22 | 0.0124 |
| xgboost_centralized_no_format | 48 | 0 | dst_n_currencies | 3862.85 | 0.0122 |
| xgboost_centralized_no_format | 48 | 0 | dow_5 | 3854.11 | 0.0122 |
| xgboost_centralized_no_format | 48 | 0 | dst_log_in_amount | 3841.12 | 0.0122 |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 35.7 | 3,165 MB | yes |
| build training matrix, L=24h | 87.0 | 9,888 MB | yes |
| train xgboost_centralized, L=24h, seed=0 | 183.2 | 9,888 MB | yes |
| train xgboost_centralized, L=24h, seed=1 | 180.7 | 9,888 MB | yes |
| train xgboost_centralized, L=24h, seed=2 | 175.6 | 9,888 MB | yes |
| train xgboost_centralized, L=24h, seed=3 | 176.6 | 9,888 MB | yes |
| train xgboost_centralized, L=24h, seed=4 | 175.0 | 9,888 MB | yes |
| train xgboost_centralized_no_format, L=24h, seed=0 | 249.9 | 9,888 MB | yes |
| train xgboost_centralized_no_format, L=24h, seed=1 | 162.9 | 9,888 MB | yes |
| train xgboost_centralized_no_format, L=24h, seed=2 | 170.8 | 9,888 MB | yes |
| train xgboost_centralized_no_format, L=24h, seed=3 | 170.8 | 9,888 MB | yes |
| train xgboost_centralized_no_format, L=24h, seed=4 | 164.3 | 9,888 MB | yes |
| build training matrix, L=48h | 114.0 | 15,907 MB | yes |
| train xgboost_centralized, L=48h, seed=0 | 234.5 | 15,907 MB | yes |
| train xgboost_centralized, L=48h, seed=1 | 235.9 | 15,907 MB | yes |
| train xgboost_centralized, L=48h, seed=2 | 235.2 | 15,907 MB | yes |
| train xgboost_centralized, L=48h, seed=3 | 356.0 | 15,907 MB | yes |
| train xgboost_centralized, L=48h, seed=4 | 244.6 | 15,907 MB | yes |
| train xgboost_centralized_no_format, L=48h, seed=0 | 220.9 | 15,907 MB | yes |
| train xgboost_centralized_no_format, L=48h, seed=1 | 227.1 | 15,907 MB | yes |
| train xgboost_centralized_no_format, L=48h, seed=2 | 224.7 | 15,907 MB | yes |
| train xgboost_centralized_no_format, L=48h, seed=3 | 223.9 | 15,907 MB | yes |
| train xgboost_centralized_no_format, L=48h, seed=4 | 223.3 | 15,907 MB | yes |
