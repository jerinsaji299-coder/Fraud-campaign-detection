# Stage 3.4: GNN training

Full-batch, one window per step, GINE 3x64 with layer
normalisation and an edge readout MLP. One frozen
hyperparameter configuration, never tuned on validation or
test. Early stopping on validation edge-level average
precision over the validation **detection** split.

## Models

| variant | lookback_h | seed | model_dir | epochs_run | best_epoch | best_val_average_precision | final_train_loss | max_abs_logit | peak_reserved_mb | cached_graph_memory_mb | deterministic | n_training_windows | n_validation_windows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn_centralized | 24 | 0 | /kaggle/working/models/gnn/gnn_centralized_L24_s0 | 56 | 45 | 0.418922 | 0.140908 | 14.14 | 7432.3 | 5924.0 | True | 13 | 4 |

## Timing and memory

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 30.6 | 3,880 MB | yes |
| fit feature spec + node preprocessor, L=24h | 54.4 | 10,599 MB | yes |
| train gnn_centralized L=24h seed=0 | 463.4 | 10,599 MB | yes |
