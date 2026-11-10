# Stage 3.4.0: GPU environment check

No model and no data - this records what the notebook is, so stage
3.4's design is sized against measurements rather than guesses.

## Environment

- Python 3.13.16, 4 CPUs
- System RAM: 33658 MB total, 32081 MB available
- torch 2.11.0+cu130, CUDA 13.0, cuDNN 92700
- torch_geometric 2.8.1

## GPU

| # | Name | Total MB | Free MB | Capability | SMs |
|---|---|---:|---:|---|---:|
| 0 | Tesla T4 | 15,636 | 15,526 | 7.5 | 40 |
| 1 | Tesla T4 | 15,636 | 15,526 | 7.5 | 40 |

## PyG extensions

None is required - GINEConv and neighbour sampling work without
them; they are faster when present.

| Package | Importable | Version / error |
|---|---|---|
| `pyg-lib` | yes | 0.9.0+pt211cu130 |
| `torch-scatter` | yes | 2.1.2+pt211cu130 |
| `torch-sparse` | yes | 0.6.18+pt211cu130 |
| `torch-cluster` | yes | 1.6.3+pt211cu130 |

## GINEConv smoke test

3 layers, hidden 64, edge_dim 54+1, node dim 6, on cuda. Every edge
is also inserted reversed with a direction flag, so the graph has 2E
edges - the same construction stage 3.4 will use.

| Edges | Nodes | fwd+bwd s | Peak GPU MB | Result |
|---:|---:|---:|---:|---|
| 50,000 | 25,000 | 0.934 | 251.5 | ok |
| 200,000 | 100,000 | 0.067 | 935.6 | ok |
| 500,000 | 250,000 | 0.164 | 2302.0 | ok |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| install torch_geometric | 3.2 | 712 MB | yes |
| install pyg-lib | 2.4 | 984 MB | yes |
| install torch-scatter | 2.4 | 984 MB | yes |
| install torch-sparse | 2.4 | 984 MB | yes |
| install torch-cluster | 2.3 | 984 MB | yes |
