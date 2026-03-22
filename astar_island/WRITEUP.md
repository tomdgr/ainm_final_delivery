# NM i AI 2026 — Astar Island: Technical Writeup

## Team
Tom Daniel Grande, Fridtjof Høyer, Tobias Korten, Henrik Skulevold + AI assistants

## Competition
Predict output distributions of a stochastic Norse settlement simulator on a 40×40 grid. 5 seeds per round, 50 API peek queries, scored by entropy-weighted KL divergence (0-100). Hidden sim parameters change every round (~2.5h). Competition: March 19-22, 2026.

## Results

| Round | Score | Rank | Model | Weight |
|-------|-------|------|-------|--------|
| R1 | 19.3 | #72/117 | Heuristic | 1.05x |
| R2 | 48.9 | #79/153 | Early sim | 1.10x |
| R8 | 85.9 | #43/214 | RF spatial | 1.48x |
| R9 | 86.6 | #61/221 | RF spatial | 1.55x |
| R13 | 87.8 | #48/186 | RF spatial | 1.89x |
| R14 | 82.4 | #31/244 | RF spatial | 1.98x |
| **R16** | **86.2** | **#24/272** | **Ensemble (RF+CNP+ANP)** | 2.18x |
| R17 | pending | - | Ensemble (RF+CNP+ANP+CNP_v5) | 2.29x |

Best rank: **#24 out of 272 teams** (Round 16)

## Approach Evolution

### Phase 1: Simulator (Rounds 1-2)
Built local physics-based simulator (simulator_v2.py) approximating the Norse settlement sim:
- 5 phases: Growth, Conflict, Trade, Winter, Environment
- Logistic growth with carrying capacity, flow-based food model
- Calibrated against ground truth: survival rate 29% (GT 31%), port formation 3% (GT 7.5%)
- **Score: ~45 avg** — structural mismatch limited ceiling

### Phase 2: Random Forest (Rounds 8-14)
RF spatial predictor trained per-round on API observations:
- 16 spatial features: terrain, distances, adjacency counts, transition matrices
- Trained on each round's own observations — no cross-round transfer needed
- Context-specific blending with empirical transition probabilities
- **Score: 83-88 avg** — stable and reliable

### Phase 3: Neural Models (Rounds 16-17)
Added neural models for ensemble diversity:

**ConvCNP (Fridtjof):** U-Net mapping 8-channel observation tensor → 6-class probabilities.
- Episodic meta-learning: random context subsets per training episode
- Data augmentation: 8-fold dihedral symmetry (rotation/flip)
- Unstable: 16-91 range. Peaks brilliantly, crashes catastrophically.
- **Scores: 91.3 peak, 16.1 worst**

**Attentive Neural Process (ANP):** Transformer cross-attention from target cells to observed cells.
- Learns which observations are most relevant for each prediction
- 3-layer cross-attention with 4 heads, d_model=128
- **Score: ~66 avg LOO** — better than CNP but below RF

**ConvCNP v5 (synthetic pre-training):** Pre-train on 500 synthetic episodes from local sim with randomized params, then fine-tune on real data.
- Parallel synthetic data generation across 40 CPU cores
- Domain randomization: sample all 14 sim params from wide priors per episode
- Pre-train 30 epochs → fine-tune 40 epochs with 10x lower LR
- **Hypothesis:** fixes overfitting by providing parameter diversity

### Phase 4: Ensemble (Rounds 16-17)
Geometric mean (log-linear) ensemble with KL-divergence fallback:
- If neural model diverges too much from RF (KL > 0.5), reduce its weight
- R16: RF(50%) + ConvCNP(35%) + ANP(15%) → **86.2, #24/272**
- R17: RF(45%) + ConvCNP(25%) + ANP(15%) + CNP_v5(15%) → pending

## Infrastructure

### GCE Pipeline
One-command deployment: `bash infra/launch-round.sh <N>`
- **g2-standard-48**: 48 vCPUs + 4x NVIDIA L4 GPUs
- Parallel workers: RF on CPUs, ConvCNP on GPU:0, ANP on GPU:1
- Synthetic data generation parallelized across all CPU cores
- Auto-uploads predictions to GCS, self-destructs

### Key Speedups
- ConvCNP: 28s/epoch CPU → 1.6s/epoch GPU (17x)
- Synthetic data: 14 min on 1 core → 3.7 min on 40 cores (4x)
- Full pipeline: ~20 min on g2-standard-48

### Monitoring
- tqdm progress bars
- SSH live monitoring
- Serial port output
- Streamlit dashboard for experiment comparison

## Key Files

```
nm_ai_ml/astar/
  spatial_predictor_rf.py    # RF predictor (baseline, 83-88)
  convcnp.py                 # ConvCNP with GPU, TTA, temperature scaling
  attentive_np.py            # Transformer Attentive NP
  simulator_v2.py            # Physics-based local simulator
  xgb_predictor.py           # XGBoost with 28 enriched features
  synth_data.py              # Parallel synthetic data generation
  param_inference.py         # Per-round parameter inference (DE)

round_pipeline.py            # Parallel model training pipeline (v2)
round_pipeline_v5.py         # V5 with parallel synth + XGBoost
ensemble_offline.py          # Offline ensemble weight optimization
local_ensemble.py            # Quick local ensemble from predictions
dashboard.py                 # Streamlit experiment tracker

infra/
  launch-round.sh            # One-command GCE deployment
  gce-pipeline.sh            # VM startup script
```

## Lessons Learned

1. **RF dominates with small data.** Neural models need 10-100x more training examples to not overfit. With only ~200 real episodes, RF's per-round training is unbeatable.

2. **Ensemble only helps if models are diverse AND reliable.** Adding an unreliable model (ConvCNP LOO=25) to a reliable one (RF=83) makes things worse. The R16 ensemble worked because all three models happened to agree on that round.

3. **Synthetic pre-training is the path to fixing neural models.** Domain randomization with diverse sim parameters provides the training volume neural models need. But the sim must be good enough that synthetic data is useful.

4. **Validate before you submit.** LOO validation is essential. Without it, we'd have submitted ConvCNP-heavy ensembles that score 50 instead of 85.

5. **GPU speedup is massive for iteration speed.** ConvCNP went from 70 min training on CPU to 4 min on L4 GPU. This enabled rapid experimentation.

6. **Later rounds matter more.** Round weights increase exponentially (R17 = 2.29x vs R1 = 1.05x). Improving late-round scores has 2x the leaderboard impact.

## What We'd Do Differently

1. **Start with RF + better features** from day 1, not simulator
2. **XGBoost with sim-derived features** instead of raw RF
3. **More synthetic data** with better sim (8000+ episodes)
4. **Pre-baked GCE images** with CUDA + deps to avoid install time
5. **Automated LOO validation** as gate before every submission
