# Uncertainty Estimation Under Distribution Shift
**Testing uncertainty methods on noisy ECG reconstruction**

For updated results, check the drive link below:
https://drive.google.com/drive/folders/1795Dk-KgEdn-SWiq1Jd4GIXXrTTRHtCp?usp=sharing

---

## Overview

Deep learning models can be overconfident, especially in medical applications. A model might reconstruct a corrupted ECG with high confidence even when the result is completely wrong. This project tests whether modern uncertainty estimation methods can detect their own failures when data gets significantly worse than training conditions.

**The core question**: Do uncertainty estimates remain meaningful under severe distribution shift?

**The answer**: It depends heavily on which type of uncertainty you're measuring, and no single method handles everything well.

---

## Why This Project Exists

Standard ML workflows output predictions without any confidence measure. In medical settings, this is dangerous - you need to know which predictions to trust and which to double-check.

I used ECG denoising as a controlled testbed because:
- Real biomedical data (MIT-BIH Arrhythmia Database)
- 1D signals allow fast experimentation
- Results transfer to medical imaging (CT, MRI, photoacoustic imaging)
- I can precisely control noise levels to stress-test methods

The goal wasn't building the best denoiser - it was understanding how uncertainty behaves when things break.

---

## Experimental Setup

**Data**: MIT-BIH ECG signals, windowed into 256-timestep segments
- ~64,000 training samples
- ~21,000 validation samples  
- ~21,000 test samples

**Two evaluation scenarios:**

| Scenario | Noise (σ) | Masking | Purpose |
|----------|-----------|---------|---------|
| **Standard** | 0.1 | 10% | In-distribution testing |
| **Extreme** | 0.5 | 30% | Out-of-distribution stress test |

**Critical design choice**: All models trained only on Standard noise (σ=0.1), then tested on both scenarios. This simulates real deployment where training data is cleaner than production conditions.

**Why these numbers?** The baseline deterministic model achieves:
- Standard: MSE = 0.0012 (handles it fine)
- Extreme: MSE = 0.0469 (complete breakdown)

That 40× error increase is extreme by design - I wanted to see methods fail to understand *how* they fail.

---

## Methods Tested

### 1. Deterministic Baseline (No Uncertainty)

Standard convolutional autoencoder with MSE loss. Gives predictions but no confidence information.

**Result**: Works well in-distribution, fails on extreme noise with no way to detect which predictions are wrong.

---

### 2. MC Dropout (Epistemic Uncertainty)

Keep dropout (p=0.2) enabled at test time, run 20 forward passes, use variance as uncertainty.

**Standard noise results:**
- MSE: 0.0012
- Mean uncertainty: 0.000511
- Spearman correlation: 0.1077
- Rejection gain: +4.3%

**Extreme noise results:**
- MSE: 0.0391 (16% better than baseline!)
- Mean uncertainty: 0.002078 (4× higher - detects the shift)
- Spearman correlation: 0.3017 (best of all methods)
- Rejection gain: -1.3% (global inflation)

**What this means**: MC Dropout is the best method for ranking which predictions are unreliable. The uncertainty correctly inflates under distribution shift, and high uncertainty predictions actually correspond to high errors.

**The limitation**: Absolute uncertainty values aren't calibrated - you can rank predictions but can't make statements like "95% confident this is within ±X."

**Why it works**: Different dropout masks produce genuinely different outputs when the model is confused. That disagreement is informative.

---

### 3. Total Variation Regularization (Physics Constraints)

Added smoothness penalty (λ=0.01) to encourage physiologically plausible reconstructions.

**Extreme noise results:**
- MSE: 0.0395
- Spearman correlation: 0.1582 (dropped from 0.30!)
- Rejection gain: -7.4% (much worse than unconstrained)

**Localization test** (does uncertainty point to actual errors?):
- Random baseline: 0.100
- MC Dropout: 0.127 (good localization)
- MC + TV: 0.123 (degraded localization)

**What went wrong**: The smoothness prior made reconstructions visually cleaner, but the model became "smoothly wrong." It confidently output plausible smooth signals that didn't match reality. Uncertainty spread across entire signals instead of localizing to problem areas.

**Key lesson**: Physics-based constraints help under mild conditions but can induce dangerous overconfidence under severe noise. The model trusts its prior more than the corrupted data.

---

### 4. Deep Ensembles (Gold Standard Epistemic)

5 independently trained models with different random seeds. Measure disagreement between models.

**Training results (validation MSE):**
- Member 1: 0.001183
- Member 2: 0.001200
- Member 3: 0.001224
- Member 4: 0.001198
- Member 5: 0.001230

All converged to similar performance (good sign of stable training).

**Extreme noise testing:**
- MSE: 0.0396 (competitive accuracy)
- Mean uncertainty: 0.002800
- Spearman correlation: 0.0950 (terrible)

**Direct comparison with MC Dropout:**

| Metric | MC Dropout | Ensemble |
|--------|------------|----------|
| MSE | 0.0391 | 0.0396 |
| Uncertainty | 0.00207 | 0.00280 |
| Spearman | 0.3017 | 0.0950 |

**Why ensembles failed**: All 5 models learned similar representations and made similar mistakes. They confidently agreed with each other but were collectively wrong.

**The insight**: Ensembles measure *model* uncertainty - disagreement about what the answer should be. When error is dominated by *data* noise (not model ambiguity), ensembles don't help. The models converge on a reasonable-but-wrong interpretation.

---

### 5. Aleatoric Uncertainty (Heteroscedastic NLL)

Model predicts both mean and variance, trained with Gaussian negative log-likelihood loss.

**Training progression (25 epochs):**
```
Epoch 1:  Train NLL -0.40  | Val NLL -0.62
Epoch 5:  Train NLL -1.92  | Val NLL -2.15
Epoch 10: Train NLL -2.69  | Val NLL -2.73
Epoch 20: Train NLL -2.80  | Val NLL -2.87
Epoch 25: Train NLL -2.82  | Val NLL -2.88
```

**Standard noise:**
- MSE: 0.00120
- Spearman correlation: 0.1920

**Extreme noise:**
- MSE: 0.0368 (best reconstruction accuracy!)
- Spearman correlation: 0.0948 (poor error ranking)

**The contradiction**: Best accuracy, worst uncertainty ranking.

**What's happening**: The NLL loss lets the model learn "there's this much noise in the data" and optimize accordingly, leading to better reconstructions. But the learned variance is calibrated to training noise (σ=0.1). When test noise jumps to σ=0.5, the model doesn't know to scale uncertainty proportionally.

Result: Accurate predictions with confidence estimates stuck at training-time levels. The uncertainty doesn't adapt to new conditions.

---

### 6. Conformal Prediction (Statistical Guarantees)

Wrapped MC Dropout with inductive conformal prediction. The promise: mathematically guaranteed coverage regardless of model quality.

**Setup:**
- Calibrated on Standard validation set
- Target: 90% coverage (α=0.1)
- Computed threshold q̂ = 2.7389

**Standard test (in-distribution):**
- Coverage: 89.99%
- Status: ✅ Perfect - matches theory exactly

**Extreme test (out-of-distribution):**
- Coverage: 30.12%
- Target: 90%
- Gap: 59.88%
- Status: ❌ Catastrophic failure

**Why this is the key finding**: Conformal prediction provides mathematical guarantees, but those guarantees assume *exchangeability* - that calibration and test data come from the same distribution.

Under severe distribution shift:
- Error magnitude increases drastically
- But prediction intervals don't expand proportionally
- The fundamental assumption is violated
- Coverage collapses

**Real-world relevance**: You calibrate on clean/controlled data, deploy in messy real conditions. This is exactly where you need guarantees most - and exactly where they break.

**Sanity check**: The perfect 90% coverage on Standard proves implementation is correct. The failure on Extreme is a real limitation, not a bug.

---

## Summary: What Actually Works?

No single method solved everything:

| Method | MSE | Spearman | Best For | Fails When |
|--------|-----|----------|----------|------------|
| **Baseline** | 0.0469 | N/A | Speed | Need safety info |
| **MC Dropout** | 0.0391 | 0.3017 | Ranking failures | Need calibrated values |
| **MC + TV** | 0.0395 | 0.1582 | Visual quality | Heavy noise (overconfident) |
| **Ensembles** | 0.0396 | 0.0950 | Model uncertainty | Data-dominated noise |
| **Aleatoric** | 0.0368 | 0.0948 | Raw accuracy | Distribution shift |
| **Conformal** | N/A | N/A | In-dist guarantees | Out-of-distribution |

**Key takeaways:**
1. **MC Dropout** was most reliable for detecting failures (Spearman 0.30)
2. **Aleatoric NLL** gave best reconstructions but worst OOD uncertainty
3. **Ensembles** failed because models agreed but were jointly wrong
4. **Conformal guarantees** break under shift (30% coverage vs 90% target)
5. **Physics constraints** can induce overconfidence

---

## Why These Results Matter

Most papers test uncertainty on MNIST or mild noise. Few stress-test under realistic distribution shift.

This work shows:
- Methods that work in-distribution can catastrophically fail OOD
- Statistical guarantees are conditional on assumptions
- Different uncertainty types answer different questions
- No free lunch - you need to match method to failure mode

Directly relevant to:
- Medical imaging deployment
- Sensor degradation scenarios
- Any safety-critical application where train ≠ deployment

---

## Repository Structure

```
.
├── data/                   # MIT-BIH ECG data
├── models/                 
│   ├── backbone.py         # Standard autoencoder
│   ├── backbone_dropout.py # MC Dropout variant
│   ├── aleatoric.py        # Heteroscedastic head
│   └── constraints.py      # TV regularization
├── notebooks/              # Numbered experiments
│   ├── 01_data_prep.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_uncertainty.ipynb      # MC Dropout
│   ├── 04_constraints.ipynb      # TV regularization
│   ├── 05_ensembles.ipynb        # Deep Ensembles
│   ├── 06_aleatoric.ipynb        # NLL training
│   └── 07_conformal.ipynb        # Coverage tests
├── results/
│   ├── figures/           # Rejection curves, uncertainty maps
│   ├── logs/              # JSON metrics
│   └── models/            # Trained weights
└── utils/
    ├── metrics.py         # Spearman, ECE, localization
    └── data_loader.py     # Noise injection pipeline
```

## Current Status

✅ All experiments complete  
✅ Results validated across multiple runs  
✅ Negative findings documented (conformal failure is key contribution)  
✅ Ready for presentation/discussion

---


## Notes

This project used AI assistance (Gemini, Claude, GPT) for coding and documentation. All experimental design, hypothesis formulation, and scientific interpretation are my own work.

The emphasis throughout has been on understanding behavior and failure modes, not just achieving benchmark numbers.

---

**Contact**: eishaankhatri@gmail.com  
**Keywords**: Uncertainty Quantification, Distribution Shift, Conformal Prediction, Medical AI, ECG Denoising

---

## What This Project Is

- Systematic empirical study of uncertainty under stress
- Honest documentation of what works and what doesn't  
- Foundation for research in trustworthy medical AI

## What This Project Isn't

- State-of-the-art benchmark competition
- Production clinical system
- Claim that one method is universally best
