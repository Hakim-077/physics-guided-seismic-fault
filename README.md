<<<<<<< HEAD
# Physics-Guided Deep Learning for Seismic Fault Detection

Code for *"Physics-Guided Deep Learning for Seismic Fault Detection Using
Migrated Marmousi-II Data"* (Arabian Journal of Geosciences).

A U-Net for seismic fault detection with a physics-guided loss term that
penalizes fault predictions aligned with the local reflector dip field.
Compared against a baseline U-Net and an Attention U-Net on Marmousi-II.

## Results

| Method | IoU | F1-score | Precision | Recall |
|---|---|---|---|---|
| Baseline U-Net | 0.523 [0.512, 0.534] | 0.687 [0.677, 0.697] | 0.803 [0.792, 0.814] | 0.600 [0.588, 0.612] |
| Attention U-Net | 0.506 [0.496, 0.517] | 0.672 [0.663, 0.682] | 0.737 [0.726, 0.748] | 0.618 [0.606, 0.630] |
| **Physics-guided U-Net** | **0.688** [0.679, 0.698] | **0.815** [0.809, 0.822] | 0.805 [0.795, 0.814] | **0.826** [0.817, 0.836] |

95% bootstrap CIs (1000 resamples). Full ablation, cross-validation, and
FCI/DRR metrics in the paper.

## Caveats

- Ground truth is gradient-threshold pseudo-labels (top 5% of gradient
  magnitude), not manually interpreted fault picks. See Limitations in
  the paper.
- 2D single inline section, not a full 3D volume — classical coherence
  attributes aren't directly comparable here.
- DRR is computed from a regional dip estimate derived from the same
  gradient field the physics loss uses, so it's not fully independent
  of the training objective. Also discussed in the paper.

## Structure

```
scripts/
  main_pipeline.py                   data loading, pseudo-labels, patches,
                                      baseline/Attention/physics-guided U-Net
                                      training, lambda tuning, 5-fold CV,
                                      ablation with bootstrap CIs
  c3_coherence.py                    eigenstructure (C3) coherence attribute
  evaluate_c3_baseline.py            C3 coherence vs the same test set
  continuity_diffraction_metrics.py  FCI and DRR
  full_section_metrics_and_fig10.py  full-section inference, FCI/DRR, RTM overlay
results/
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Place the Marmousi-II SEG-Y file as `SYNTHETIC.segy` in the working
directory, then:

```bash
python scripts/main_pipeline.py
```

Trains all three models, tunes lambda, runs 5-fold CV and the ablation
study, prints the SOTA comparison table. GPU recommended.

For FCI/DRR and the RTM overlay (run in the same session after the models
are trained, or adapt to load saved weights):

```bash
python scripts/full_section_metrics_and_fig10.py
```

For the C3 coherence baseline:

```bash
python scripts/evaluate_c3_baseline.py SYNTHETIC.segy
```

## Models

- **Baseline U-Net**: 2-level encoder-decoder, skip connections
- **Attention U-Net**: same depth/width as baseline, additive attention
  gates (Oktay et al. 2018) on skip connections
- **Physics-guided U-Net**: baseline architecture + dip-alignment loss
  term, plus sparsity/continuity/contrast terms (Algorithm 1 in the paper)

## Citation

```
Zeghichi, A., Wu, Y. Physics-Guided Deep Learning for Seismic Fault
Detection Using Migrated Marmousi-II Data. Arabian Journal of Geosciences
(under review).
```

## Contact

Abdelhakim Zeghichi — abdelhakimzeghichi@gmail.com
College of Geophysics and Geomatics, Chang'an University, Xi'an, China
=======
# physics-guided-seismic-fault
Code and scripts for physics-guided seismic fault detection on Marmousi-II
>>>>>>> e745eb56a30002de49f6355592142a8552fc012e
