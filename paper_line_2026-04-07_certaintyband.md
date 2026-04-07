# Paper Line: VGGT Certainty Distillation (2026-04-07)

## Working thesis

The viable direction is not generic `VGGT teacher -> single-view student` distillation.
The viable direction is:

- distill visible certainty strongly
- distill boundary-local hidden support as hard certainty
- treat deep hidden support as ambiguity, not as dense positive supervision

This is the first version in the project that now looks like a paper line instead of a loose ablation thread.

## New ingredients added in this round

### 1. Raw hidden-support packet export

Implemented in:

- `src/teacher3d/vggt_integration.py`
- `src/teacher3d/teacher_packets.py`
- `scripts/precompute_vggt_packets.py`

New packet fields:

- `teacher_hidden_confidence_raw`
- `teacher_hidden_count`
- `teacher_hidden_gap`

### 2. Boundary-local teacher targets

Teacher adapter already derives:

- `hidden_local_support`
- `hidden_deep_support`
- `hidden_interior_negative`

### 3. Ambiguity-band loss

Implemented in:

- `src/teacher3d/losses.py`

New loss term:

- `hidden_ambiguous_band`

Meaning:

- deep hidden regions are no longer forced toward confidence 1
- they are constrained into a moderate-probability band
- this better matches the hidden-ambiguity story

### 4. Boundary-local evaluation

Implemented in:

- `src/teacher3d/eval.py`

New metrics:

- `hidden_local_f1`
- `hidden_local_precision`
- `hidden_local_recall`
- `hidden_on_deep_active`
- `hidden_on_interior_negative_active`

These are important because global hidden F1 alone mixes together certainty and ambiguity.

## Main experiment configs in this round

### A. Sparse local-only raw support

Config:

- `configs/v1_re10k_vggt_decoupledalpha_confmargin_rawsupportlocalonly_largelocal.yaml`

Best result:

- hidden F1: `0.0754`
- hidden local F1: `0.0207`
- hidden-on-visible active: `0.3347`
- visible F1: `0.3554`

Interpretation:

- removing deep-support BCE recovers performance from the failed full raw-support run
- but visible-region bleed remains high

### B. Certainty-band

Config:

- `configs/v1_re10k_vggt_decoupledalpha_confmargin_certaintyband_largelocal.yaml`

Best by global hidden F1:

- hidden F1: `0.0707`
- hidden local F1: `0.0178`
- hidden-on-visible active: `0.4929`
- visible F1: `0.5155`

Best by boundary-local hidden F1:

- hidden local F1: `0.0233`
- hidden F1: `0.0614`
- hidden-on-visible active: `0.0679`
- hidden-on-interior-negative active: `0.0626`
- visible F1: `0.5155`

Interpretation:

- ambiguity-band training makes the model much more conservative near visible support
- this version is useful for the paper story because it separates “boundary-local certainty” from “global hidden recall”

### C. Certainty-band local-plus

Config:

- `configs/v1_re10k_vggt_decoupledalpha_confmargin_certaintyband_localplus_largelocal.yaml`

Best result:

- hidden F1: `0.0798`
- hidden local F1: `0.0281`
- hidden-on-visible active: `0.1681`
- hidden-on-interior-negative active: `0.1604`
- visible F1: `0.2297`
- novel-view L1: `0.1255`

Interpretation:

- this is the best current hidden result under the new certainty-distillation framing
- it improves both global hidden F1 and boundary-local hidden F1 over the reproducible plain baseline
- it also reduces visible bleed slightly relative to baseline
- the tradeoff is lower visible confidence recall at the fixed visible threshold

## Baseline under the same boundary-local metric definition

Baseline checkpoint evaluated against the new raw-support local metrics:

- checkpoint: `outputs/v1_re10k_vggt_decoupledalpha_confmargin_largelocal/model.pt`
- eval file: `outputs/local_ablation/threshold_sweep_decoupledalpha_confmargin_largelocal_evalonrawsupportlocalmetrics_20260407.json`

Best result:

- hidden F1: `0.0753`
- hidden local F1: `0.0237`
- hidden-on-visible active: `0.1809`
- visible F1: `0.3256`

## Why this now looks paper-worthy

The project finally has a structured claim rather than a loose engineering one:

- dense deep hidden positives are harmful
- boundary-local hidden certainty is useful
- deep hidden should be trained as ambiguity, not certainty

And there is now empirical support for that claim:

- full raw-support BCE collapsed
- local-only recovered performance
- certainty-band changed the precision/recall geometry in a meaningful way
- certainty-band local-plus is the first variant to beat the currently reproducible baseline on both hidden F1 and hidden-local F1 while also lowering hidden-on-visible bleed

## Current best candidate method

If the paper had to freeze today, the most defensible method would be:

- teacher: VGGT
- visible supervision: existing visible geometry/confidence supervision
- hidden supervision:
  - boundary-local raw support as positive certainty
  - deep support as ambiguity band
  - interior visible regions as negatives
- evaluation:
  - global hidden F1
  - boundary-local hidden F1
  - hidden-on-visible bleed
  - hidden-on-interior-negative bleed

## Immediate next steps

1. Improve visible confidence calibration for `certaintyband_localplus`
   - current fixed-threshold visible F1 is too low
   - likely a calibration/threshold issue more than a geometry issue

2. Run a small visible-threshold sweep in addition to hidden-threshold sweep
   - current visible F1 is reported at one fixed threshold while hidden is swept
   - this is not ideal for a paper table

3. Add an ablation table with 4 rows only
   - baseline
   - local-only raw support
   - certainty-band
   - certainty-band local-plus

4. Freeze this as the paper direction
   - no teacher switching
   - no more generic confidence hacks
   - all further work should sharpen the certainty-vs-ambiguity framing


## Dual-threshold fair comparison

To remove the visible-threshold confound, I ran a joint visible/hidden threshold sweep for:

- plain baseline re-evaluated on the raw-support packet root
- `certaintyband_localplus`

Saved at:

- `outputs/local_ablation/dual_threshold_sweep_baseline_vs_certaintyband_localplus_20260407.json`

A reusable script is now available at:

- `scripts/dual_threshold_sweep.py`

### Key finding

The earlier visible-F1 drop of `certaintyband_localplus` was mostly a thresholding artifact.

Under the same fair visible threshold (`visible_threshold = 0.20`), both models obtain the same visible F1:

- baseline visible F1: `0.6218`
- certaintyband_localplus visible F1: `0.6218`

At that same visible threshold, `certaintyband_localplus` is better on the hidden metrics that matter for the paper story:

- hidden F1: `0.0798` vs baseline `0.0753`
- hidden local F1: `0.0281` vs baseline `0.0237`
- hidden-on-visible active: `0.1681` vs baseline `0.1809`

Interpretation:

- visible confidence degradation is not the main issue anymore
- once visible threshold is calibrated fairly, `certaintyband_localplus` is strictly better than the reproducible baseline on both global hidden and boundary-local hidden detection
- this is the strongest current evidence for the paper claim

### Updated strongest claim

A defensible current claim is now:

> Distilling boundary-local hidden certainty while treating deep hidden regions as ambiguity yields better single-view hidden reconstruction than a plain VGGT-distilled baseline, at matched visible performance.


## Extra experiments after the fair dual-threshold result

### 1. Visible-calibration training branch

Config:

- `configs/v1_re10k_vggt_decoupledalpha_confmargin_certaintyband_localplus_viscal_largelocal.yaml`

Dual-threshold result (with visible F1 constraint and visible-bleed constraint):

- visible F1: `0.6218`
- hidden F1: `0.0042`
- hidden local F1: `0.0005`
- hidden-on-visible active: `0.0337`

Interpretation:

- training-time visible calibration collapses hidden recall too aggressively
- this branch should not be continued

### 2. Sharper boundary-local support

Config:

- `configs/v1_re10k_vggt_decoupledalpha_confmargin_certaintyband_localplus_boundary35_largelocal.yaml`

Dual-threshold result (with visible F1 constraint and visible-bleed constraint):

- visible F1: `0.6218`
- hidden F1: `0.0455`
- hidden local F1: `0.0142`
- hidden-on-visible active: `0.0784`

Interpretation:

- stronger boundary sparsification suppresses bleed
- but it suppresses true hidden detection too much
- this is over-regularized relative to the current best method

## Updated paper candidate status

Among all runs under the fair dual-threshold comparison:

- baseline: hidden F1 `0.0753`, hidden local F1 `0.0237`, hidden-on-visible active `0.1809`
- certaintyband_localplus: hidden F1 `0.0798`, hidden local F1 `0.0281`, hidden-on-visible active `0.1681`
- certaintyband_localplus_viscal: hidden F1 `0.0042`, hidden local F1 `0.0005`, hidden-on-visible active `0.0337`
- certaintyband_localplus_boundary35: hidden F1 `0.0455`, hidden local F1 `0.0142`, hidden-on-visible active `0.0784`

So the current best paper candidate remains:

- `configs/v1_re10k_vggt_decoupledalpha_confmargin_certaintyband_localplus_largelocal.yaml`

It is still the only variant that improves over the reproducible baseline on both:

- global hidden F1
- boundary-local hidden F1

while also slightly reducing visible-region bleed at matched visible performance.
