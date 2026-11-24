# AI Model Documentation

## Purpose

This document records the model and integration work completed for the ECG project.

The target design is:

- Monitor page: keep the existing PTBDB live binary pipeline.
- Analysis page: use a separate MIT-BIH arrhythmia pipeline.

The final MIT-BIH analysis design is a two-stage pipeline instead of a direct 5-class model.

## Why The Design Changed

The first attempt was a direct practical 5-label MIT-BIH model:

- `N`
- `S`
- `V`
- `F`
- `Q`

This approach did not perform well enough, mainly because the MIT-BIH grouped dataset is highly imbalanced, especially for:

- `F`
- `Q`

The class distribution used from the grouped MIT-BIH cache was:

- Train: `N=36406`, `S=774`, `V=3150`, `F=399`, `Q=8`
- Val: `N=9446`, `S=170`, `V=638`, `F=15`
- Test: `N=44242`, `S=1837`, `V=3220`, `F=388`, `Q=7`

Because `Q` is extremely rare, it was not reliable to learn it directly as a stable class.

## Dataset And Split

The training pipeline used the prepared MIT-BIH grouped cache:

- `AI/models_experiments/cache/mitbih_5class_ds1train_ds1val_ds2_187.npz`

This cache was produced with an inter-patient split workflow from:

- `AI/models_experiments/prepare_mitbih_5class.py`

The final grouped labels in the cache are:

- `N`: Normal-like
- `S`: Supraventricular ectopic
- `V`: Ventricular ectopic
- `F`: Fusion beat
- `Q`: Unknown / paced

All models use beats of length `187`.

## Rejected Attempt

The first direct 5-label experiment was implemented in:

- `AI/models_experiments/train_mitbih_resnet_5class_practical.py`

Its final performance was not good enough for application use:

- Test known-class macro-F1: `0.3683`
- Test practical 5-label macro-F1: `0.2873`

Main problems:

- `S` was weak
- `F` was very weak
- `Q` was not usable as a learned class

This model was not selected for integration.

## Final Selected Approach

The final selected MIT-BIH analysis pipeline is:

1. Stage 1 binary classifier: `Normal vs Abnormal`
2. Stage 2 abnormal subtype classifier: `S / V / F`
3. `Q` is assigned by confidence rejection, not learned directly

This keeps the final analysis labels as:

- `N`
- `S`
- `V`
- `F`
- `Q`

## Model Architecture

Both selected MIT-BIH models use a similar architecture:

- 1D ResNet-like CNN
- SE blocks inside residual blocks
- Input shape: `187 x 1`
- Per-beat z-score normalization

Common structure:

- `Conv1D(32, kernel=7)`
- Residual block with `32` filters
- Residual block with `64` filters
- `MaxPool1D`
- Residual block with `96` filters
- Residual block with `128` filters
- `GlobalAveragePooling1D`
- `Dropout`
- `Dense(96, relu)`
- `Dropout`

The output layer differs by stage:

- Stage 1: `Dense(1, sigmoid)`
- Stage 2: `Dense(3, softmax)`

## Training Augmentation

The selected scripts use light ECG-friendly augmentation:

- additive Gaussian noise
- amplitude scaling
- small temporal shift
- light baseline wander

## Selected Training Scripts

### Stage 1

Script:

- `AI/models_experiments/train_mitbih_stage1_binary.py`

Output files:

- `Models/mitbih_stage1_binary.keras`
- `Models/mitbih_stage1_binary_meta.json`

Task:

- binary classification
- `Normal` vs `Abnormal`

Loss:

- binary focal loss

### Stage 2

Script:

- `AI/models_experiments/train_mitbih_stage2_subtypes_svf.py`

Output files:

- `Models/mitbih_stage2_subtypes_svf.keras`
- `Models/mitbih_stage2_subtypes_svf_meta.json`

Task:

- abnormal subtype classification only
- `S / V / F`

Loss:

- multiclass focal loss

## Stage 1 Results

The selected hard threshold for stage 1 was:

- `0.76`

The selected uncertainty band for stage 1 was:

- `p_abn <= 0.45` -> `N`
- `p_abn >= 0.80` -> pass to Stage 2
- otherwise -> `Q`

### Stage 1 Test Results At Hard Threshold 0.76

- Accuracy: `0.9150`
- F1 abnormal: `0.6279`
- Macro average F1: `0.7900`

Classification summary:

- Normal precision: `0.9569`
- Normal recall: `0.9472`
- Normal F1: `0.9520`
- Abnormal precision: `0.6039`
- Abnormal recall: `0.6539`
- Abnormal F1: `0.6279`

### Stage 1 Test Results With Uncertainty Band [0.45, 0.80]

- Coverage: `40787 / 49694`
- Unknown rate: `0.1792`
- Covered accuracy: `0.9284`
- Covered macro-F1: `0.8273`

Covered classification summary:

- Normal precision: `0.9711`
- Normal recall: `0.9480`
- Normal F1: `0.9594`
- Abnormal precision: `0.6376`
- Abnormal recall: `0.7643`
- Abnormal F1: `0.6952`

Interpretation:

- Stage 1 is suitable as a gatekeeper.
- It separates clearly normal beats from clearly abnormal beats.
- Borderline beats are redirected to `Q`.

## Stage 2 Results

The selected confidence threshold for stage 2 was:

- `0.67`

### Stage 2 Test Results Without Rejection

- Accuracy: `0.8369`
- Macro-F1: `0.6720`
- Weighted-F1: `0.8296`

Per-class results:

- `S`: precision `0.8589`, recall `0.7654`, F1 `0.8094`
- `V`: precision `0.8662`, recall `0.9469`, F1 `0.9047`
- `F`: precision `0.3542`, recall `0.2629`, F1 `0.3018`

### Stage 2 Test Results With Confidence Rejection At 0.67

- Coverage: `4227 / 5445`
- Unknown rate: `0.2237`
- Covered accuracy: `0.9040`
- Covered macro-F1: `0.6854`

Covered per-class results:

- `S`: precision `0.8907`, recall `0.8481`, F1 `0.8689`
- `V`: precision `0.9335`, recall `0.9774`, F1 `0.9549`
- `F`: precision `0.3106`, recall `0.1855`, F1 `0.2323`

Interpretation:

- `V` is strong.
- `S` is strong enough for analysis use.
- `F` remains weaker and should be used carefully.

## Recommended Final Inference Logic

The recommended final decision flow for the analysis page is:

1. Run Stage 1 binary model.
2. If `p_abn <= 0.45`, return `N`.
3. If `0.45 < p_abn < 0.80`, return `Q`.
4. If `p_abn >= 0.80`, run Stage 2 subtype model.
5. In Stage 2:
   - if `max_prob < 0.67`, return `Q`
   - otherwise return `S`, `V`, or `F`

Extra safety recommendation for `F`:

- If Stage 2 predicts `F` but confidence is below `0.85`, return `Q` instead of `F`

This produces the final practical labels:

- `N`
- `S`
- `V`
- `F`
- `Q`

## Integration Decision

The final integration decision is:

- Do not use the direct practical 5-label model.
- Use the selected two-stage MIT-BIH pipeline.
- Keep the monitor page on the existing PTBDB live binary model.
- Use the MIT-BIH two-stage pipeline only for the analysis page.

## UI Integration Implemented

The MIT-BIH analysis pipeline has been integrated into the desktop application.

### Application design after integration

- Monitor page:
  - still uses the existing PTBDB live binary model
  - still reports `Normal / Abnormal / Noisy`
- Analysis page:
  - runs the MIT-BIH two-stage analysis after the session ends
  - adds MIT-BIH subtype outputs for `N / S / V / F / Q`
  - keeps the original session summary metrics separate from subtype analysis

### Integration behavior

- The monitor page decision path was intentionally left unchanged.
- MIT-BIH models are not used during live beat-by-beat monitoring.
- MIT-BIH runs only when session analysis is triggered.
- Raw beats are now stored in session records so they can be re-analyzed after capture.
- MIT-BIH results are added to:
  - analysis findings
  - summary narrative
  - PDF report metrics

### Implemented inference behavior in the app

The desktop app wrapper follows this logic:

1. Run Stage 1 binary model.
2. If `p_abn <= 0.45`, assign `N`.
3. If `0.45 < p_abn < 0.80`, assign `Q`.
4. If `p_abn >= 0.80`, run Stage 2 subtype model.
5. In Stage 2:
   - if `max_prob < 0.67`, assign `Q`
   - if Stage 2 predicts `F` but confidence `< 0.85`, assign `Q`
   - otherwise assign `S`, `V`, or `F`

## Application Files Added Or Updated

### New files added

- `AI/models_experiments/train_mitbih_5class_practical.py`
- `AI/models_experiments/train_mitbih_stage1_binary.py`
- `AI/models_experiments/train_mitbih_stage2_subtypes_svf.py`
- `Models/mitbih_stage1_binary.keras`
- `Models/mitbih_stage1_binary_meta.json`
- `Models/mitbih_stage2_subtypes_svf.keras`
- `Models/mitbih_stage2_subtypes_svf_meta.json`
- `desktop_app/mitbih_analysis.py`

### Existing application files updated

- `desktop_app/session_analysis.py`
  - `BeatRecord` now stores the raw beat plus MIT-BIH output fields
  - `SessionResult` now stores MIT-BIH summary fields
- `desktop_app/ui/main_window.py`
  - loads the MIT-BIH analyzer
  - stores raw beats during session capture
  - runs MIT-BIH analysis during `Analyze Now`
  - shows MIT-BIH output on the analysis page
  - includes MIT-BIH metrics in the exported PDF
  - shows a separate MIT-BIH analysis readiness status on the home page
- `desktop_app/reporting.py`
  - adds MIT-BIH dominant type and subtype distribution to the PDF
- `desktop_app/ai_summary.py`
  - includes the dominant MIT-BIH subtype in the English local narrative

## Hardware Monitor Calibration Update

After integrating the analysis path, the live hardware path was reviewed separately.

The hardware monitor was found to be intentionally conservative and likely to produce `Abnormal` only rarely because it requires:

- high abnormal probability
- high signal quality
- low motion
- stable abnormal probability across recent beats

To keep the live monitor conservative but slightly more sensitive, a limited hardware-only calibration update was applied.

### Updated hardware parameters

In `desktop_app/ui/main_window.py`, the following values were adjusted:

- `hw_abn_min_prob`: `0.88 -> 0.84`
- `hw_min_quality_for_abn`: `0.78 -> 0.72`

### Why this change was chosen

- The change is intentionally small.
- It does not alter the MIT-BIH analysis logic.
- It does not remove the existing noisy/motion gates.
- It does not change the session analysis thresholds.
- It only makes abnormal detection on hardware slightly less strict.

## Current Practical Status

The current implementation status is:

- MIT-BIH two-stage training completed
- selected thresholds recorded in model meta files
- desktop wrapper implemented
- analysis-page integration implemented
- PDF integration implemented
- local English narrative integration implemented
- conservative hardware sensitivity tuning implemented

## Known Practical Notes

- `S` and `V` are the strongest subtype outputs.
- `F` remains weaker and is guarded by a stricter confidence rule.
- `Q` is a confidence-rejection label, not a directly learned stable class.
- The main session percentages `Normal / Abnormal / Noisy` still come from the live/session binary path.
- MIT-BIH subtype output is an additional analysis layer and should be interpreted as rhythm-pattern support, not a final diagnosis.

## Verification Notes

Code review and integration review were completed for the new path.

The following were specifically checked:

- stage 1 and stage 2 model meta compatibility
- session beat storage for post-session analysis
- analysis-page population
- report export fields
- separation between monitor-page logic and analysis-page logic

A full runtime verification from the command line could not be completed inside the current environment because the local `.venv` launcher resolves through a Windows Store Python path and fails to start reliably from this session.

This means:

- the integration was reviewed carefully in code
- model file names and meta keys were checked
- final smoke testing should still be done directly in the application UI

## Recommended Manual Smoke Test

Before final delivery, the application should be checked with a short manual test:

1. Start a hardware or replay session and confirm the monitor page still shows only `Normal / Abnormal / Noisy`.
2. Run `Analyze Now` and confirm the analysis page shows MIT-BIH subtype output.
3. Confirm the PDF report contains:
   - `Dominant MIT-BIH Type`
   - `MIT-BIH Distribution`
4. Confirm the home page shows separate readiness text for:
   - core models
   - MIT-BIH analysis models

## Next Suggested Step

The next practical step is manual UI validation, not retraining.

If the hardware path still produces too few abnormal beats after the current conservative tuning, the next adjustment should also remain small and localized to the hardware thresholds only.
