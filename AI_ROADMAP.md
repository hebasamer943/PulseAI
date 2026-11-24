# AI Strengthening Roadmap

## Goal

Upgrade the project from a strong AI demo into a more rigorous AI system with:

- measurable model performance
- better hardware-to-model alignment
- cleaner separation of AI logic from UI
- clearer clinical-style session interpretation

## Current Honest Status

### Strong points

- End-to-end workflow is already present: source -> model -> session analysis -> report
- Multiple input modes exist: CSV, image, and live hardware
- Session summary and PDF reporting are already implemented
- UI is product-like, not just a prototype script

### Main weaknesses

- No formal evaluation dashboard or script with standard metrics
- Hardware behavior still suffers from domain shift relative to training data
- Too much AI decision logic lives inside `desktop_app/ui/main_window.py`
- No dedicated calibration flow for hardware
- No explicit uncertainty/review mode beyond threshold heuristics
- No automated test layer for model / session logic

## Priority Order

1. Formal evaluation
2. Hardware calibration
3. Better beat alignment with training distribution
4. Separation of AI logic from UI
5. Session-level AI interpretation improvements
6. Tests and validation artifacts

## 7-Day Execution Plan

### Day 1: Build evaluation baseline

Create a model evaluation script and produce baseline metrics.

Deliverables:

- `desktop_app/model_eval.py`
- Metrics for PTBDB
- Metrics for MITBIH
- Confusion matrix export

Minimum metrics:

- Accuracy
- Precision
- Recall
- F1 score
- Specificity
- ROC-AUC if probabilities are available

### Day 2: Separate metrics by source

Do not mix all data into one result.

Create separate reports for:

- PTBDB
- MITBIH
- hardware validation sessions

Deliverables:

- one summary table comparing all sources
- one note describing where the model performs best and worst

### Day 3: Collect hardware validation data

Record short hardware sessions and store beat-level data.

For each beat, keep:

- timestamp
- beat window
- BPM
- quality score
- motion score
- probability of abnormal
- predicted label

Deliverables:

- a local folder of hardware validation captures
- at least a small manually reviewed subset labeled as:
  - normal
  - noisy
  - abnormal-like

### Day 4: Calibrate hardware behavior

Tune the live inference layer using the hardware validation set.

Tune these items:

- beat cropping window
- threshold floor
- abnormal minimum probability
- noisy gates
- motion sensitivity
- quality fusion behavior

Deliverables:

- one chosen hardware config
- one short note explaining why that config was selected

### Day 5: Add hardware calibration mode

Build a calibration flow in the app.

Suggested behavior:

- user presses `Calibrate Hardware`
- app records 20 to 30 seconds
- app estimates:
  - signal baseline
  - amplitude range
  - noise baseline
  - motion baseline
  - recommended threshold values

Deliverables:

- calibration button in UI
- saved calibration profile
- visible message confirming calibration result

### Day 6: Refactor AI logic out of UI

Move hardware inference logic into dedicated modules.

Suggested files:

- `desktop_app/hw_decision.py`
- `desktop_app/hw_calibration.py`
- `desktop_app/model_eval.py`

Move these responsibilities out of `main_window.py`:

- abnormal/noisy decision logic
- probability fusion
- motion and quality gating
- hardware calibration policy

Deliverables:

- thinner `main_window.py`
- easier debugging and testing

### Day 7: Add AI performance and model documentation

Create a clear explanation layer for the project.

Deliverables:

- one performance page or report
- one `MODEL_CARD.md`

The model card should include:

- training datasets
- input shape
- preprocessing assumptions
- supported use case
- failure cases
- hardware limitations
- non-diagnostic disclaimer

## File-by-File Implementation Map

### `desktop_app/model_eval.py`

Purpose:

- run offline evaluation on CSV datasets
- compute metrics
- export text/CSV summaries

Should contain:

- dataset loaders
- probability prediction runner
- threshold sweep utility
- metrics computation
- confusion matrix generation

### `desktop_app/hw_decision.py`

Purpose:

- isolate live beat decision logic from UI

Should contain:

- noisy gating
- abnormal gating
- dynamic threshold policy
- decision explanation fields

Suggested output object:

- `pred_label`
- `is_noisy`
- `p_abn_raw`
- `p_abn_smoothed`
- `threshold_used`
- `quality_used`
- `motion_used`
- `reason`

### `desktop_app/hw_calibration.py`

Purpose:

- estimate hardware-specific live settings

Should contain:

- calibration session collector
- baseline stats
- recommended thresholds
- crop recommendation
- profile save/load

### `desktop_app/live_serial.py`

Purpose:

- raw serial parsing
- beat window extraction
- BPM update

Planned improvements:

- log calibration diagnostics
- optionally store both raw and processed windows
- keep hardware crop settings configurable, not hard-coded

### `desktop_app/signal_quality.py`

Purpose:

- beat quality estimation

Planned improvements:

- expose more diagnostic features
- separate contact failure from motion noise
- return structured detail, not just score/label/hint

Suggested extra outputs:

- baseline_wander
- high_freq_noise
- clipping_flag
- motion_proxy

### `desktop_app/session_analysis.py`

Purpose:

- session-level interpretation

Planned improvements:

- add confidence-aware session scoring
- add burden severity categories
- add noise-adjusted summary logic
- add uncertainty flag if too many beats are low confidence

### `desktop_app/reporting.py`

Purpose:

- final report generation

Planned improvements:

- include AI confidence notes
- include calibration state
- include data source notes
- include statement on signal reliability

### `desktop_app/ui/main_window.py`

Purpose:

- UI only

Target state:

- keeps rendering, controls, and user interaction
- calls AI modules instead of containing their full logic

## AI Features That Would Most Improve the Project

### 1. Probability calibration

Current raw abnormal probability is likely a model score, not a fully calibrated clinical probability.

Recommended:

- Platt scaling
- or isotonic regression

Benefit:

- displayed `P(abn)` becomes more meaningful

### 2. Uncertainty mode

Current behavior is mostly binary plus noisy.

Add:

- `Normal`
- `Abnormal`
- `Uncertain / Review`

Use `Uncertain` when:

- probability is near threshold
- models disagree
- quality is borderline
- crop alignment is unstable

Benefit:

- fewer false alarms
- more honest AI behavior

### 3. Hardware domain adaptation

If possible, collect hardware beats and:

- fine-tune the final layer
- or calibrate threshold on hardware-only validation data

Benefit:

- biggest reduction in false abnormal on live hardware

### 4. Decision explanation

For every live beat, expose the reason:

- `abnormal_high_probability`
- `blocked_by_motion`
- `blocked_by_low_quality`
- `sent_to_noisy`
- `uncertain_probability`

Benefit:

- easier debugging
- easier to defend in review/demo

## What "Excellent" Looks Like for This Project

The project becomes genuinely strong when you can say:

- here is offline performance on PTBDB
- here is offline performance on MITBIH
- here is hardware validation behavior
- here is calibration logic
- here is how uncertainty is handled
- here are known limitations

That is the difference between:

- "an app with a model"

and:

- "an AI system with measurable behavior"

## Minimum Deliverables Before Final Review

- `model_eval.py` exists and runs
- one metrics report exists
- one hardware calibration flow exists
- one hardware validation dataset exists
- one model card exists
- Arabic encoding issue in `desktop_app/ai_summary.py` is fixed
- main decision logic is moved out of `main_window.py`

## Recommended Next Build Order

1. build `desktop_app/model_eval.py`
2. create hardware validation capture folder
3. add calibration profile support
4. move live decision logic into `desktop_app/hw_decision.py`
5. add uncertainty/review mode
6. add performance summary page
