# Research Pipeline Stage — Closure and Limits

## Objective

Implement and test the originally specified seven-stage research pipeline, without silently turning empirical data ranges or regression predictions into engineering safety approvals.

## Delivered in this patch

1. User-entered fixed renewable power and hypothetical same-scenario Solar+Wind model prediction modes.
2. Solar and Wind model inference feeding one **fixed** renewable-power value for each hydrogen scenario. Inputs are hypothetical; their original time series are not chronologically aligned.
3. Baseline and six-variable seeded Latin Hypercube research search.
4. A **seven-dimensional joint observed-support diagnostic** (power plus the six controls), in addition to marginal bounds. Distance cutoff is the 95th percentile of nearest-other-observation distance in IQR-scaled feature space; this is **not** a physical safety limit.
5. Two explicitly different *research-only* Fuel Cell bridge policies:
   - `ASSUMED_REGULATED_DRAW`: fixed downstream flow and fixed FC operating point. Classifier inputs are identical for all adequately supplied candidates; model `Pass` is **not candidate-discriminating**. Hydrogen supply is still checked per candidate.
   - `ASSUMED_CANDIDATE_FLOW`: use each candidate's entire theoretical H2 reference-volume rate as its hypothetical FC input; candidates whose rate falls outside the FC training data's observed flow interval are `NOT_VERIFIED`, not `Pass`. No pressure regulation or device coupling is assumed to be validated.
   - `BLOCKED_NO_MAPPING`: deliberately return **no feasible candidate**.
6. Clear distinction between research SEC benchmark (55 kWh/kg in the demonstration) and **measured** electrolyzer consumption (missing).
7. Four-model spot prediction fingerprint plus repeatable acceptance test runner.

## Acceptance command

Run at the project root:

```powershell
python .\tools\verify_manifest.py
python .\tools\check_model_prediction_consistency.py
python .\tools\validate_research_stage.py
```

Accepted status is `RESEARCH_SOFTWARE_STAGE_ACCEPTED_WITH_DISCLOSED_LIMITATIONS` in `outputs/RESEARCH_STAGE_ACCEPTANCE.json`. This is **software research acceptance only**, **not a physically validated plant system**.

## Why there is no real-world certified optimum

The user-provided training artifacts and data do not establish (a) synchronized, correctly scoped real-world Solar+Wind inputs; (b) measured electrolyzer energy use as a function of six operating controls; (c) a real connected Fuel Cell flow/pressure-regulation measurement chain and FC operating policy; or (d) independent certification of safe operating ranges and model generalization. No code update can legitimately manufacture this evidence.

## Sample scenarios

- `configs/scenarios/RESEARCH_DEMO_200KW.json` — 200-kW user-entered power; regulated FC draw. Expected all eligible candidate model FC classes may remain identical; this is disclosed, not a proof of FC selectivity.
- `configs/scenarios/RESEARCH_FOUR_MODELS_HYPOTHETICAL.json` — 4-model run, hypothetically combining one solar source and one wind turbine from *unaligned source eras*, with around 200 kW predicted under the shipped sample feature values.
- `configs/scenarios/RESEARCH_FC_DIRECT_FLOW_SUPPORT_CHECK.json` — candidate-varying FC inlet reference flow; properly rejects out-of-support extrapolation rather than printing Pass.
- `configs/scenarios/STRICT_NO_FC_BRIDGE.json` — no FC mapping, no conditional feasible result.

## Remaining steps for engineering integration

Collect verified renewable capacity/time provenance, an electrolyzer manufacturer or measured specific-energy function and safe operating envelope, Fuel Cell actual inlet operating protocol and measured H2 delivered/demanded, plus evidence for model compatibility across Python/scikit-learn/XGBoost versions. Only after these are supplied and independently validated can real-world feasibility be claimed.

No previous models were retrained or rewritten. Existing dashboard code remains unchanged; binding it to this research pipeline is a later UI task, not implied by the current update.
