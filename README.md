# Green Hydrogen Prescriptive Engine — Organized Research Source

**Status:** Original seven-stage pipeline implemented and locally tested for a clearly labeled **research simulation**. **NOT** a validated physical Solar–Wind–Electrolyzer–Fuel Cell system, and **NOT** a safety-certified controller.

## Run the new pipeline on Windows

1. Extract the entire ZIP. Do not overwrite the original project before testing.
2. Create/activate a Python virtual environment and run `python -m pip install -r requirements.txt`.
3. From the extracted project root, run:

```powershell
python run_pipeline.py --scenario configs/scenarios/RESEARCH_DEMO_200KW.json
```

4. Open `outputs/scenarios/RESEARCH_DEMO_200KW/pipeline_status.json`, `baseline.csv`, `all_candidates.csv`, and `best_conditional_candidate.csv`.
5. Run the **strict** scenario without an assumed hydrogen-to-FC gas connection:

```powershell
python run_pipeline.py --scenario configs/scenarios/STRICT_NO_FC_BRIDGE.json
```

The strict scenario correctly produces `BLOCKED_FC_BRIDGE_NO_FEASIBLE_RESULT` and no recommended point.

For tests: `python -m unittest discover -s tests -q`.

## The seven requested stages

1. **User Scenario Inputs**: JSON declares fixed power or hypothetical model inputs, a baseline, six bounds and research assumptions.
2. **Renewable Power Prediction**: supports fixed user-supplied available power; model-based Solar + one Wind turbine is coded and the Solar artifact is now supplied, but the component scenario must supply both models’ required features; aggregation and timeline semantics remain unverified. Historical dates are never silently aligned.
3. **Baseline Prediction**: actual saved Hydrogen CatBoost model at user-specified baseline controls and the same fixed power.
4. **Hydrogen Flow Mapping**: calculates *reference-condition* L/min from kg/h with the ideal-gas law and, only when explicitly enabled, applies an assumed regulated FC draw. Not a measured gas connection.
5. **Fuel Cell Feasibility Check**: candidate hydrogen *supply capacity* is evaluated; an XGB classifier predicts Pass/Fail for the separately assumed FC operating point. This is a **model prediction**, not a measured/safety-validated Pass.
6. **Prescriptive Optimization**: seeded Latin Hypercube search changes only the six requested controls; power remains strictly fixed. Applies a research reference energy comparison and conditional modeled FC filter, selecting maximum **predicted** H2 (no global optimality claim).
7. **Results & Save**: detailed candidate/baseline/best CSVs and JSON audit flags.

## Original input contract

`renewable_power_kw`: **fixed** throughout one scenario; optimize exactly:

- `voltage_v`
- `temperature_c`
- `current_density_a_cm2`
- `electrolyte_concentration_wt_pct`
- `water_flow_l_min`
- `pressure_bar`

**Objective:** maximize `predicted_hydrogen_kg_h`. **Conditional modeled constraint:** `model_fc_class = Pass`, but the actual device connection has **not** been verified.

## Important scientific limitations

- Solar model `.pkl` is now supplied. This enables artifact loading, but not a claim of verified chronological or physical two-source power aggregation.
- Solar chronology, Wind relative times and Hydrogen dates do not form a validated shared timeseries.
- The 200 kW demo is a user-entered research supply assumption, **not** the measured energy delivered to an electrolyzer. The 55 kWh/kg reference is a benchmark, **not** calibrated device consumption.
- The demo FC setpoint is based on median historical Pass records, and its gas demand is an explicit hypothetical draw. All candidate FC model calls at fixed demand use the **same operating point** if sufficient gas is available. Thus a shared predicted Pass does not independently validate every physical candidate.
- Six marginal observed support ranges do not prove physically safe or joint-distribution-valid points. Predictions are not globally optimized or experimentally validated.
- Serialized XGBoost/scikit-learn models may issue version-compatibility warnings. Reproduce the original training environment and independently evaluate held-out predictions before publication.

## Layout

```text
prescriptive_engine/
  engine.py             Legacy compatible engine
  pipeline/             Organized original 7-stage pipeline modules
models/                 Supplied trained models/scalers (flat legacy configuration)
data/                   Supplied datasets (Wind ~319 MB)
selected_features/      Supplied model feature order contracts
configs/scenarios/      Ready research demo + strict and template scenarios
outputs/scenarios/      Generated pipeline scenario results
tests/                  Legacy + original pipeline unit/integration tests
dashboard/              Existing historical dashboard (not yet wired to the new pipeline)
docs/                   Scientific scope, manifest and GitHub instructions
training_source/        Available original training notebooks
research/               Prior research/diagnostic evidence
```

Existing flat model paths and legacy module entrypoints were preserved to avoid breaking tested code. Legacy utilities retained for audit compatibility; do not treat them as the new public API. The new entrypoint is `run_pipeline.py`.

## GitHub

Read `docs/GITHUB_SETUP.md` before publishing. Datasets, notebooks and binary models are ignored from regular Git to avoid >100 MB pushes and unreviewed publication rights. They are included in this private delivery ZIP for local operation.

## Download packaging

For reliable downloads, the current delivery is provided as one **core project ZIP** and separate ZIPs holding pieces of the large compressed Wind dataset. See `DOWNLOAD_AND_RESTORE_EN.md` and run `python RESTORE_WIND_DATA.py` after extracting all packages. This is a packaging change only; research calculations and model weights are unchanged.

## Research stage acceptance update

See `docs/RESEARCH_STAGE_COMPLETION.md`. The four-model hypothetical scenario and the FC out-of-support challenge are now executable. The empirical joint-support gate and four-model prediction spot check are research diagnostics, NOT physical-engineering validation.

Run `python tools/validate_research_stage.py` to execute the research-only acceptance suite.
