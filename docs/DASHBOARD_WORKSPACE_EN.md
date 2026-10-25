# Research Scenario Workspace — User Guide

## Start (Windows)

Open a PowerShell terminal in the existing project folder:

```powershell
cd D:\prescriptive_engine
python .\run_dashboard.py
```

The browser opens `http://127.0.0.1:8765/`. You can also double-click `RUN_DASHBOARD.cmd` in the project directory. The service binds **localhost only**; it is not hosted publicly and does not control equipment. Press **Ctrl+C** to stop the terminal service.

The existing historical Pareto dashboard remains visible below the new **Scenario Workspace**. Historical chart points are **not** the newly optimized points. Newly run scenarios show their baseline, optimized candidate, research eligibility, caveats, and CSV exports inside the workspace panel.

## Workflow

1. Choose one of the existing scenario templates.
2. Choose fixed user-input power or hypothetical Solar + Wind model predictions. For model predictions, enter the inputs for both models and acknowledge that they are **not proven synchronized physical power**.
3. Set the six baseline operating variables, and each minimum/maximum research bound. The six controls vary during Latin Hypercube search; power is fixed for the scenario.
4. Select a Fuel Cell mapping policy: `BLOCKED_NO_MAPPING` (strict; prevents a validated candidate), `ASSUMED_REGULATED_DRAW` (fixed FC input, often non-discriminating), or `ASSUMED_CANDIDATE_FLOW` (candidate-specific estimated flow, which is not a verified device link).
5. Explicitly acknowledge all applicable research assumptions and select **Run Research Scenario**.
6. View the status, baseline, best conditional result, the 12 highest conditional candidates, and CSV downloads. Select saved runs to inspect earlier results.

Runs are persisted as separate folders under `outputs/dashboard_runs/dashboard_<timestamp>_<random>/`. Each run stores its own `scenario.json`, `pipeline_status.json` and the pipeline's CSV artifacts. Originals in `configs/scenarios/` and trained models in `models/` are not edited.

## Limitations and scope

- Results are **model predictions**; physical safety has not been approved.
- The electricity requirement check uses an **illustrative specific-energy benchmark**, not measured electricity consumption. The displayed fixed power represents **available**, not **consumed**, power.
- Solar/Wind model predictions in hypothetical mode are **not verified temporally synchronized measurements**.
- Fuel Cell model classes are not engineering-certified pass/fail outcomes. Assumed conversion to hydrogen L/min and downstream flow regulation do not establish a tested device link.
- Operating feature support is empirical and does not establish manufacturer safe-operation limits.
- Saved pickle estimators may warn about training/runtime version differences; the previous spot checks are not complete validation.
- The app binds `127.0.0.1` and exposes only allowlisted local templates/outputs. Do not expose its port to the network, an external reverse proxy, or the Internet. It has no user authentication.
- Requires the project's existing Python model dependencies and trusted local model artifacts. No new pip packages are required for the HTTP server itself.

## Acceptance checks

```powershell
python -m unittest discover -s tests -q
python .\tools\validate_research_stage.py
python .\run_dashboard.py
```

Then open the local URL, select `RESEARCH_DEMO_200KW`, accept research-only conditions, and run. Verify the saved run contains a status and the CSVs. Repeat with the strict Fuel Cell mode: it must **not** produce a falsely eligible candidate.

The older `build_dashboard.py` may still regenerate the historical portion. Launching via `run_dashboard.py` dynamically injects the new workspace into the existing HTML, so no template overwrite is needed.
