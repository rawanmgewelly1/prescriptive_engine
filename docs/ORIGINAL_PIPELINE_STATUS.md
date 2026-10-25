# Original seven-stage pipeline: acceptance record

| Stage | Implementation status | Verification limitation |
|---|---|---|
| 1. User Scenario Inputs | Implemented | Explicit JSON, no interactive GUI yet |
| 2. Renewable Power Prediction | Fixed-power mode operational; two-model branch written | Missing Solar `.pkl`; model chronology and scope not validated |
| 3. Baseline Prediction | Executed by saved Hydrogen CatBoost | Research predictive model, not independently externally validated |
| 4. Hydrogen Flow Mapping | Ideal-gas reference volume with explicitly assumed regulated FC demand | Not a measured device-level hydrogen chain |
| 5. Fuel Cell Feasibility | XGB predicted class for hypothesized operating point; supply-capacity check per candidate | Fixed FC operating point, physical Feasibility **not** verified |
| 6. Prescriptive Optimization | Fixed 200 kW demo; six-variable seeded LHS search; objective H2 | No global optimum, safety constraints, or real consumed-power proof |
| 7. Results & Save | Auditable JSON + CSV outputs | Existing dashboard not yet linked to new results |

**Research statuses are conditional and must not be renamed to `PHYSICAL_PASS`, `SAFETY_APPROVED`, or `SYSTEM_VALIDATED`.**

## Demonstration run (locally tested with available user artifacts)

The included `configs/scenarios/RESEARCH_DEMO_200KW.json` uses an explicit baseline from Hydrogen source sample 1511 (observed power 199.91 kW, re-evaluated at fixed 200 kW). Bounds are observed 5th–95th percentiles, *not engineering limits*. The FC point is an assumed median of historically labeled Pass rows. See the outputs for numbers; they are **not** the 535/1622 from the prior screening pipeline.

The strict alternative refuses to recommend any candidate without an assumed FC bridge. This is the appropriate operating interpretation until connection evidence is supplied.
