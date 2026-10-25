from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prescriptive_engine import engine


class FakeRegressor:
    def __init__(self, names, value=None):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.value = value

    def predict(self, X):
        if self.value is not None:
            return np.full(len(X), self.value, dtype=float)
        return np.asarray(X, dtype=float).sum(axis=1)


class FakeScaler:
    def __init__(self, names):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)

    def transform(self, X):
        return (np.asarray(X, dtype=float) - 1.0) / 2.0


class AnonymousScaler:
    n_features_in_ = 2

    def transform(self, X):
        return np.asarray(X, dtype=float)


class WrongCountScaler(FakeScaler):
    def __init__(self, names):
        super().__init__(names)
        self.n_features_in_ = len(names) + 1


class WrongLengthRegressor(FakeRegressor):
    def predict(self, X):
        return np.asarray([1.0])


class FakeClassifier:
    classes_ = np.asarray(["Fail", "Pass"], dtype=object)
    feature_names_in_ = np.asarray(
        ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"], dtype=object
    )
    n_features_in_ = 4

    def predict(self, X):
        return np.asarray(["Pass" if row[0] > 0 else "Fail" for row in np.asarray(X)])

    def predict_proba(self, X):
        labels = self.predict(X)
        p = np.where(labels == "Pass", 0.9, 0.1)
        return np.column_stack([1 - p, p])


class FakeNumericClassifier:
    classes_ = np.asarray([0, 1])
    feature_names_in_ = np.asarray(
        ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"], dtype=object
    )
    n_features_in_ = 4

    def predict(self, X):
        return np.asarray([1 if row[0] > 0 else 0 for row in np.asarray(X)])

    def predict_proba(self, X):
        labels = self.predict(X)
        p = np.where(labels == 1, 0.8, 0.2)
        return np.column_stack([1 - p, p])



class NamelessRegressor:
    n_features_in_ = 2

    def predict(self, X):
        return np.asarray(X, dtype=float).sum(axis=1)


class BadProbabilitySumClassifier(FakeClassifier):
    def predict_proba(self, X):
        return np.full((len(X), 2), 0.2)


class BadProbabilityRangeClassifier(FakeClassifier):
    def predict_proba(self, X):
        return np.tile([1.2, -0.2], (len(X), 1))


class BadProbabilityShapeClassifier(FakeClassifier):
    def predict_proba(self, X):
        return np.ones((len(X), 1))


class FakePipeline:
    def __init__(self, names):
        self.steps = [("scale", FakeScaler(names)), ("model", FakeRegressor(names))]
        self.named_steps = dict(self.steps)

    def predict(self, X):
        return self.named_steps["model"].predict(X)

def small_cfg():
    return {
        "hydrogen_target": "hydrogen_production_kg_h",
        "wind_target": "wind_power_kw",
        "solar_target": "solar_power_kw",
        "decision_variables": ["voltage_v", "temperature_c"],
        "candidate_generation": {
            "method": "correlated_resample", "n_candidates": 20, "random_seed": 7,
            "objective": "hydrogen_production_kg_h", "ranking_direction": "maximize", "top_n": 5
        },
        "approved_ranges": {},
        "constraints": [],
        "alignment": {
            "frequency": "h", "timestamps_compatible_verified": False,
            "solar_power_semantics_verified": False, "wind_power_semantics_verified": False,
            "wind_aggregation_semantics_verified": False,
            "wind_first_order_verified": False, "solar_first_order_verified": False,
            "wind_duplicate_timestamp_semantics_verified": False,
            "solar_duplicate_timestamp_semantics_verified": False,
            "wind_aggregation": "mean", "wind_duplicate_timestamp_aggregation": "mean",
            "solar_duplicate_timestamp_aggregation": "mean",
            "solar_power_scope": "farm_total", "wind_power_scope": "farm_total",
        },
        "alignment_verification_reference": {
            "timestamps_compatible_verified": "", "solar_power_semantics_verified": "",
            "wind_power_semantics_verified": "", "wind_aggregation_semantics_verified": "",
            "timestamp_timezone_mode": "", "wind_first_order_verified": "", "solar_first_order_verified": "",
            "wind_duplicate_timestamp_semantics_verified": "", "solar_duplicate_timestamp_semantics_verified": "",
            "solar_power_scope": "solar target is farm-total kW", "wind_power_scope": "wind aggregation yields farm-total kW",
        },
        "renewable_combination": {"method": "sum"},
        "wind_aggregation_evidence": {
            "target_semantics": "farm_total_power", "turbine_id_column": "turbine_id",
            "power_unit": "kW", "simultaneous_observations_verified": False,
            "turbine_coverage_verified": False, "reference": "synthetic test fixture semantics",
        },
        "fuel_cell_pass_constraint": {
            "required_class_label": "Pass", "class_label_mapping": {},
            "class_mapping_verified": False, "class_mapping_reference": None,
        },
        "require_verified_preprocessing": True,
        "require_verified_target_units": True,
        "target_units_verified": {"solar": True, "wind": True, "hydrogen": True},
        "target_units_verification_reference": {"solar": "training target evidence", "wind": "training target evidence", "hydrogen": "training target evidence"},
        "observed_target_units_verified": {"solar": True, "wind": True, "hydrogen": True},
        "observed_target_units_verification_reference": {"solar": "source target evidence", "wind": "source target evidence", "hydrogen": "source target evidence"},
        "input_modes": {"solar": "raw", "wind": "raw", "hydrogen": "raw", "fuel_cell": "raw"},
        "input_modes_verified": {"solar": True, "wind": True, "hydrogen": True, "fuel_cell": True},
        "input_modes_verification_reference": {"solar": "training preprocessing evidence", "wind": "training preprocessing evidence", "hydrogen": "training preprocessing evidence", "fuel_cell": "training preprocessing evidence"},
        "model_feature_order_verified": {"solar": False, "wind": False, "hydrogen": False, "fuel_cell": False},
        "model_feature_order_reference": {"solar": "", "wind": "", "hydrogen": "", "fuel_cell": ""},
        "scaler_feature_order_verified": {"solar": False, "wind": False, "hydrogen": False, "fuel_cell": False},
        "scaler_feature_order_reference": {"solar": "", "wind": "", "hydrogen": "", "fuel_cell": ""},
        "wind_chunksize": 2,
        "wind_time_numeric_unit": None,
        "wind_calendar": {"day_index_zero_date": "2024-01-01", "day_index_zero_date_verified": True, "day_index_zero_date_reference": "source dataset documentation"},
        "power_validation": {"reject_negative_predictions": True},
    }


class EngineTests(unittest.TestCase):
    def test_expected_estimator_type_check_is_explicit(self):
        FakeCatBoostRegressor = type("CatBoostRegressor", (), {})
        FakeXGBClassifier = type("XGBClassifier", (), {})
        self.assertTrue(engine.has_expected_model_type(FakeCatBoostRegressor(), "solar"))
        self.assertFalse(engine.has_expected_model_type(FakeCatBoostRegressor(), "wind"))
        self.assertTrue(engine.has_expected_model_type(FakeXGBClassifier(), "fuel_cell"))
        WrappedCatBoost = type("CatBoostRegressor", (), {})
        wrapper = type("EstimatorWrapper", (), {"named_steps": {"model": WrappedCatBoost()}})()
        self.assertTrue(engine.has_expected_model_type(wrapper, "hydrogen"))

    def test_config_is_valid_json_and_has_safe_defaults(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        self.assertTrue(cfg["require_verified_preprocessing"])
        self.assertEqual(cfg["candidate_generation"]["method"], "correlated_resample")
        self.assertEqual(cfg["constraints"], [])
        self.assertTrue(cfg["fuel_cell_pass_constraint"]["class_mapping_verified"])
        self.assertEqual(cfg["fuel_cell_pass_constraint"]["class_label_mapping"], {"0": "Fail", "1": "Pass"})
        self.assertIn("fuel_cell_training.ipynb", cfg["fuel_cell_pass_constraint"]["class_mapping_reference"])

    def test_read_config_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                engine.read_config(Path(tmp) / "missing.json")

    def test_prepare_input_enforces_exact_feature_order(self):
        names = ["a", "b"]
        model = FakeRegressor(names)
        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        X = engine.prepare_input(frame, model, names, None, "raw", "test")
        self.assertEqual(list(X.columns), names)
        with self.assertRaisesRegex(ValueError, "feature order differs"):
            engine.prepare_input(frame, model, ["b", "a"], None, "raw", "test")

    def test_prepare_input_rejects_missing_features(self):
        with self.assertRaisesRegex(ValueError, "missing model input columns"):
            engine.prepare_input(pd.DataFrame({"a": [1]}), FakeRegressor(["a", "b"]), ["a", "b"], None, "raw", "test")

    def test_scaled_mode_uses_scaler_and_validates_order(self):
        names = ["a", "b"]
        X = engine.prepare_input(
            pd.DataFrame({"a": [3.0], "b": [5.0]}), FakeRegressor(names), names,
            FakeScaler(names), "scaled", "test"
        )
        np.testing.assert_allclose(X.to_numpy(), [[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "scaler feature order mismatch"):
            engine.prepare_input(
                pd.DataFrame({"a": [3.0], "b": [5.0]}), FakeRegressor(names), names,
                FakeScaler(["b", "a"]), "scaled", "test"
            )

    def test_scaled_mode_requires_scaler(self):
        with self.assertRaisesRegex(engine.ConfigurationError, "scaler file is not configured"):
            engine.prepare_input(pd.DataFrame({"a": [1.]}), FakeRegressor(["a"]), ["a"], None, "scaled", "test")

    def test_scaler_without_feature_names_requires_explicit_order_evidence(self):
        frame = pd.DataFrame({"a": [1.], "b": [2.]})
        with self.assertRaisesRegex(engine.StageBlocked, "feature identity/order cannot be verified"):
            engine.prepare_input(frame, FakeRegressor(["a", "b"]), ["a", "b"],
                                 AnonymousScaler(), "scaled", "test")
        result = engine.prepare_input(frame, FakeRegressor(["a", "b"]), ["a", "b"],
                                      AnonymousScaler(), "scaled", "test",
                                      scaler_order_verified=True,
                                      scaler_order_reference="training preprocessing source",
                                      scaler_feature_order=["a", "b"])
        self.assertEqual(result.shape, (1, 2))

    def test_scaler_feature_count_mismatch_is_rejected(self):
        frame = pd.DataFrame({"a": [1.], "b": [2.]})
        with self.assertRaisesRegex(ValueError, "scaler expects 3 features"):
            engine.prepare_input(frame, FakeRegressor(["a", "b"]), ["a", "b"],
                                 WrongCountScaler(["a", "b"]), "scaled", "test")

    def test_prediction_row_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "predictions for 2 rows"):
            engine.predict(
                WrongLengthRegressor(["a"]), pd.DataFrame({"a": [1., 2.]}),
                ["a"], None, "raw", "test"
            )

    def test_nonfinite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            engine.prepare_input(pd.DataFrame({"a": [np.inf]}), FakeRegressor(["a"]), ["a"], None, "raw", "test")

    def test_target_units_gate_blocks_unverified_prediction_scale(self):
        cfg = small_cfg()
        cfg["target_units_verified"]["solar"] = False
        with self.assertRaisesRegex(engine.StageBlocked, "target units/output transform lack a verified flag"):
            engine.require_verified_target_units(cfg, "solar")

    def test_nonfinite_regression_predictions_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            engine.predict(FakeRegressor(["a"], value=np.inf), pd.DataFrame({"a": [1.]}), ["a"], None, "raw", "test")

    def test_candidate_generation_is_deduplicated_reproducible_and_observed_only(self):
        data = pd.DataFrame({
            "voltage_v": [1.0, 2.0, 2.0, 3.0],
            "temperature_c": [10.0, 20.0, 20.0, 30.0],
            "hydrogen_production_kg_h": [0.1, 0.2, 0.2, 0.3],
        })
        cfg = small_cfg()
        a = engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])
        b = engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 3)
        self.assertTrue(a["observed_row_support"].all())
        self.assertFalse(a["candidate_is_synthetic"].any())
        self.assertTrue((a["search_interpretation"] == "RANKING_OBSERVED_OPERATING_POINTS_ONLY").all())

    def test_conflicting_targets_for_identical_operating_points_are_blocked(self):
        cfg = small_cfg()
        data = pd.DataFrame({
            "voltage_v": [1., 1.], "temperature_c": [10., 10.],
            "hydrogen_production_kg_h": [0.1, 0.2],
        })
        with self.assertRaisesRegex(engine.StageBlocked, "conflicting observed target values"):
            engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])

    def test_unverified_target_units_use_neutral_observed_target_column(self):
        cfg = small_cfg()
        cfg["observed_target_units_verified"]["hydrogen"] = False
        data = pd.DataFrame({
            "voltage_v": [1., 2.], "temperature_c": [10., 20.],
            "hydrogen_production_kg_h": [0.1, 0.2],
        })
        got = engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])
        self.assertIn("observed_hydrogen_target_value_unverified_units", got.columns)
        self.assertNotIn("observed_hydrogen_production_kg_h", got.columns)
        self.assertTrue((got["observed_target_units_status"] == "UNVERIFIED_SOURCE_TARGET_SCALE_OR_UNITS").all())

    def test_observed_row_method_does_not_create_new_points(self):
        cfg = small_cfg()
        cfg["candidate_generation"]["method"] = "observed_rows"
        data = pd.DataFrame({
            "voltage_v": [1., 2.], "temperature_c": [10., 20.],
            "hydrogen_production_kg_h": [0.1, 0.2],
        })
        got = engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])
        self.assertEqual(set(zip(got.voltage_v, got.temperature_c)), {(1., 10.), (2., 20.)})

    def test_disabled_constraints_do_not_claim_feasibility(self):
        result, used, verified = engine.evaluate_constraints(
            pd.DataFrame({"y": [1., 2.]}), {"constraints": [{"enabled": False}]}
        )
        self.assertEqual(used, [])
        self.assertFalse(verified)
        self.assertTrue((result["feasibility_status"] == "UNVERIFIED_NO_ENABLED_CONSTRAINTS").all())

    def test_constraint_json_boolean_is_not_evidence(self):
        df = pd.DataFrame({"y": [1., 2.]})
        cfg = {"constraints": [{
            "name": "limit", "enabled": True, "verified": True, "column": "y",
            "operator": "<=", "value": 1.5,
        }]}
        result, used, verified = engine.evaluate_constraints(df, cfg)
        self.assertFalse(verified)
        self.assertEqual(result["feasibility_status"].tolist(), ["UNVERIFIED", "INFEASIBLE"])

    def test_constraint_requires_units_and_reference_even_if_declared_verified(self):
        df = pd.DataFrame({"y": [1.]})
        cfg = {"constraints": [{
            "name": "limit", "enabled": True, "verified": True, "verified_by": "Supervisor",
            "verification_reference": "approval-1", "units": "kW",
            "column": "y", "operator": "<=", "value": 2.,
        }]}
        result, used, verified = engine.evaluate_constraints(df, cfg)
        self.assertTrue(verified)
        self.assertEqual(result.loc[0, "feasibility_status"], "FEASIBLE")

    def test_invalid_constraint_column_fails_clearly(self):
        with self.assertRaisesRegex(engine.ConfigurationError, "missing column"):
            engine.evaluate_constraints(pd.DataFrame({"y": [1]}), {"constraints": [
                {"name": "bad", "enabled": True, "verified": True, "column": "missing",
                 "operator": "<=", "value": 1}
            ]})

    def test_missing_constraint_value_is_conservatively_infeasible(self):
        result, _, verified = engine.evaluate_constraints(
            pd.DataFrame({"y": [np.nan]}),
            {"constraints": [{"name": "limit", "enabled": True, "verified": True,
                              "verified_by": "Supervisor", "verification_reference": "ref",
                              "units": "kW", "column": "y", "operator": "<=", "value": 2}]}
        )
        self.assertTrue(verified)
        self.assertEqual(result.loc[0, "feasibility_status"], "INFEASIBLE")

    def test_alignment_blocked_without_verified_timestamps(self):
        solar = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "predicted_solar_power_kw": [1.]})
        wind = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "predicted_wind_power_kw": [2.]})
        with self.assertRaisesRegex(engine.StageBlocked, "timestamps_compatible_verified lacks traceable documentary evidence"):
            engine.align_renewables(small_cfg(), solar, wind)

    def test_alignment_reports_overlap_and_duplicates(self):
        cfg = small_cfg()
        cfg["alignment"].update({
            "timestamps_compatible_verified": True, "solar_power_semantics_verified": True,
            "wind_power_semantics_verified": True, "wind_aggregation_semantics_verified": True,
            "timestamp_timezone_mode": "naive_same_timezone",
            "solar_duplicate_timestamp_semantics_verified": True,
            "wind_duplicate_timestamp_semantics_verified": True,
        })
        cfg["alignment_verification_reference"].update({
            "timestamps_compatible_verified": "timestamp source docs",
            "solar_power_semantics_verified": "solar model target docs",
            "wind_power_semantics_verified": "wind model target docs",
            "wind_aggregation_semantics_verified": "wind aggregation source docs",
            "timestamp_timezone_mode": "both source timestamps documented as same naive local zone",
            "solar_duplicate_timestamp_semantics_verified": "solar temporal aggregation docs",
            "wind_duplicate_timestamp_semantics_verified": "wind temporal aggregation docs",
            "solar_power_scope": "solar farm total evidence", "wind_power_scope": "wind farm total evidence",
        })
        solar = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 00:05", "2024-01-01 00:10", "2024-01-01 02:00"]),
            "predicted_solar_power_kw": [1., 3., 9.],
        })
        wind = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
            "predicted_wind_power_kw": [2., 4.], "wind_output_scope": ["farm_total", "farm_total"],
        })
        result = engine.align_renewables(cfg, solar, wind)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "predicted_total_renewable_power_kw"], 4.)
        self.assertEqual(result.loc[0, "solar_duplicate_timestamp_rows"], 2)
        self.assertEqual(result.loc[0, "solar_timestamps_dropped_no_overlap"], 1)

    def test_alignment_no_overlap_is_blocked_with_counts(self):
        cfg = small_cfg()
        cfg["alignment"].update({
            "timestamps_compatible_verified": True, "solar_power_semantics_verified": True,
            "wind_power_semantics_verified": True, "wind_aggregation_semantics_verified": True,
            "timestamp_timezone_mode": "naive_same_timezone",
            "solar_duplicate_timestamp_semantics_verified": True,
            "wind_duplicate_timestamp_semantics_verified": True,
        })
        cfg["alignment_verification_reference"].update({
            "timestamps_compatible_verified": "timestamp source docs",
            "solar_power_semantics_verified": "solar model target docs",
            "wind_power_semantics_verified": "wind model target docs",
            "wind_aggregation_semantics_verified": "wind aggregation source docs",
            "timestamp_timezone_mode": "both source timestamps documented as same naive local zone",
            "solar_duplicate_timestamp_semantics_verified": "solar temporal aggregation docs",
            "wind_duplicate_timestamp_semantics_verified": "wind temporal aggregation docs",
            "solar_power_scope": "solar farm total evidence", "wind_power_scope": "wind farm total evidence",
        })
        solar = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "predicted_solar_power_kw": [1.]})
        wind = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02"]), "predicted_wind_power_kw": [2.], "wind_output_scope": ["farm_total"]})
        with self.assertRaisesRegex(engine.StageBlocked, "solar_rows_before=1"):
            engine.align_renewables(cfg, solar, wind)

    def test_solar_derives_calendar_features_and_preserves_source_timestamp(self):
        cfg = small_cfg()
        cfg["solar_timestamp_col"] = "timestamp"
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset_dir"] = tmp
            cfg["solar_dataset_file"] = "solar.csv"
            cfg["alignment"]["solar_timestamp_column"] = "timestamp"
            pd.DataFrame({
                "timestamp": ["2024-02-03 04:05:00", "bad timestamp"],
                "temperature_c": [20., 21.],
                "air_density_kg_m3": [1.2, 1.2],
                "global_irradiance_w_m2": [500., 500.],
                "plane_irradiance_w_m2": [600., 600.],
                "cloud_fraction": [0.1, 0.2],
                "solar_power_kw": [2., 3.],
            }).to_csv(Path(tmp) / "solar.csv", index=False)
            features = [
                "temperature_c", "air_density_kg_m3", "global_irradiance_w_m2",
                "plane_irradiance_w_m2", "cloud_fraction", "Hour", "Day", "DayOfYear",
            ]
            # Calendar columns do not need to be present in the raw CSV.
            result = engine.solar_power(cfg, FakeRegressor(features), None, features)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "timestamp"], pd.Timestamp("2024-02-03 04:05:00"))
            self.assertEqual(result.loc[0, "source_timestamp"], "2024-02-03 04:05:00")
            self.assertEqual(result.loc[0, "input_rows_seen"], 2)
            self.assertEqual(result.loc[0, "invalid_timestamp_rows_rejected"], 1)
            self.assertIn("observed_solar_power_kw", result.columns)

    def test_selected_feature_file_order_mismatch_is_not_silently_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "features.csv").write_text("feature\nb\na\n", encoding="utf-8")
            cfg = {"selected_features_dir": str(root), "selected_feature_files": {"solar": "features.csv"}}
            with self.assertRaisesRegex(engine.ConfigurationError, "order differs from config"):
                engine.load_feature_file(cfg, "solar", ["a", "b"])

    def test_parse_clock_requires_documented_numeric_units(self):
        with self.assertRaisesRegex(engine.StageBlocked, "Numeric wind time_of_day values require"):
            engine.parse_clock(pd.Series([120, 240]), None)
        values = engine.parse_clock(pd.Series(["01:30", "02:00"]), None)
        self.assertEqual(values.tolist(), [90.0, 120.0])

    def test_parse_clock_handles_mixed_values_with_explicit_units(self):
        values = engine.parse_clock(pd.Series(["01:30", 120, "02:15", np.nan]), "minutes")
        self.assertEqual(values.iloc[0], 90.)
        self.assertEqual(values.iloc[1], 120.)
        self.assertEqual(values.iloc[2], 135.)
        self.assertTrue(np.isnan(values.iloc[3]))

    def test_wind_invalid_calendar_origin_blocks_month_feature(self):
        cfg = small_cfg()
        cfg["wind_calendar"]["day_index_zero_date"] = None
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            pd.DataFrame({"day_index": [0], "time_of_day": ["01:00"]}).to_csv(Path(tmp) / "wind.csv", index=False)
            with self.assertRaisesRegex(engine.StageBlocked, "day_index_zero_date lacks documentary verification"):
                engine.wind_power(cfg, FakeRegressor(["Hour", "Minute", "Month"]), None, ["Hour", "Minute", "Month"])

    def test_wind_clock_only_input_never_parsed_as_current_calendar_month(self):
        import warnings
        cfg = small_cfg()
        cfg["wind_calendar"]["day_index_zero_date"] = None
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            pd.DataFrame({"day_index": [0], "time_of_day": ["01:00"]}).to_csv(Path(tmp) / "wind.csv", index=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(engine.StageBlocked, "day_index_zero_date lacks documentary verification"):
                    engine.wind_power(cfg, FakeRegressor(["Hour", "Minute", "Month"]), None, ["Hour", "Minute", "Month"])
            self.assertFalse(any("Could not infer format" in str(w.message) for w in caught))

    def test_wind_mixed_clock_and_datetime_values_are_blocked(self):
        cfg = small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            pd.DataFrame({"day_index": [0, 0], "time_of_day": ["2026-10-02 01:00", "01:10"]}).to_csv(Path(tmp) / "wind.csv", index=False)
            with self.assertRaisesRegex(engine.StageBlocked, "mixes full datetimes and clock-only"):
                engine.wind_power(cfg, FakeRegressor(["Hour", "Minute", "Month"]), None, ["Hour", "Minute", "Month"])

    def test_wind_chunk_processing_and_mean_aggregation(self):
        cfg = small_cfg()
        cfg["alignment"]["wind_aggregation"] = "mean"
        cfg["alignment"]["wind_power_semantics_verified"] = False
        features = ["wind_speed_m_s", "Hour", "Minute", "Month"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            data = pd.DataFrame({
                "day_index": [0, 0, 0], "time_of_day": ["01:00", "01:20", "02:00"],
                "wind_speed_m_s": [1., 3., 5.], "wind_power_kw": [1., 3., 5.],
            })
            data.to_csv(Path(tmp) / "wind.csv", index=False)
            # Model uses wind_speed + Hour + Minute + Month, hence predictions 1+1+0+1=3,
            # 3+1+20+1=25, and 5+2+0+1=8. First two share the same hourly timestamp.
            model = FakeRegressor(features)
            result = engine.wind_power(cfg, model, None, features)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.iloc[0]["observations_aggregated"], 2)
            self.assertEqual(result.iloc[0]["aggregation_method"], "mean")
            self.assertEqual(result.iloc[0]["input_rows_seen"], 3)
            self.assertEqual(result.iloc[0]["input_rows_rejected"], 0)

    def test_wind_sum_aggregation_blocked_without_semantics_evidence(self):
        cfg = small_cfg()
        cfg["alignment"]["wind_aggregation"] = "sum"
        cfg["alignment"]["wind_aggregation_semantics_verified"] = False
        with tempfile.TemporaryDirectory() as tmp:
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            pd.DataFrame({"day_index": [0], "time_of_day": ["01:00"]}).to_csv(Path(tmp) / "wind.csv", index=False)
            with self.assertRaisesRegex(engine.StageBlocked, "sum aggregation is blocked"):
                engine.wind_power(cfg, FakeRegressor(["Hour", "Minute", "Month"]), None, ["Hour", "Minute", "Month"])

    def test_mandatory_fuel_cell_constraint_prevents_feasible_status_without_candidate_mapping(self):
        cfg = small_cfg()
        cfg["fuel_cell_pass_constraint"]["enabled"] = True
        candidates = pd.DataFrame({
            "candidate_id": [1, 2],
            "feasibility_status": ["FEASIBLE", "INFEASIBLE"],
        })
        out, used, verified = engine.apply_fuel_cell_candidate_gate(candidates, cfg, ["approved_range"], True)
        self.assertFalse(verified)
        self.assertIn("fuel_cell_pass_constraint", used)
        self.assertEqual(out["feasibility_status"].tolist(), ["UNVERIFIED", "INFEASIBLE"])
        self.assertTrue((out["fuel_cell_candidate_constraint_status"] ==
                         "BLOCKED_MISSING_VALIDATED_CANDIDATE_INPUT_MAPPING").all())

    def test_fuel_cell_numeric_classes_block_without_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["fuel_cell_dataset_file"] = "fuel.csv"
            pd.DataFrame({
                "hydrogen_flow_l_min": [1.], "voltage_v": [10.],
                "temperature_c": [50.], "pressure_bar": [2.],
            }).to_csv(Path(tmp) / "fuel.csv", index=False)
            result, evidence = engine.fuel_cell_diagnostics(
                cfg, FakeNumericClassifier(), None,
                ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"]
            )
            self.assertEqual(evidence["status"], "BLOCKED_NUMERIC_CLASS_MAPPING_UNVERIFIED")
            self.assertTrue(result["predicted_outcome"].iloc[0].startswith("UNMAPPED_CLASS:"))

    def test_fuel_cell_numeric_classes_use_verified_semantic_mapping_and_probability_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["fuel_cell_dataset_file"] = "fuel.csv"
            cfg["fuel_cell_pass_constraint"].update({
                "class_label_mapping": {"0": "Fail", "1": "Pass"},
                "class_mapping_verified": True,
                "class_mapping_reference": "original LabelEncoder mapping in training notebook",
            })
            pd.DataFrame({
                "hydrogen_flow_l_min": [1., -1.], "voltage_v": [10., 10.],
                "temperature_c": [50., 50.], "pressure_bar": [2., 2.],
            }).to_csv(Path(tmp) / "fuel.csv", index=False)
            result, evidence = engine.fuel_cell_diagnostics(
                cfg, FakeNumericClassifier(), None,
                ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"]
            )
            self.assertEqual(evidence["status"], "PASS")
            self.assertEqual(result["predicted_outcome"].tolist(), ["Pass", "Fail"])
            self.assertAlmostEqual(result["required_label_probability"].iloc[0], 0.8)

    def test_run_engine_writes_blocked_diagnostics_when_real_assets_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
            cfg["dataset_dir"] = "data"
            cfg["models_dir"] = "models"
            cfg["selected_features_dir"] = "selected_features"
            cfg["output_dir"] = "outputs"
            (root / "data").mkdir()
            (root / "models").mkdir()
            (root / "selected_features").mkdir()
            out = root / "outputs"
            out.mkdir()
            (out / cfg["outputs"]["best"]).write_text("stale result")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            result = engine.run_engine(config_path)
            self.assertEqual(result["hydrogen_status"], "BLOCKED")
            self.assertFalse((out / cfg["outputs"]["best"]).exists())
            self.assertTrue((out / cfg["outputs"]["diagnostics"]).exists())
            diag = pd.read_csv(out / cfg["outputs"]["diagnostics"])
            self.assertTrue((diag["status"].astype(str).str.contains("BLOCKED")).any())

    def test_stale_output_cleanup_only_removes_configured_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stale.csv").write_text("old")
            (root / "keep.csv").write_text("keep")
            removed = engine.invalidate_stale_outputs(root, {"best": "stale.csv"})
            self.assertEqual(removed, ["stale.csv"])
            self.assertFalse((root / "stale.csv").exists())
            self.assertTrue((root / "keep.csv").exists())


    def test_config_rejects_malformed_nested_sections(self):
        base = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        cases = [
            ("alignment", []),
            ("candidate_generation", "not-an-object"),
            ("input_modes", None),
            ("outputs", []),
            ("approved_ranges", []),
        ]
        for key, value in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                cfg = dict(base)
                cfg[key] = value
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaises(engine.ConfigurationError):
                    engine.read_config(path)

    def test_config_rejects_noninteger_candidate_count_and_duplicate_output_names(self):
        base = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        for invalid in ("NaN", 0, 1.5, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                cfg = dict(base)
                cfg["candidate_generation"] = dict(base["candidate_generation"], n_candidates=invalid)
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaisesRegex(engine.ConfigurationError, "candidate_generation.n_candidates"):
                    engine.read_config(path)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(base)
            cfg["outputs"] = dict(base["outputs"])
            cfg["outputs"]["best"] = cfg["outputs"]["top"]
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(engine.ConfigurationError, "duplicate filenames"):
                engine.read_config(path)

    def test_config_rejects_output_traversal_and_disabling_evidence_gates(self):
        base = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        for mutate, message in (
            (lambda cfg: cfg["outputs"].__setitem__("best", "../best.csv"), "plain, non-reserved filename"),
            (lambda cfg: cfg.__setitem__("require_verified_target_units", False), "is not supported"),
            (lambda cfg: cfg.__setitem__("require_verified_preprocessing", False), "is not supported"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                cfg = json.loads(json.dumps(base))
                mutate(cfg)
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaisesRegex(engine.ConfigurationError, message):
                    engine.read_config(path)

    def test_config_requires_scaler_and_reference_when_scaled_mode_is_enabled(self):
        base = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            cfg = json.loads(json.dumps(base))
            cfg["input_modes"]["solar"] = "scaled"
            cfg["solar_scaler_file"] = ""
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(engine.ConfigurationError, "solar_scaler_file"):
                engine.read_config(path)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = json.loads(json.dumps(base))
            cfg["input_modes_verified"]["solar"] = True
            cfg["input_modes_verification_reference"]["solar"] = ""
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(engine.ConfigurationError, "input_modes_verification_reference.solar"):
                engine.read_config(path)

    def test_model_feature_identity_requires_evidence_when_artifact_names_are_missing(self):
        frame = pd.DataFrame({"a": [1.], "b": [2.]})
        with self.assertRaisesRegex(engine.StageBlocked, "does not expose feature names"):
            engine.prepare_input(frame, NamelessRegressor(), ["a", "b"], None, "raw", "nameless")
        result = engine.prepare_input(
            frame, NamelessRegressor(), ["a", "b"], None, "raw", "nameless",
            model_order_verified=True,
            model_order_reference="original training pipeline source",
        )
        self.assertEqual(result.shape, (1, 2))

    def test_external_scaler_is_blocked_when_model_pipeline_already_transforms_inputs(self):
        names = ["a", "b"]
        frame = pd.DataFrame({"a": [1.], "b": [2.]})
        with self.assertRaisesRegex(engine.StageBlocked, "internal transformer"):
            engine.prepare_input(frame, FakePipeline(names), names, FakeScaler(names), "scaled", "pipeline")

    def test_generic_artifact_feature_names_are_not_treated_as_identity_evidence(self):
        FakeCatBoostRegressor = type("CatBoostRegressor", (), {"feature_names_": ["0", "1"]})
        self.assertIsNone(engine.model_names(FakeCatBoostRegressor()))
        FakeXGBRegressor = type("XGBRegressor", (), {"feature_names": ["f0", "f1"]})
        self.assertIsNone(engine.model_names(FakeXGBRegressor()))

    def test_parse_clock_valid_seconds_and_invalid_out_of_range_values(self):
        values = engine.parse_clock(pd.Series(["01:02:30", "23:59:59", "24:00", "25:00", "bad", np.nan]), None)
        self.assertAlmostEqual(values.iloc[0], 62.5)
        self.assertAlmostEqual(values.iloc[1], 1439 + 59 / 60)
        self.assertTrue(values.iloc[2:].isna().all())

    def test_wind_results_are_equivalent_across_chunk_sizes(self):
        features = ["wind_speed_m_s", "Hour", "Minute", "Month"]
        data = pd.DataFrame({
            "day_index": [0, 0, 0, 0, 1],
            "time_of_day": ["01:00", "01:20", "01:40", "02:00", "01:00"],
            "wind_speed_m_s": [1., 3., 5., 7., 9.],
            "wind_power_kw": [1., 3., 5., 7., 9.],
        })
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            data.to_csv(Path(tmp) / "wind.csv", index=False)
            for size in (1, 2, 3, 100):
                cfg["wind_chunksize"] = size
                results.append(engine.wind_power(cfg, FakeRegressor(features), None, features))
        for result in results[1:]:
            self.assertEqual(result["timestamp"].tolist(), results[0]["timestamp"].tolist())
            np.testing.assert_allclose(result["predicted_wind_power_kw"], results[0]["predicted_wind_power_kw"], rtol=0, atol=1e-12)
            self.assertEqual(result["observations_aggregated"].tolist(), results[0]["observations_aggregated"].tolist())

    def test_wind_sum_and_first_aggregation_require_evidence_and_preserve_method(self):
        features = ["wind_speed_m_s", "Hour", "Minute", "Month"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            pd.DataFrame({
                "day_index": [0, 0], "time_of_day": ["01:00", "01:00"],
                "turbine_id": ["T1", "T2"],
                "wind_speed_m_s": [1., 3.], "wind_power_kw": [1., 3.],
            }).to_csv(Path(tmp) / "wind.csv", index=False)
            model = FakeRegressor(features)
            cfg["alignment"]["frequency"] = "10min"
            cfg["alignment"]["wind_aggregation"] = "sum"
            cfg["alignment"]["wind_aggregation_semantics_verified"] = True
            cfg["alignment_verification_reference"]["wind_aggregation_semantics_verified"] = "documented simultaneous turbine sum semantics"
            cfg["wind_aggregation_evidence"].update({
                "target_semantics": "per_turbine_power", "turbine_id_column": "turbine_id",
                "power_unit": "kW", "simultaneous_observations_verified": True,
                "turbine_coverage_verified": True, "reference": "source turbine identity and sampling documentation",
            })
            result = engine.wind_power(cfg, model, None, features)
            self.assertEqual(result.loc[0, "aggregation_method"], "sum")
            self.assertEqual(result.loc[0, "wind_output_scope"], "farm_total")
            self.assertAlmostEqual(result.loc[0, "predicted_wind_power_kw"], 8.0)
            cfg["alignment"]["wind_aggregation"] = "first"
            cfg["alignment"]["wind_first_order_verified"] = False
            with self.assertRaisesRegex(engine.StageBlocked, "depends on file order"):
                engine.wind_power(cfg, model, None, features)
            cfg["alignment"]["wind_first_order_verified"] = True
            cfg["alignment_verification_reference"]["wind_first_order_verified"] = "documented chronological row order"
            first = engine.wind_power(cfg, model, None, features)
            self.assertEqual(first.loc[0, "aggregation_method"], "first")
            self.assertAlmostEqual(first.loc[0, "predicted_wind_power_kw"], 3.0)

    def test_wind_sum_blocks_temporal_summing_of_ten_minute_observations(self):
        features = ["wind_speed_m_s", "Hour", "Minute", "Month"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            cfg["alignment"]["frequency"] = "h"
            cfg["alignment"]["wind_aggregation"] = "sum"
            cfg["alignment"]["wind_aggregation_semantics_verified"] = True
            cfg["alignment_verification_reference"]["wind_aggregation_semantics_verified"] = "turbine sum docs"
            cfg["wind_aggregation_evidence"].update({
                "target_semantics": "per_turbine_power", "turbine_id_column": "turbine_id",
                "power_unit": "kW", "simultaneous_observations_verified": True,
                "turbine_coverage_verified": True, "reference": "source sampling documentation",
            })
            pd.DataFrame({
                "day_index": [0, 0], "time_of_day": ["01:00", "01:20"],
                "turbine_id": ["T1", "T2"], "wind_speed_m_s": [1., 3.], "wind_power_kw": [1., 3.],
            }).to_csv(Path(tmp) / "wind.csv", index=False)
            with self.assertRaisesRegex(engine.StageBlocked, "different times inside one output-frequency bucket"):
                engine.wind_power(cfg, FakeRegressor(features), None, features)

    def test_wind_sum_blocks_duplicate_turbine_identity_across_chunks(self):
        features = ["wind_speed_m_s", "Hour", "Minute", "Month"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["wind_dataset_file"] = "wind.csv"
            cfg["wind_chunksize"] = 1
            cfg["alignment"]["frequency"] = "10min"
            cfg["alignment"]["wind_aggregation"] = "sum"
            cfg["alignment"]["wind_aggregation_semantics_verified"] = True
            cfg["alignment_verification_reference"]["wind_aggregation_semantics_verified"] = "turbine sum docs"
            cfg["wind_aggregation_evidence"].update({
                "target_semantics": "per_turbine_power", "turbine_id_column": "turbine_id",
                "power_unit": "kW", "simultaneous_observations_verified": True,
                "turbine_coverage_verified": True, "reference": "source sampling documentation",
            })
            pd.DataFrame({
                "day_index": [0, 0], "time_of_day": ["01:00", "01:00"],
                "turbine_id": ["T1", "T1"], "wind_speed_m_s": [1., 2.], "wind_power_kw": [1., 2.],
            }).to_csv(Path(tmp) / "wind.csv", index=False)
            with self.assertRaisesRegex(engine.StageBlocked, "turbine repeated across chunks"):
                engine.wind_power(cfg, FakeRegressor(features), None, features)

    def test_alignment_blocks_timezone_aware_naive_mismatch(self):
        cfg = small_cfg()
        cfg["alignment"].update({
            "timestamps_compatible_verified": True, "solar_power_semantics_verified": True,
            "wind_power_semantics_verified": True, "wind_aggregation_semantics_verified": True,
            "timestamp_timezone_mode": "naive_same_timezone",
        })
        for key in ("timestamps_compatible_verified", "solar_power_semantics_verified", "wind_power_semantics_verified", "wind_aggregation_semantics_verified"):
            cfg["alignment_verification_reference"][key] = "source documentation"
        cfg["alignment_verification_reference"]["timestamp_timezone_mode"] = "source timezone documentation"
        solar = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "predicted_solar_power_kw": [1.]})
        wind = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")], "predicted_wind_power_kw": [2.], "wind_output_scope": ["farm_total"]})
        with self.assertRaisesRegex(engine.StageBlocked, "timezone-aware"):
            engine.align_renewables(cfg, solar, wind)

    def test_alignment_reports_invalid_timestamp_rows(self):
        cfg = small_cfg()
        cfg["alignment"].update({
            "timestamps_compatible_verified": True, "solar_power_semantics_verified": True,
            "wind_power_semantics_verified": True, "wind_aggregation_semantics_verified": True,
            "timestamp_timezone_mode": "naive_same_timezone",
        })
        cfg["alignment_verification_reference"].update({
            "timestamps_compatible_verified": "source documentation", "solar_power_semantics_verified": "solar target docs",
            "wind_power_semantics_verified": "wind target docs", "wind_aggregation_semantics_verified": "wind aggregation docs",
            "timestamp_timezone_mode": "same local timezone docs",
        })
        solar = pd.DataFrame({"timestamp": ["2024-01-01 00:00", "bad"], "predicted_solar_power_kw": [1., 99.]})
        wind = pd.DataFrame({"timestamp": ["2024-01-01 00:00"], "predicted_wind_power_kw": [2.], "wind_output_scope": ["farm_total"]})
        result = engine.align_renewables(cfg, solar, wind)
        self.assertEqual(result.loc[0, "solar_invalid_timestamp_rows"], 1)

    def test_approved_ranges_require_metadata_and_do_not_modify_observed_candidates(self):
        data = pd.DataFrame({
            "voltage_v": [1., 2., 3.], "temperature_c": [10., 20., 30.],
            "hydrogen_production_kg_h": [0.1, 0.2, 0.3],
        })
        cfg = small_cfg()
        cfg["approved_ranges"] = {"voltage_v": {"enabled": True, "min": 1., "max": 3., "units": "V"}}
        with self.assertRaisesRegex(engine.ConfigurationError, "approved_by"):
            engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])
        cfg["approved_ranges"]["voltage_v"].update({"approved_by": "Reviewer", "approval_reference": "approval-123"})
        candidates = engine.generate_candidates(data, cfg, ["voltage_v", "temperature_c"])
        self.assertTrue(candidates["approved_range_voltage_v_pass"].all())
        self.assertEqual(set(candidates["voltage_v"]), {1., 2., 3.})

    def test_native_pass_fail_labels_still_require_training_semantics_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["fuel_cell_dataset_file"] = "fuel.csv"
            pd.DataFrame({"hydrogen_flow_l_min": [1.], "voltage_v": [10.], "temperature_c": [50.], "pressure_bar": [2.]}).to_csv(Path(tmp) / "fuel.csv", index=False)
            result, evidence = engine.fuel_cell_diagnostics(
                cfg, FakeClassifier(), None,
                ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"],
            )
            self.assertEqual(evidence["status"], "BLOCKED_CLASS_MAPPING_UNVERIFIED")
            cfg["fuel_cell_pass_constraint"].update({
                "class_label_mapping": {"Fail": "Fail", "Pass": "Pass"},
                "class_mapping_verified": True,
                "class_mapping_reference": "original training target and source data documentation",
            })
            result, evidence = engine.fuel_cell_diagnostics(
                cfg, FakeClassifier(), None,
                ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"],
            )
            self.assertEqual(evidence["status"], "PASS")
            self.assertTrue(result["probabilities_validated"].all())
            np.testing.assert_allclose(result["probability_row_sum"], 1.0)

    def test_fuel_cell_incomplete_ambiguous_and_mismatched_mappings_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["fuel_cell_dataset_file"] = "fuel.csv"
            pd.DataFrame({"hydrogen_flow_l_min": [1.], "voltage_v": [10.], "temperature_c": [50.], "pressure_bar": [2.]}).to_csv(Path(tmp) / "fuel.csv", index=False)
            features = ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"]
            cases = [
                ({"Pass": "Pass"}, "Pass"),
                ({"Fail": "Pass", "Pass": "Pass"}, "Pass"),
                ({"Fail": "Fail", "Pass": "Pass"}, "Maybe"),
            ]
            expected_statuses = ["BLOCKED_CLASS_MAPPING_UNVERIFIED", "BLOCKED_CLASS_MAPPING_UNVERIFIED", "BLOCKED_REQUIRED_LABEL_NOT_IN_CLASSES"]
            for (mapping, required_label), expected in zip(cases, expected_statuses):
                with self.subTest(mapping=mapping, required_label=required_label):
                    cfg["fuel_cell_pass_constraint"].update({
                        "class_label_mapping": mapping, "class_mapping_verified": True,
                        "class_mapping_reference": "training label mapping reference",
                        "required_class_label": required_label,
                    })
                    _, evidence = engine.fuel_cell_diagnostics(cfg, FakeClassifier(), None, features)
                    self.assertEqual(evidence["status"], expected)

    def test_fuel_cell_probability_shape_range_and_sum_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg["dataset_dir"] = tmp
            cfg["fuel_cell_dataset_file"] = "fuel.csv"
            cfg["fuel_cell_pass_constraint"].update({
                "class_label_mapping": {"Fail": "Fail", "Pass": "Pass"},
                "class_mapping_verified": True,
                "class_mapping_reference": "training label mapping reference",
            })
            pd.DataFrame({"hydrogen_flow_l_min": [1.], "voltage_v": [10.], "temperature_c": [50.], "pressure_bar": [2.]}).to_csv(Path(tmp) / "fuel.csv", index=False)
            features = ["hydrogen_flow_l_min", "voltage_v", "temperature_c", "pressure_bar"]
            for model, phrase in (
                (BadProbabilityShapeClassifier(), "dimensions"),
                (BadProbabilityRangeClassifier(), r"outside \[0, 1\]"),
                (BadProbabilitySumClassifier(), "sum to 1"),
            ):
                with self.subTest(model=type(model).__name__):
                    result, evidence = engine.fuel_cell_diagnostics(cfg, model, None, features)
                    self.assertTrue(result.empty)
                    self.assertEqual(evidence["status"], "BLOCKED_CLASSIFIER_INFERENCE")
                    self.assertRegex(evidence["details"], phrase)

    def test_atomic_csv_failure_preserves_previous_complete_output(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.csv"
            path.write_text("previous complete output", encoding="utf-8")
            with patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("simulated write failure")):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    engine.atomic_to_csv(pd.DataFrame({"a": [1]}), path)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous complete output")
            self.assertEqual([x.name for x in Path(tmp).iterdir()], ["result.csv"])

    def test_run_archives_old_outputs_and_records_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs"
            out.mkdir()
            (out / "best.csv").write_text("old", encoding="utf-8")
            run_id = "run-test-123"
            removed = engine.invalidate_stale_outputs(out, {"best": "best.csv"}, run_id=run_id)
            self.assertEqual(removed, ["best.csv"])
            self.assertFalse((out / "best.csv").exists())
            self.assertEqual((out / "_previous_runs" / run_id / "best.csv").read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main(verbosity=2)

class WindUnusedMonthRegressionTests(unittest.TestCase):
    def test_unused_month_can_predict_relative_time_without_calendar(self):
        class Booster:
            feature_names = ['Hour', 'Minute', 'Month']
            def get_score(self, importance_type='weight'):
                return {'Hour': 2, 'Minute': 1}
        class Model(FakeRegressor):
            def get_booster(self):
                return Booster()
        cfg = small_cfg()
        cfg['wind_calendar']['day_index_zero_date'] = None
        cfg['wind_calendar']['day_index_zero_date_verified'] = False
        cfg['wind_calendar']['allow_unused_month_relative_inference'] = True
        cfg['wind_target'] = 'wind_power_kw'
        with tempfile.TemporaryDirectory() as tmp:
            cfg['dataset_dir'] = tmp
            cfg['wind_dataset_file'] = 'wind.csv'
            cfg['wind_chunksize'] = 100
            pd.DataFrame({'day_index':[1,1], 'time_of_day':['01:00','01:10'], 'wind_power_kw':[2.,3.]}).to_csv(Path(tmp)/'wind.csv',index=False)
            output=engine.wind_power(cfg,Model(['Hour','Minute','Month'],value=10.),None,['Hour','Minute','Month'])
            self.assertEqual(len(output),1)
            self.assertNotIn('timestamp',output.columns)
            self.assertEqual(output['calendar_status'].iloc[0], 'RELATIVE_ONLY_MONTH_UNUSED_VERIFIED_NO_HISTORICAL_DATE')

    def test_month_used_in_split_blocks_relative_inference(self):
        class Booster:
            feature_names = ['Hour','Minute','Month']
            def get_score(self, importance_type='weight'):
                return {'Hour':2,'Month':1}
        class Model(FakeRegressor):
            def get_booster(self):
                return Booster()
        cfg=small_cfg()
        cfg['wind_calendar']['day_index_zero_date']=None
        cfg['wind_calendar']['day_index_zero_date_verified']=False
        cfg['wind_calendar']['allow_unused_month_relative_inference']=True
        with tempfile.TemporaryDirectory() as tmp:
            cfg['dataset_dir'] = tmp
            cfg['wind_dataset_file'] = 'wind.csv'
            cfg['wind_chunksize'] = 100
            pd.DataFrame({'day_index':[1], 'time_of_day':['01:00'], 'wind_power_kw':[2.]}).to_csv(Path(tmp)/'wind.csv',index=False)
            with self.assertRaisesRegex(engine.StageBlocked,'zero tree splits'):
                engine.wind_power(cfg,Model(['Hour','Minute','Month']),None,['Hour','Minute','Month'])

class WindRawDiagnosticPreservationTests(unittest.TestCase):
    def test_negative_and_positive_predictions_preserved_without_approval(self):
        class Booster:
            feature_names = ['Hour', 'Minute', 'Month']
            def get_score(self, importance_type='weight'):
                return {'Hour': 1}
        class MixedModel(FakeRegressor):
            def __init__(self):
                super().__init__(['Hour', 'Minute', 'Month'])
            def get_booster(self):
                return Booster()
            def predict(self, X):
                return np.asarray([-7.5, 8.25])
        cfg = small_cfg()
        cfg['wind_calendar']['day_index_zero_date'] = None
        cfg['wind_calendar']['day_index_zero_date_verified'] = False
        cfg['wind_calendar']['allow_unused_month_relative_inference'] = True
        with tempfile.TemporaryDirectory() as tmp:
            cfg['dataset_dir'] = tmp
            cfg['output_dir'] = str(Path(tmp) / 'outputs')
            cfg['wind_dataset_file'] = 'wind.csv'
            cfg['wind_chunksize'] = 100
            pd.DataFrame({
                'day_index': [1, 1], 'time_of_day': ['01:00', '01:10'],
                'wind_power_kw': [0., 12.],
            }).to_csv(Path(tmp) / 'wind.csv', index=False)
            with self.assertRaisesRegex(engine.StageBlocked, '2 valid-row predictions'):
                engine.wind_power(cfg, MixedModel(), None, ['Hour', 'Minute', 'Month'])
            raw = pd.read_csv(Path(tmp) / 'outputs' / 'wind_raw_predictions_diagnostic.csv')
            self.assertEqual(raw['raw_model_prediction_kw'].tolist(), [-7.5, 8.25])
            self.assertEqual(raw['negative_prediction'].tolist(), [True, False])
            self.assertEqual(raw['approval_status'].unique().tolist(), ['DIAGNOSTIC_ONLY_NOT_APPROVED'])

class WindPerTurbineDiagnosticTests(unittest.TestCase):
    def test_per_turbine_raw_predictions_kept_and_not_approved(self):
        class Booster:
            feature_names = ['Hour', 'Minute', 'Month']
            def get_score(self, importance_type='weight'):
                return {'Hour': 1}
        class Model(FakeRegressor):
            def __init__(self):
                super().__init__(['Hour', 'Minute', 'Month'])
            def get_booster(self):
                return Booster()
            def predict(self, X):
                return np.array([-2., 3., -4.])
        cfg = small_cfg()
        cfg['wind_calendar']['day_index_zero_date'] = None
        cfg['wind_calendar']['day_index_zero_date_verified'] = False
        cfg['wind_calendar']['allow_unused_month_relative_inference'] = True
        with tempfile.TemporaryDirectory() as tmp:
            cfg['dataset_dir'] = tmp
            cfg['output_dir'] = str(Path(tmp) / 'outputs')
            cfg['wind_dataset_file'] = 'wind.csv'
            cfg['wind_chunksize'] = 100
            pd.DataFrame({'day_index':[1,1,1], 'time_of_day':['01:00','01:00','01:10'],
                          'turbine_id':['A','B','A'], 'wind_power_kw':[0.,2.,0.]}).to_csv(Path(tmp)/'wind.csv',index=False)
            with self.assertRaises(engine.StageBlocked):
                engine.wind_power(cfg, Model(), None, ['Hour','Minute','Month'])
            raw = pd.read_csv(Path(tmp)/'outputs'/'wind_raw_predictions_diagnostic.csv')
            self.assertEqual(raw['turbine_id'].tolist(), ['A','B','A'])
            self.assertEqual(raw['raw_model_prediction_kw'].tolist(), [-2.,3.,-4.])
            summary = pd.read_csv(Path(tmp)/'outputs'/'wind_per_turbine_diagnostic.csv').set_index('turbine_id')
            self.assertEqual(int(summary.loc['A','negative_prediction_rows']),2)
            self.assertEqual(int(summary.loc['B','negative_prediction_rows']),0)
            self.assertEqual(summary.loc['A','approval_status'], 'DIAGNOSTIC_ONLY_NOT_APPROVED')
