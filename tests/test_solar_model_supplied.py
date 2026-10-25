"""Regression guard for the supplied Solar CatBoost artifact (not physical validation)."""
import unittest
from pathlib import Path
import joblib
import numpy as np
from prescriptive_engine.engine import read_config, solar_power

ROOT = Path(__file__).resolve().parents[1]


class SuppliedSolarArtifactTests(unittest.TestCase):
    def test_saved_solar_model_predicts_without_changing_feature_contract(self):
        cfg = read_config(ROOT / 'config.json')
        model = joblib.load(ROOT / 'models' / cfg['solar_model_file'])
        self.assertEqual(type(model).__name__, 'CatBoostRegressor')
        self.assertEqual(list(model.feature_names_), cfg['solar_features'])
        predictions = solar_power(cfg, model, None, cfg['solar_features'])
        values = predictions['predicted_solar_power_kw'].to_numpy(dtype=float)
        self.assertEqual(len(values), 1745)
        self.assertTrue(np.isfinite(values).all())
        # The original regression produces some negative power predictions;
        # retaining them in diagnostics must not be misrepresented as valid supply.
        self.assertGreater(int((values < 0).sum()), 0)


if __name__ == '__main__':
    unittest.main()
