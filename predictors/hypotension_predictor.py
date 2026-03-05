"""
Hypotension Predictor — Tasks 13-15
Predicts hypotension/shock (MAP < 65 mmHg) at 1, 3, and 6 hour horizons.
"""

import pandas as pd
from typing import Dict
from datetime import timedelta
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class HypotensionPredictor(BasePredictor):
    """
    Hypotension / Shock Prediction

    Clinical definition:
        Hypotension = Mean Arterial Pressure (MAP) < 65 mmHg
        MAP = (Systolic + 2 × Diastolic) / 3

    Logic:
        IF any meanbp reading < 65 in the future window → label = 1
        ELSE → label = 0

    Constraint:
        Even a single MAP reading below threshold triggers the label.
        If no future vitals exist → returns 0.

    Best model: TCN (short-term temporal patterns) or XGBoost
    """

    TASK_NAME = "hypotension"
    TASK_DESCRIPTION = "Hypotension/shock prediction (MAP < 65 mmHg) at 1/3/6h"
    WINDOWS = [1, 3, 6]
    LABEL_PREFIX = "hypotension"

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        self.threshold = self.config.get('HYPOTENSION_THRESHOLD', 65)

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        labels = {}
        for window in self.WINDOWS:
            label_name = f'{self.LABEL_PREFIX}_{window}h'
            labels[label_name] = self._check_hypotension(vitals, current_time, window)
        return labels

    def _check_hypotension(self, vitals, current_time, window_hours) -> int:
        if len(vitals) == 0 or 'meanbp' not in vitals.columns:
            return 0

        window_end = current_time + timedelta(hours=window_hours)
        future = vitals[
            (vitals.index > current_time) & (vitals.index <= window_end)
        ]

        if len(future) == 0:
            return 0

        return 1 if (future['meanbp'] < self.threshold).any() else 0


if __name__ == "__main__":
    p = HypotensionPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
    print(f"MAP threshold: {p.threshold} mmHg")
