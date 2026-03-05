"""
Composite Predictor — Task 22
Produces a unified deterioration score combining all prediction tasks.
Uses MultitaskLSTM with shared encoder + task-specific heads.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class CompositePredictor(BasePredictor):
    """
    Composite Deterioration Score

    Architecture:
        Shared LSTM encoder → 21 task-specific heads → individual predictions
        Extra head → composite deterioration score (0-1)

    This predictor trains on ALL labels simultaneously.
    It doesn't generate its own labels — it uses labels from all other predictors.

    Best model: MultitaskLSTM (designed for this purpose)
    """

    TASK_NAME = "composite"
    TASK_DESCRIPTION = "Unified deterioration score combining all prediction tasks"
    WINDOWS = []  # No windows — uses all labels
    LABEL_PREFIX = "composite"
    MODELS_TO_TRY = ['multitask_lstm']  # Only MultitaskLSTM makes sense here

    def get_label_names(self) -> List[str]:
        """Composite uses ALL labels from other predictors."""
        # This will be set dynamically by the pipeline
        if hasattr(self, '_all_label_names') and self._all_label_names:
            return self._all_label_names
        # Default fallback
        return ['composite_score']

    def set_all_labels(self, all_label_names: List[str]):
        """Set the full list of labels this predictor will train on."""
        self._all_label_names = all_label_names

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        """Composite doesn't generate its own labels — returns empty dict."""
        return {}

    def train_all_models(self, X, y, timestamps, output_dir='output'):
        """
        Train MultitaskLSTM on ALL labels simultaneously.
        The model's final head produces the composite score.
        """
        import os
        import json
        os.makedirs(output_dir, exist_ok=True)

        num_tasks = y.shape[1]
        input_size = X.shape[2]

        logger.info(f"[composite] Training MultitaskLSTM on {num_tasks} tasks...")

        config = self.config.copy()
        config['input_size'] = input_size
        config['num_tasks'] = num_tasks + 1  # +1 for composite score head

        try:
            metrics = self._train_dl_model('multitask_lstm', X, y, timestamps, config)
            self.best_model_name = 'multitask_lstm'
            self.best_auroc = metrics.get('mean_test_auroc', 0)

            self.results = {
                'task': self.TASK_NAME,
                'description': self.TASK_DESCRIPTION,
                'best_model': 'multitask_lstm',
                'best_auroc': self.best_auroc,
                'num_tasks_combined': num_tasks,
                'comparison': {'multitask_lstm': metrics},
            }

            report_path = os.path.join(output_dir, 'composite_report.json')
            with open(report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"[composite] MultitaskLSTM AUROC: {self.best_auroc:.4f}")

        except Exception as e:
            logger.error(f"[composite] Training failed: {e}")
            self.results = {'error': str(e)}

        return self.results


if __name__ == "__main__":
    p = CompositePredictor()
    print(p)
