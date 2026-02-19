"""
Label Generation for MIMIC-III ICU Prediction Tasks
Generates outcome labels for mortality, sepsis, AKI, hypotension, vasopressor, and ventilation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import timedelta
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate prediction labels for ICU outcomes"""
    
    def __init__(self, config_path: str):
        """
        Initialize label generator
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = self._load_config(config_path)
        self.time_windows = self.config.get('TIME_WINDOWS', {})
        self.sepsis_criteria = self.config.get('SEPSIS_CRITERIA', {})
        self.aki_criteria = self.config.get('AKI_KDIGO_STAGES', {})
        self.hypotension_threshold = self.config.get('HYPOTENSION_THRESHOLD', 65)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_mortality_labels(self,
                                  icu_data: pd.DataFrame,
                                  current_time: pd.Timestamp,
                                  window_hours: int = 24) -> int:
        """
        Generate mortality label for a specific time point
        
        Args:
            icu_data: Row from merged ICU dataset with patient info
            current_time: Current prediction time
            window_hours: Prediction window (6, 12, or 24 hours)
            
        Returns:
            1 if patient dies within window_hours, 0 otherwise
        """
        if pd.isna(icu_data['dod']):
            return 0
        
        time_to_death = (icu_data['dod'] - current_time).total_seconds() / 3600
        
        # Return 1 if death occurs within the window
        if 0 <= time_to_death <= window_hours:
            return 1
        return 0
    
    def generate_sepsis_labels(self,
                              vitals: pd.DataFrame,
                              labs: pd.DataFrame,
                              prescriptions: pd.DataFrame,
                              diagnoses: pd.DataFrame,
                              current_time: pd.Timestamp,
                              window_hours: int = 6) -> int:
        """
        Generate sepsis onset label using SIRS criteria + infection markers
        
        Sepsis = SIRS (≥2 criteria) + Suspected/Confirmed Infection
        SIRS criteria:
        - Temperature > 38.3°C or < 36°C
        - Heart rate > 90 bpm
        - Respiratory rate > 20
        - WBC > 12,000 or < 4,000
        
        Args:
            vitals: Vital signs DataFrame
            labs: Lab tests DataFrame
            prescriptions: Medications DataFrame
            diagnoses: ICD diagnosis codes
            current_time: Current prediction time
            window_hours: Prediction window
            
        Returns:
            1 if sepsis onset within window, 0 otherwise
        """
        # Get future window
        window_end = current_time + timedelta(hours=window_hours)
        
        # Filter vitals and labs to future window
        future_vitals = vitals[
            (vitals.index > current_time) & (vitals.index <= window_end)
        ]
        future_labs = labs[
            (labs.index > current_time) & (labs.index <= window_end)
        ]
        
        if len(future_vitals) == 0:
            return 0
        
        # Combine for SIRS calculation
        combined = pd.concat([future_vitals, future_labs], axis=1)
        
        # Check SIRS criteria at each timepoint
        sirs_count = pd.Series(0, index=combined.index)
        
        # Temperature
        if 'tempc' in combined.columns:
            temp_abnormal = (combined['tempc'] > 38.3) | (combined['tempc'] < 36.0)
            sirs_count += temp_abnormal.fillna(False).astype(int)
        
        # Heart rate
        if 'heartrate' in combined.columns:
            hr_abnormal = combined['heartrate'] > 90
            sirs_count += hr_abnormal.fillna(False).astype(int)
        
        # Respiratory rate
        if 'resprate' in combined.columns:
            rr_abnormal = combined['resprate'] > 20
            sirs_count += rr_abnormal.fillna(False).astype(int)
        
        # WBC (convert from K to absolute)
        if 'wbc' in combined.columns:
            wbc_abnormal = (combined['wbc'] > 12) | (combined['wbc'] < 4)
            sirs_count += wbc_abnormal.fillna(False).astype(int)
        
        # SIRS positive if ≥2 criteria
        sirs_positive = sirs_count >= 2
        
        if not sirs_positive.any():
            return 0
        
        # Check for infection markers
        # 1. Antibiotic prescriptions
        antibiotics_given = False
        if prescriptions is not None and len(prescriptions) > 0:
            antibiotic_keywords = ['cillin', 'mycin', 'cycline', 'cephalosporin', 
                                  'quinolone', 'vancomycin', 'meropenem', 'azithromycin']
            future_meds = prescriptions[
                (prescriptions['startdate'] > current_time) & 
                (prescriptions['startdate'] <= window_end)
            ]
            if len(future_meds) > 0:
                for keyword in antibiotic_keywords:
                    if future_meds['drug'].str.contains(keyword, case=False, na=False).any():
                        antibiotics_given = True
                        break
        
        # 2. Sepsis-related ICD codes
        sepsis_diagnosis = False
        if diagnoses is not None and len(diagnoses) > 0:
            sepsis_codes = ['038', '995.91', '995.92', '785.52']  # Sepsis ICD-9 codes
            for code in sepsis_codes:
                if diagnoses['icd9_code'].str.startswith(code, na=False).any():
                    sepsis_diagnosis = True
                    break
        
        # Sepsis = SIRS + infection evidence
        if sirs_positive.any() and (antibiotics_given or sepsis_diagnosis):
            return 1
        
        return 0
    
    def generate_aki_labels(self,
                           labs: pd.DataFrame,
                           current_time: pd.Timestamp,
                           window_hours: int = 24) -> Dict[str, int]:
        """
        Generate AKI (Acute Kidney Injury) labels using KDIGO criteria
        
        KDIGO Staging:
        - Stage 1: Creatinine increase ≥0.3 mg/dL or 1.5× baseline
        - Stage 2: 2.0× baseline
        - Stage 3: 3.0× baseline or creatinine >4.0 mg/dL
        
        Args:
            labs: Lab tests DataFrame
            current_time: Current prediction time
            window_hours: Prediction window
            
        Returns:
            Dict with aki_stage1, aki_stage2, aki_stage3 binary labels
        """
        labels = {
            'aki_stage1': 0,
            'aki_stage2': 0,
            'aki_stage3': 0
        }
        
        if 'creatinine' not in labs.columns:
            return labels
        
        # Get baseline creatinine (lowest in past 48 hours or current)
        lookback_time = current_time - timedelta(hours=48)
        past_cr = labs[
            (labs.index >= lookback_time) & (labs.index <= current_time)
        ]['creatinine'].dropna()
        
        if len(past_cr) == 0:
            return labels
        
        baseline_cr = past_cr.min()
        
        # Get future creatinine values
        window_end = current_time + timedelta(hours=window_hours)
        future_cr = labs[
            (labs.index > current_time) & (labs.index <= window_end)
        ]['creatinine'].dropna()
        
        if len(future_cr) == 0:
            return labels
        
        max_future_cr = future_cr.max()
        cr_increase = max_future_cr - baseline_cr
        
        # Stage 1: Increase ≥0.3 or 1.5× baseline
        if cr_increase >= 0.3 or max_future_cr >= 1.5 * baseline_cr:
            labels['aki_stage1'] = 1
        
        # Stage 2: 2.0× baseline
        if max_future_cr >= 2.0 * baseline_cr:
            labels['aki_stage2'] = 1
        
        # Stage 3: 3.0× baseline or >4.0
        if max_future_cr >= 3.0 * baseline_cr or max_future_cr > 4.0:
            labels['aki_stage3'] = 1
        
        return labels
    
    def generate_hypotension_labels(self,
                                   vitals: pd.DataFrame,
                                   current_time: pd.Timestamp,
                                   window_hours: int = 3) -> int:
        """
        Generate hypotension/shock label
        
        Hypotension: Mean Arterial Pressure (MAP) < 65 mmHg
        
        Args:
            vitals: Vital signs DataFrame
            current_time: Current prediction time
            window_hours: Prediction window (1, 3, or 6 hours)
            
        Returns:
            1 if hypotension occurs within window, 0 otherwise
        """
        if 'meanbp' not in vitals.columns:
            return 0
        
        # Get future vitals
        window_end = current_time + timedelta(hours=window_hours)
        future_vitals = vitals[
            (vitals.index > current_time) & (vitals.index <= window_end)
        ]
        
        if len(future_vitals) == 0:
            return 0
        
        # Check for MAP < threshold
        hypotensive = future_vitals['meanbp'] < self.hypotension_threshold
        
        if hypotensive.any():
            return 1
        return 0
    
    def generate_vasopressor_labels(self,
                                   prescriptions: pd.DataFrame,
                                   current_time: pd.Timestamp,
                                   window_hours: int = 6) -> int:
        """
        Generate vasopressor requirement label
        
        Vasopressors: norepinephrine, epinephrine, vasopressin, dopamine, dobutamine
        
        Args:
            prescriptions: Medications DataFrame
            current_time: Current prediction time
            window_hours: Prediction window
            
        Returns:
            1 if vasopressor started within window, 0 otherwise
        """
        if prescriptions is None or len(prescriptions) == 0:
            return 0
        
        # Get future prescriptions
        window_end = current_time + timedelta(hours=window_hours)
        future_meds = prescriptions[
            (prescriptions['startdate'] > current_time) &
            (prescriptions['startdate'] <= window_end)
        ]
        
        if len(future_meds) == 0:
            return 0
        
        # Check for vasopressors
        vasopressor_keywords = ['norepinephrine', 'epinephrine', 'vasopressin', 
                               'dopamine', 'dobutamine']
        
        for keyword in vasopressor_keywords:
            if future_meds['drug'].str.contains(keyword, case=False, na=False).any():
                return 1
        
        return 0
    
    def generate_ventilation_labels(self,
                                   chartevents: pd.DataFrame,
                                   procedures: pd.DataFrame,
                                   current_time: pd.Timestamp,
                                   window_hours: int = 12) -> int:
        """
        Generate mechanical ventilation requirement label
        
        Args:
            chartevents: Chart events (contains ventilation settings)
            procedures: Procedures (contains intubation codes)
            current_time: Current prediction time
            window_hours: Prediction window
            
        Returns:
            1 if mechanical ventilation within window, 0 otherwise
        """
        # This is a simplified version - in real implementation would check:
        # - Ventilator settings in chartevents
        # - Intubation procedure codes
        # - Ventilation mode changes
        
        # For now, return 0 as placeholder
        # Would need specific itemids for ventilation from D_ITEMS
        return 0
    
    def generate_all_labels(self,
                           icu_stay_data: pd.DataFrame,
                           vitals: pd.DataFrame,
                           labs: pd.DataFrame,
                           prescriptions: pd.DataFrame,
                           diagnoses: pd.DataFrame,
                           current_time: pd.Timestamp) -> Dict[str, int]:
        """
        Generate all prediction labels for a single timepoint
        
        Args:
            icu_stay_data: ICU stay metadata (from merged dataframe)
            vitals: Vital signs time-series
            labs: Lab tests time-series
            prescriptions: Prescriptions data
            diagnoses: Diagnosis codes
            current_time: Current prediction time
            
        Returns:
            Dictionary with all binary labels
        """
        labels = {}
        
        # Mortality labels (multiple windows)
        for window in self.time_windows.get('mortality', [6, 12, 24]):
            label_name = f'mortality_{window}h'
            labels[label_name] = self.generate_mortality_labels(
                icu_stay_data, current_time, window
            )
        
        # Sepsis labels
        for window in self.time_windows.get('sepsis', [6, 12, 24]):
            label_name = f'sepsis_{window}h'
            labels[label_name] = self.generate_sepsis_labels(
                vitals, labs, prescriptions, diagnoses, current_time, window
            )
        
        # AKI labels
        for window in self.time_windows.get('aki', [24, 48]):
            aki_labels = self.generate_aki_labels(labs, current_time, window)
            for stage, value in aki_labels.items():
                labels[f'{stage}_{window}h'] = value
        
        # Hypotension labels
        for window in self.time_windows.get('hypotension', [1, 3, 6]):
            label_name = f'hypotension_{window}h'
            labels[label_name] = self.generate_hypotension_labels(
                vitals, current_time, window
            )
        
        # Vasopressor labels
        for window in self.time_windows.get('vasopressor', [6, 12]):
            label_name = f'vasopressor_{window}h'
            labels[label_name] = self.generate_vasopressor_labels(
                prescriptions, current_time, window
            )
        
        # Ventilation labels
        for window in self.time_windows.get('ventilation', [6, 12, 24]):
            label_name = f'ventilation_{window}h'
            labels[label_name] = 0  # Placeholder
        
        return labels


if __name__ == "__main__":
    # Test label generation
    from data_loader import MIMICDataLoader
    from feature_engineering import FeatureEngineer
    
    logger.info("Testing Label Generation...")
    
    # Load data
    loader = MIMICDataLoader('demo', 'config.yaml')
    merged = loader.merge_data()
    
    # Initialize
    fe = FeatureEngineer('config.yaml')
    lg = LabelGenerator('config.yaml')
    
    # Test on first ICU stay
    first_stay = merged.iloc[0]
    icustay_id = first_stay['icustay_id']
    
    logger.info(f"\nGenerating labels for ICU stay {icustay_id}...")
    
    # Extract features
    features = fe.extract_features_for_stay(
        icustay_id=icustay_id,
        icu_intime=first_stay['intime'],
        icu_outtime=first_stay['outtime'],
        chartevents=loader.chartevents,
        labevents=loader.labevents,
        d_items=loader.d_items,
        d_labitems=loader.d_labitems,
        window_hours=6
    )
    
    if len(features) > 0:
        # Split features into vitals and labs
        vital_cols = [c for c in features.columns if any(v in c for v in ['heartrate', 'bp', 'temp', 'spo2', 'resp', 'glucose'])]
        lab_cols = [c for c in features.columns if any(l in c for l in ['lactate', 'creatinine', 'wbc', 'hemoglobin', 'platelet'])]
        
        vitals = features[vital_cols] if vital_cols else pd.DataFrame()
        labs = features[lab_cols] if lab_cols else pd.DataFrame()
        
        # Generate labels at ICU admission + 12 hours
        prediction_time = first_stay['intime'] + timedelta(hours=12)
        
        # Get prescriptions and diagnoses for this stay
        stay_prescriptions = loader.prescriptions[
            loader.prescriptions['icustay_id'] == icustay_id
        ] if 'icustay_id' in loader.prescriptions.columns else pd.DataFrame()
        
        stay_diagnoses = loader.diagnoses[
            loader.diagnoses['hadm_id'] == first_stay['hadm_id']
        ]
        
        labels = lg.generate_all_labels(
            icu_stay_data=first_stay,
            vitals=vitals,
            labs=labs,
            prescriptions=stay_prescriptions,
            diagnoses=stay_diagnoses,
            current_time=prediction_time
        )
        
        print(f"\n=== Label Generation Test ===")
        print(f"Prediction time: {prediction_time}")
        print(f"\nGenerated {len(labels)} labels:")
        for label_name, value in labels.items():
            print(f"  {label_name}: {value}")
    else:
        print("No features available for label generation test")
