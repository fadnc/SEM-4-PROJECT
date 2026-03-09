"""
MIMIC-III Data Loader
Loads and merges MIMIC-III demo tables for ICU patient analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """Load and process MIMIC-III clinical database"""
    
    def __init__(self, data_dir: str, config_path: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to MIMIC-III data directory (e.g., 'demo/')
            config_path: Path to config.yaml file
        """
        self.data_dir = data_dir
        self.config = self._load_config(config_path)
        
        # Data containers — core tables
        self.patients = None
        self.admissions = None
        self.icu_stays = None
        self.chartevents = None
        self.labevents = None
        self.diagnoses = None
        self.prescriptions = None
        self.d_items = None
        self.d_labitems = None
        
        # Data containers — additional tables
        self.inputevents_mv = None
        self.outputevents = None
        self.procedureevents = None
        self.procedures_icd = None
        self.microbiologyevents = None
        self.transfers = None
        self.callout = None
        self.services = None
        self.d_icd_diagnoses = None
        self.d_icd_procedures = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase for compatibility with both
        full MIMIC-III (UPPERCASE) and demo data (lowercase)."""
        df.columns = [c.lower() for c in df.columns]
        return df
    
    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics"""
        logger.info("Loading PATIENTS...")
        filepath = os.path.join(self.data_dir, 'PATIENTS.csv')
        self.patients = self._normalize_columns(pd.read_csv(filepath))
        
        # Convert dates
        self.patients['dob'] = pd.to_datetime(self.patients['dob'])
        self.patients['dod'] = pd.to_datetime(self.patients['dod'])
        
        logger.info(f"Loaded {len(self.patients)} patients")
        return self.patients
    
    def load_admissions(self) -> pd.DataFrame:
        """Load hospital admissions"""
        logger.info("Loading ADMISSIONS...")
        filepath = os.path.join(self.data_dir, 'ADMISSIONS.csv')
        self.admissions = self._normalize_columns(pd.read_csv(filepath))
        
        # Convert timestamps
        time_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in time_cols:
            if col in self.admissions.columns:
                self.admissions[col] = pd.to_datetime(self.admissions[col])
        
        logger.info(f"Loaded {len(self.admissions)} admissions")
        return self.admissions
    
    def load_icu_stays(self) -> pd.DataFrame:
        """Load ICU stays"""
        logger.info("Loading ICUSTAYS...")
        filepath = os.path.join(self.data_dir, 'ICUSTAYS.csv')
        self.icu_stays = self._normalize_columns(pd.read_csv(filepath))
        
        # Convert timestamps
        self.icu_stays['intime'] = pd.to_datetime(self.icu_stays['intime'])
        self.icu_stays['outtime'] = pd.to_datetime(self.icu_stays['outtime'])
        
        logger.info(f"Loaded {len(self.icu_stays)} ICU stays")
        return self.icu_stays
    
    def load_chartevents(self, sample_n: Optional[int] = None) -> pd.DataFrame:
        """
        Load vital signs and chart events.
        
        For large files (>1GB), uses chunked loading — filters to only keep
        rows with relevant vital sign or ventilation itemids to fit in RAM.
        
        Args:
            sample_n: If provided, load only first N rows (for testing)
        """
        logger.info("Loading CHARTEVENTS...")
        filepath = os.path.join(self.data_dir, 'CHARTEVENTS.csv')
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # For small files (demo data), load directly
        if file_size_mb < 500 or sample_n:
            if sample_n:
                self.chartevents = self._normalize_columns(
                    pd.read_csv(filepath, nrows=sample_n, low_memory=False)
                )
                logger.info(f"Loaded {sample_n} chart events (sampled)")
            else:
                self.chartevents = self._normalize_columns(
                    pd.read_csv(filepath, low_memory=False)
                )
                logger.info(f"Loaded {len(self.chartevents)} chart events")
        else:
            # Large file — chunked loading with itemid filtering
            logger.info(f"CHARTEVENTS is {file_size_mb:.0f} MB — using chunked loading")
            
            # Get relevant itemids (vitals + ventilation)
            relevant_ids = self._get_relevant_chartevents_itemids()
            logger.info(f"Filtering to {len(relevant_ids)} relevant itemids")
            
            chunks = []
            total_read = 0
            chunk_size = 2_000_000

            CHARTEVENTS_DTYPES = {
                    'ROW_ID': 'int32',
                    'SUBJECT_ID': 'int32',
                    'HADM_ID': 'float32',
                    'ICUSTAY_ID': 'float32',
                    'ITEMID': 'int32',
                    'VALUENUM': 'float32',
                    'VALUEUOM': 'str',
                    'ERROR': 'float32',
                }
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=CHARTEVENTS_DTYPES, 
                    usecols= ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 
                    'ITEMID', 'CHARTTIME', 'VALUENUM'],low_memory=False):
                chunk = self._normalize_columns(chunk)
                if 'itemid' in chunk.columns:
                    filtered = chunk[chunk['itemid'].isin(relevant_ids)]
                    if len(filtered) > 0:
                        chunks.append(filtered)
                total_read += len(chunk)
                if total_read % 5_000_000 == 0:
                    kept = sum(len(c) for c in chunks)
                    logger.info(f"  Read {total_read:,} rows, kept {kept:,}")
            
            if chunks:
                self.chartevents = pd.concat(chunks, ignore_index=True)
            else:
                self.chartevents = pd.DataFrame()
            
            logger.info(f"Loaded {len(self.chartevents):,} relevant chart events "
                       f"(from {total_read:,} total rows)")
        
        # Convert timestamps and numerics
        if len(self.chartevents) > 0:
            if 'charttime' in self.chartevents.columns:
                self.chartevents['charttime'] = pd.to_datetime(self.chartevents['charttime'])
            if 'valuenum' in self.chartevents.columns:
                self.chartevents['valuenum'] = pd.to_numeric(
                    self.chartevents['valuenum'], errors='coerce'
                )
        
        return self.chartevents
    
    def _get_relevant_chartevents_itemids(self) -> set:
        """Get the set of all itemids we need from CHARTEVENTS.
        Uses specific known MIMIC-III vital sign and ventilation itemids
        rather than broad keyword matching to minimize data loaded."""
        relevant = set()
        
        # Ventilation itemids from config
        vent_ids = self.config.get('VENTILATION_ITEMIDS', [225792, 225794, 226260])
        relevant.update(vent_ids)
        
        # Specific known MIMIC-III vital sign itemids
        # MetaVision (>220000) and CareVue (<220000)
        vital_itemids = {
            # Heart Rate
            220045,  # Heart Rate (MetaVision)
            211,     # Heart Rate (CareVue)
            # Systolic BP
            220050, 220179,  # ABP sys, NBP sys (MetaVision)
            51, 455,         # ABP sys, NBP sys (CareVue)
            # Diastolic BP
            220051, 220180,  # ABP dias, NBP dias (MetaVision)
            8368, 8441,      # ABP dias, NBP dias (CareVue)
            # Mean BP
            220052, 220181,  # ABP mean, NBP mean (MetaVision)
            52, 456,         # ABP mean, NBP mean (CareVue)
            # Respiratory Rate
            220210, 224690,  # Resp Rate, Resp Rate (Total) (MetaVision)
            618, 615,        # Resp Rate (CareVue)
            # Temperature
            223761, 223762,  # Temp C, Temp F (MetaVision)
            676, 678,        # Temp C, Temp F (CareVue)
            # SpO2
            220277,  # SpO2 (MetaVision)
            646,     # SpO2 (CareVue)
            # Glucose
            220621, 226537,  # Glucose, Glucose (finger stick) (MetaVision)
            807, 811, 1529,  # Glucose (CareVue)
        }
        relevant.update(vital_itemids)
        
        return relevant
    
    def load_labevents(self) -> pd.DataFrame:
        """Load laboratory test results and assign icustay_id via time-based join"""
        logger.info("Loading LABEVENTS...")
        filepath = os.path.join(self.data_dir, 'LABEVENTS.csv')
        relevant_subjects = set(self.icu_stays['subject_id'].unique())
        chunks = []
        new_path = pd.read_csv(filepath, chunksize=1_000_000,
            dtype={'SUBJECT_ID': 'int32', 'ITEMID': 'int32', 'VALUENUM': 'float32'},
            usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'])
        for chunk in new_path:
            chunk = self._normalize_columns(chunk)
            #filter to relevant patients before expensice join
            chunk = chunk[chunk['subject_id'].isin(relevant_subjects)]
            if len(chunk) > 0:
                chunks.append(chunk)
        self.labevents = pd.concat(chunks, ignore_index=True)
        
        # Convert timestamps
        self.labevents['charttime'] = pd.to_datetime(self.labevents['charttime'])
        
        # Convert numeric values
        self.labevents['valuenum'] = pd.to_numeric(self.labevents['valuenum'], errors='coerce')
        
        logger.info(f"Loaded {len(self.labevents)} lab events")
        
        # LABEVENTS.csv has no icustay_id column — assign via ICUSTAYS join
        if self.icu_stays is not None:
            logger.info("Assigning icustay_id to lab events via hadm_id + time overlap...")
            labevents_merged = self.labevents.merge(
                self.icu_stays[['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime']],
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            # Keep only lab events that fall within an ICU stay timeframe
            mask = (
                labevents_merged['icustay_id'].notna() &
                (labevents_merged['charttime'] >= labevents_merged['intime']) &
                (labevents_merged['charttime'] <= labevents_merged['outtime'])
            )
            self.labevents = labevents_merged[mask].drop(columns=['intime', 'outtime']).copy()
            logger.info(f"Assigned icustay_id to {len(self.labevents)} lab events within ICU stays")
        else:
            logger.warning("ICU stays not loaded yet — labevents will lack icustay_id")
        
        return self.labevents
    
    def load_diagnoses(self) -> pd.DataFrame:
        """Load ICD-9 diagnosis codes"""
        logger.info("Loading DIAGNOSES_ICD...")
        filepath = os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv')
        if not os.path.exists(filepath):
            logger.warning("DIAGNOSES_ICD.csv not found, skipping")
            self.diagnoses = pd.DataFrame()
            return self.diagnoses
        self.diagnoses = self._normalize_columns(pd.read_csv(filepath))
        
        logger.info(f"Loaded {len(self.diagnoses)} diagnoses")
        return self.diagnoses
    
    def load_prescriptions(self) -> pd.DataFrame:
        """Load medication prescriptions"""
        logger.info("Loading PRESCRIPTIONS...")
        filepath = os.path.join(self.data_dir, 'PRESCRIPTIONS.csv')
        self.prescriptions = self._normalize_columns(pd.read_csv(filepath))
        
        # Convert timestamps
        time_cols = ['startdate', 'enddate']
        for col in time_cols:
            if col in self.prescriptions.columns:
                self.prescriptions[col] = pd.to_datetime(self.prescriptions[col])
        
        logger.info(f"Loaded {len(self.prescriptions)} prescriptions")
        return self.prescriptions
    
    def load_inputevents(self) -> pd.DataFrame:
        """Load IV input events (vasopressors, fluids)"""
        logger.info("Loading INPUTEVENTS_MV...")
        filepath = os.path.join(self.data_dir, 'INPUTEVENTS_MV.csv')
        if not os.path.exists(filepath):
            logger.warning("INPUTEVENTS_MV.csv not found, skipping")
            self.inputevents_mv = pd.DataFrame()
            return self.inputevents_mv
        self.inputevents_mv = self._normalize_columns(pd.read_csv(filepath))
        for col in ['starttime', 'endtime']:
            if col in self.inputevents_mv.columns:
                self.inputevents_mv[col] = pd.to_datetime(self.inputevents_mv[col])
        logger.info(f"Loaded {len(self.inputevents_mv)} input events")
        return self.inputevents_mv
    
    def load_outputevents(self) -> pd.DataFrame:
        """Load output events (urine output, drains)"""
        logger.info("Loading OUTPUTEVENTS...")
        filepath = os.path.join(self.data_dir, 'OUTPUTEVENTS.csv')
        if not os.path.exists(filepath):
            logger.warning("OUTPUTEVENTS.csv not found, skipping")
            self.outputevents = pd.DataFrame()
            return self.outputevents
        self.outputevents = self._normalize_columns(pd.read_csv(filepath))
        if 'charttime' in self.outputevents.columns:
            self.outputevents['charttime'] = pd.to_datetime(self.outputevents['charttime'])
        self.outputevents['value'] = pd.to_numeric(self.outputevents['value'], errors='coerce')
        logger.info(f"Loaded {len(self.outputevents)} output events")
        return self.outputevents

    def load_procedureevents(self) -> pd.DataFrame:
        """Load procedure events (MetaVision)"""
        logger.info("Loading PROCEDUREEVENTS_MV...")
        filepath = os.path.join(self.data_dir, 'PROCEDUREEVENTS_MV.csv')
        if not os.path.exists(filepath):
            logger.warning("PROCEDUREEVENTS_MV.csv not found, skipping")
            self.procedureevents = pd.DataFrame()
            return self.procedureevents
        self.procedureevents = self._normalize_columns(pd.read_csv(filepath))
        for col in ['starttime', 'endtime']:
            if col in self.procedureevents.columns:
                self.procedureevents[col] = pd.to_datetime(self.procedureevents[col])
        logger.info(f"Loaded {len(self.procedureevents)} procedure events")
        return self.procedureevents
    
    def load_procedures_icd(self) -> pd.DataFrame:
        """Load ICD-9 procedure codes"""
        logger.info("Loading PROCEDURES_ICD...")
        filepath = os.path.join(self.data_dir, 'PROCEDURES_ICD.csv')
        if not os.path.exists(filepath):
            logger.warning("PROCEDURES_ICD.csv not found, skipping")
            self.procedures_icd = pd.DataFrame()
            return self.procedures_icd
        self.procedures_icd = self._normalize_columns(pd.read_csv(filepath))
        logger.info(f"Loaded {len(self.procedures_icd)} procedure codes")
        return self.procedures_icd
    
    def load_microbiologyevents(self) -> pd.DataFrame:
        """Load microbiology culture results (for sepsis detection)"""
        logger.info("Loading MICROBIOLOGYEVENTS...")
        filepath = os.path.join(self.data_dir, 'MICROBIOLOGYEVENTS.csv')
        if not os.path.exists(filepath):
            logger.warning("MICROBIOLOGYEVENTS.csv not found, skipping")
            self.microbiologyevents = pd.DataFrame()
            return self.microbiologyevents
        self.microbiologyevents = self._normalize_columns(pd.read_csv(filepath))
        for col in ['chartdate', 'charttime']:
            if col in self.microbiologyevents.columns:
                self.microbiologyevents[col] = pd.to_datetime(self.microbiologyevents[col], errors='coerce')
        logger.info(f"Loaded {len(self.microbiologyevents)} microbiology events")
        return self.microbiologyevents
    
    def load_transfers(self) -> pd.DataFrame:
        """Load patient transfers (for ICU readmission detection)"""
        logger.info("Loading TRANSFERS...")
        filepath = os.path.join(self.data_dir, 'TRANSFERS.csv')
        if not os.path.exists(filepath):
            logger.warning("TRANSFERS.csv not found, skipping")
            self.transfers = pd.DataFrame()
            return self.transfers
        self.transfers = self._normalize_columns(pd.read_csv(filepath))
        for col in ['intime', 'outtime']:
            if col in self.transfers.columns:
                self.transfers[col] = pd.to_datetime(self.transfers[col])
        logger.info(f"Loaded {len(self.transfers)} transfers")
        return self.transfers
    
    def load_callout(self) -> pd.DataFrame:
        """Load discharge callout data"""
        logger.info("Loading CALLOUT...")
        filepath = os.path.join(self.data_dir, 'CALLOUT.csv')
        if not os.path.exists(filepath):
            logger.warning("CALLOUT.csv not found, skipping")
            self.callout = pd.DataFrame()
            return self.callout
        self.callout = self._normalize_columns(pd.read_csv(filepath))
        for col in ['createtime', 'updatetime', 'acknowledgetime', 'outcometime']:
            if col in self.callout.columns:
                self.callout[col] = pd.to_datetime(self.callout[col], errors='coerce')
        logger.info(f"Loaded {len(self.callout)} callout records")
        return self.callout
    
    def load_services(self) -> pd.DataFrame:
        """Load service transfers"""
        logger.info("Loading SERVICES...")
        filepath = os.path.join(self.data_dir, 'SERVICES.csv')
        if not os.path.exists(filepath):
            logger.warning("SERVICES.csv not found, skipping")
            self.services = pd.DataFrame()
            return self.services
        self.services = self._normalize_columns(pd.read_csv(filepath))
        if 'transfertime' in self.services.columns:
            self.services['transfertime'] = pd.to_datetime(self.services['transfertime'])
        logger.info(f"Loaded {len(self.services)} service records")
        return self.services
    
    def load_dictionaries(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data dictionaries for items and lab tests"""
        logger.info("Loading data dictionaries...")
        
        # D_ITEMS - chart event definitions
        filepath = os.path.join(self.data_dir, 'D_ITEMS.csv')
        self.d_items = self._normalize_columns(pd.read_csv(filepath))
        
        # D_LABITEMS - lab test definitions
        filepath = os.path.join(self.data_dir, 'D_LABITEMS.csv')
        self.d_labitems = self._normalize_columns(pd.read_csv(filepath))
        
        logger.info(f"Loaded {len(self.d_items)} item definitions, {len(self.d_labitems)} lab definitions")
        return self.d_items, self.d_labitems
    
    def get_vital_sign_itemids(self) -> Dict[str, List[int]]:
        """
        Map vital sign names to itemids using D_ITEMS dictionary
        
        Returns:
            Dictionary mapping vital names to list of itemids
        """
        if self.d_items is None:
            self.load_dictionaries()
        
        # Mapping of vital signs to search keywords
        vital_mapping = {
            'heartrate': ['heart rate', 'hr'],
            'sysbp': ['systolic', 'nbp systolic', 'arterial bp systolic'],
            'diasbp': ['diastolic', 'nbp diastolic', 'arterial bp diastolic'],
            'meanbp': ['mean', 'nbp mean', 'arterial bp mean'],
            'resprate': ['respiratory rate', 'resp rate', 'rr'],
            'tempc': ['temperature c', 'temperature celsius'],
            'spo2': ['spo2', 'o2 saturation'],
            'glucose': ['glucose', 'fingerstick glucose']
        }
        
        itemid_map = {}
        for vital_name, keywords in vital_mapping.items():
            # Search for items matching keywords
            mask = self.d_items['label'].str.lower().str.contains('|'.join(keywords), na=False, case=False)
            itemids = self.d_items[mask]['itemid'].unique().tolist()
            itemid_map[vital_name] = itemids
        
        return itemid_map
    
    def get_lab_itemids(self) -> Dict[str, List[int]]:
        """
        Map lab test names to itemids using D_LABITEMS dictionary
        
        Returns:
            Dictionary mapping lab names to list of itemids
        """
        if self.d_labitems is None:
            self.load_dictionaries()
        
        # Lab tests from config
        lab_tests = self.config.get('LAB_TESTS', [])
        
        itemid_map = {}
        for lab_name in lab_tests:
            # Search for items matching lab name
            mask = self.d_labitems['label'].str.lower().str.contains(lab_name, na=False, case=False)
            itemids = self.d_labitems[mask]['itemid'].unique().tolist()
            itemid_map[lab_name] = itemids
        
        return itemid_map
    
    def merge_data(self, load_chart_sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load and merge all relevant tables
        
        Args:
            load_chart_sample: Optional number of chart events to sample
            
        Returns:
            Merged dataframe with patients, ICU stays, and demographics
        """
        logger.info("Starting data merge pipeline...")
        
        # Load all tables
        # Order matters: dictionaries first (for chunked CHARTEVENTS filtering),
        # then ICU stays (for LABEVENTS join)
        self.load_patients()
        self.load_admissions()
        self.load_icu_stays()
        self.load_dictionaries()                    # Must be before chartevents (for itemid filtering)
        self.load_chartevents(sample_n=load_chart_sample)  # Uses D_ITEMS for chunked filtering
        self.load_labevents()                       # Requires icu_stays for icustay_id join
        self.load_diagnoses()
        self.load_prescriptions()
        
        # Load additional tables
        self.load_inputevents()
        self.load_outputevents()
        self.load_procedureevents()
        self.load_procedures_icd()
        self.load_microbiologyevents()
        self.load_transfers()
        self.load_callout()
        self.load_services()
        
        # Merge patients with ICU stays
        logger.info("Merging patients with ICU stays...")
        merged = self.icu_stays.merge(
            self.patients[['subject_id', 'gender', 'dob', 'dod', 'expire_flag']],
            on='subject_id',
            how='left'
        )
        
        # Merge with admissions
        logger.info("Merging with admissions...")
        merged = merged.merge(
            self.admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 
                           'deathtime', 'admission_type', 'diagnosis']],
            on=['subject_id', 'hadm_id'],
            how='left'
        )
        
        # Calculate age at ICU admission (handle overflow for invalid dates)
        try:
            merged['age'] = (merged['intime'] - merged['dob']).dt.total_seconds() / (365.25 * 24 * 3600)
        except OverflowError:
            # Calculate age row by row for problematic dates
            def safe_age_calc(row):
                try:
                    return (row['intime'] - row['dob']).total_seconds() / (365.25 * 24 * 3600)
                except:
                    return np.nan
            merged['age'] = merged.apply(safe_age_calc, axis=1)
        
        # Calculate time to death (if applicable)
        try:
            merged['hours_to_death'] = (merged['dod'] - merged['intime']).dt.total_seconds() / 3600
        except:
            def safe_death_calc(row):
                try:
                    if pd.isna(row['dod']):
                        return np.nan
                    return (row['dod'] - row['intime']).total_seconds() / 3600
                except:
                    return np.nan
            merged['hours_to_death'] = merged.apply(safe_death_calc, axis=1)
        
        logger.info(f"Merged dataset: {len(merged)} ICU stays")
        logger.info(f"  - Unique patients: {merged['subject_id'].nunique()}")
        logger.info(f"  - Deceased patients: {merged['expire_flag'].sum()}")
        logger.info(f"  - Mean age: {merged['age'].mean():.1f} years")
        
        return merged
    
    def get_patient_timeseries(self, icustay_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get time-series data for a specific ICU stay
        
        Args:
            icustay_id: ICU stay identifier
            
        Returns:
            Tuple of (chartevents_df, labevents_df) for this stay
        """
        charts = self.chartevents[self.chartevents['icustay_id'] == icustay_id].copy()
        labs = self.labevents[self.labevents['icustay_id'] == icustay_id].copy()
        
        # Sort by time
        charts = charts.sort_values('charttime')
        labs = labs.sort_values('charttime')
        
        return charts, labs


if __name__ == "__main__":
    # Test the data loader
    loader = MIMICDataLoader('data', 'config.yaml')
    
    # Load and merge data
    merged_data = loader.merge_data()
    
    print("\n=== Data Loader Test ===")
    print(f"Loaded {len(merged_data)} ICU stays")
    print(f"\nColumns: {list(merged_data.columns)}")
    print(f"\nFirst 3 rows:\n{merged_data.head(3)}")
    
    # Test vital sign mapping
    vital_itemids = loader.get_vital_sign_itemids()
    print(f"\n=== Vital Sign Itemids ===")
    for vital, itemids in vital_itemids.items():
        print(f"{vital}: {len(itemids)} items")
    
    # Test lab mapping
    lab_itemids = loader.get_lab_itemids()
    print(f"\n=== Lab Test Itemids ===")
    for lab, itemids in lab_itemids.items():
        print(f"{lab}: {len(itemids)} items")
