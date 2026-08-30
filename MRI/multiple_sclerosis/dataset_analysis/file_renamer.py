#!/usr/bin/env python3

"""
this script analyses DICOM files in patient folders and renames them based on metadata from their headers 
format used is {PatientID}_{ProtocolName}_{SeriesNumber}_{InstanceNumber}

features:
analyses DICOM headers for metadata extraction
handles missing/invalid DICOM tags gracefully
prevents filename conflicts by automatic numbering
creates detailed reports of renaming operations
supports dry-run mode for testing
multi threaded processing for large datasets
preserves original files 

"""
import os
import sys
import shutil
from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError
import logging
import time
import re
from collections import defaultdict, Counter
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

#Logging configuration

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    #log directory create
    Path("logs").mkdir(exist_ok=True)

    #setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/dicom_renaming.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    #reduce pydicom logging noise
    logging.getLogger('pydicom').setLevel(logging.WARNING)
    return logging.getLogger(__name__)

#dicom file renamer class
class DicomFileRenamer:
    def __init__(self, patient_folders_dir, dry_run = False, num_threads = 4):
        self.patient_folders_dir = Path(patient_folders_dir)
        self.dry_run = dry_run
        self.num_threads = num_threads

        self.total_files = 0
        self.processed_files = 0
        self.renamed_files = 0
        self.error_files = 0
        self.skipped_files = 0

        self.patient_analysis = {}
        self.protocol_stats = {}
        self.series_stats = {}
        self.error_log = []


        self.logger = logging.getLogger(__name__)

        #Validate input directory
        if not self.patient_folders_dir.exists():
            raise FileNotFoundError(f"Patient")



