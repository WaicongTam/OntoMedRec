#!/bin/bash

python src/preprocessing/process_ontologies.py
python src/preprocessing/process_admission.py
python src/preprocessing/process_diagnoses.py
python src/preprocessing/process_procedures.py
python src/preprocessing/process_prescriptions.py