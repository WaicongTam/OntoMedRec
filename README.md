# OntoMedRec
The model repository of OntoMedRec. To run the model, please make sure you have the following packages:

* PyTorch >= 1.12.0
* LTNTorch
* Numpy
* Pandas
* sklearn

# Data Preprocessing
To get access to MIMIC dataset, please follow the instruction on https://physionet.org/content/mimiciii/1.4/
Download the raw DDI dataset as per the instruction in https://github.com/ycq091044/SafeDrug
Please put the diagnoses, procedures, prescriptions and admissions datasets to the `./data` directory.
To get the training, testing and validation records, please and run the following commands:
```
cd data
python processing.py
```

# Running the fine-tuning for OntoMedRec and other baselines
We prepared the pretrained embeddings of OntoMedRec and other baselines. To fine-tune the model, you can simply run the following command
```
python src/downstream/MICRON.py --embd_mode omr --pro_taxo --epochs 60
```
Other options of the `--embd_mode` parameters are `gat`, `gcn` and `random`. `--pro_taxo` parameter means using the OntoMedRec representation for procedures.

To test the performance of the fine-tuned model, simply add `--test` to the above command.

```
python src/downstream/MICRON.py --embd_mode omr --pro_taxo --test --epochs 60
```
# Acknowledgement
We would like to express our sincere gratitute to the repective authors of the following papers their code base:

* https://github.com/BarryRun/COGNet
* https://github.com/ycq091044/SafeDrug
* https://github.com/Melinda315/4SDrug
