import pandas as pd
import dill
import numpy as np


if __name__ == "__main__":
    # Read the two tables
    raw_hps = pd.read_csv("src/pretraining/input/raw/MEDI_HPS.csv")
    raw_all = pd.read_csv("src/pretraining/input/raw/MEDI_ATC.csv")
    
    # Drop irrelevant columns
    raw_hps = raw_hps[["RXCUI_IN", "ICD9"]]
    raw_all = raw_all[["RXCUI_IN", "ATC", "CODE"]]
    
    # Drop NA values
    raw_all.dropna(subset=["ATC"],
                   inplace=True)
    
    # Rename columns
    raw_all.rename(columns={"CODE": "ICD9"}, inplace=True)
    
    # Get the unique pairs
    hps_keys = raw_hps[["RXCUI_IN", "ICD9"]].drop_duplicates()
    
    # Merge the tables to get ATC-to-ICD
    atc_hps = raw_all.merge(hps_keys, on=["RXCUI_IN", "ICD9"], how="inner")
    
    # Load the code2id dictionaries
    icd2id = dill.load(open("src/pretraining/input/processed/diag_code2index.pkl", 'rb'))
    atc2id = dill.load(open("src/pretraining/input/processed/atc_code2index.pkl", 'rb'))
    
    # Map medical codes to ID
    atc_hps["MED_ID"] = atc_hps["ATC"].map(atc2id)
    atc_hps["DIAG_ID"] = atc_hps["ICD9"].map(icd2id) 
    
    # Drop NA values
    atc_hps.dropna(subset=["MED_ID", "DIAG_ID"], inplace=True)
    atc_hps = atc_hps[["MED_ID", "DIAG_ID"]]
    
    # Get the unique medication indication pairs 
    atc_pairs = [[int(med), int(diag)] for med, diag in zip(atc_hps["MED_ID"], atc_hps["DIAG_ID"])]
    
    # Serialise them to a file
    dill.dump(atc_pairs, open("src/pretraining/input/processed/medi_drug_diag_pairs.pkl", "wb"))
    
    # Build a graph adj matrix
    indication_adj = np.zeros((len(atc2id), len(icd2id)))
    
    for med, diag in atc_pairs:
        indication_adj[int(med), int(diag)] = 1
    
    dill.dump(indication_adj, open("src/pretraining/input/processed/indication_adj.pkl", "wb"))
    