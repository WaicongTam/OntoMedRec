import dill
import torch


def read_gnn_embeddings(model, date_param):
    all_sym_embd = torch.load(f"saved/pretraining/{model}/{date_param}/sym_embd.pt")
    all_pro_embd = torch.load(f"saved/pretraining/{model}/{date_param}/pro_embd.pt")
    all_med_embd = torch.load(f"saved/pretraining/{model}/{date_param}/med_embd.pt")
    return all_sym_embd, all_pro_embd, all_med_embd

def read_expansion_embeddings(model):
    all_sym_embd = torch.load(f"src/pretraining/{model}/diag-_mainembeddings.pt")
    all_pro_embd = torch.load(f"src/pretraining/{model}/proc-_mainembeddings.pt")
    all_med_embd = torch.load(f"src/pretraining/{model}/atc-_mainembeddings.pt")
    return all_sym_embd, all_pro_embd, all_med_embd


def transfer(voc, date_param=None, model="gat"):

    med_voc = voc["med_voc"]
    pro_voc = voc["pro_voc"]
    sym_voc = voc["diag_voc"]

    med_code2idx = dill.load(open("src/pretraining/input/processed/atc_code2index.pkl", "rb"))
    sym_code2idx = dill.load(open("src/pretraining/input/processed/diag_code2index.pkl", "rb"))
    pro_code2idx = dill.load(open("src/pretraining/input/processed/proc_code2index.pkl", "rb"))
    
    sym_code2idx = {k.replace(".", ""): v for k, v in sym_code2idx.items()}
    pro_code2idx = {k.replace(".", ""): v for k, v in pro_code2idx.items()}

    sym_embd_placeholders = [0 for _ in range(len(sym_voc.idx2word))]
    pro_embd_placeholders = [0 for _ in range(len(pro_voc.idx2word))]
    med_embd_placeholders = [0 for _ in range(len(med_voc.idx2word))]

    if model in ["gat", "gcn", "omr"]:
        all_sym_embd, all_pro_embd, all_med_embd = read_gnn_embeddings(model, date_param)
    elif model in ["QEN", "TMN"]:
        all_sym_embd, all_pro_embd, all_med_embd = read_expansion_embeddings(model)

    for k, v in med_voc.word2idx.items():
        med_embd_placeholders[v] = all_med_embd[med_code2idx[k], :].unsqueeze(0)
    med_embd = torch.cat(med_embd_placeholders, dim=0)
    
    for k, v in sym_voc.word2idx.items():
        sym_embd_placeholders[v] = all_sym_embd[sym_code2idx[k], :].unsqueeze(0)
    sym_embd = torch.cat(sym_embd_placeholders, dim=0)
    
    for k, v in pro_voc.word2idx.items():
        pro_embd_placeholders[v] = all_pro_embd[pro_code2idx[k], :].unsqueeze(0)
    pro_embd = torch.cat(pro_embd_placeholders, dim=0)
    
    torch.save(med_embd, f"data/med_embd_{model}.pt")
    torch.save(sym_embd, f"data/sym_embd_{model}.pt")
    torch.save(pro_embd, f"data/pro_embd_{model}.pt")
    

if __name__ == "__main__":
    voc = dill.load(open("data/voc_final.pkl", "rb"))
    model_date = {"gcn": "Jan27_13-32-12_medi_64_random_neural_0.001", 
                  "gat": "Jan27_11-29-59_medi_64_random_neural_0.001"}
    # model_date = {"omr": "Jan17_16-53-10_64_random_mlp_0.001"}
    for m, d in model_date.items():
        transfer(voc, model=m, date_param=d)
    # transfer(voc, model="TMN")
