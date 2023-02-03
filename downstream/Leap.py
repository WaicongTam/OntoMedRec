import torch
import torch.nn as nn
import argparse
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
import dill
import json
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict

import sys

sys.path.append("..")
from models import Leap
from util import (
    llprint,
    sequence_metric,
    sequence_output_process,
    ddi_rate_score,
    get_n_params,
    get_sparse_test
)

torch.manual_seed(1203)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_name = "Leap"
# resume_path = "saved/{}/Epoch_49_JA_0.4603_DDI_0.07427.model".format(model_name)

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False, help="test mode")
parser.add_argument("--embd_mode", default="taxo", help="test mode")
parser.add_argument("--sparse", action="store_true", default=False, help="test mode")
parser.add_argument("--epochs", type=int, default=50, help="coefficient of P signal")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--pro_taxo", action="store_true", default=False, help="test mode")
# parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--cuda", type=int, default=0, help="which cuda")
parser.add_argument("--sparse_percentage", type=float, default=0.3, help="percentage of sparse stuff")
parser.add_argument("--tolerance", type=float, default=1, help="percentage of sparse stuff")
parser.add_argument("--sparse_field", type=str, default="med", help="percentage of sparse stuff")

args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch, sparse_test=False):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        if sparse_test:
            input=[input[-1]]
        for adm_index, adm in enumerate(input):
            output_logits = model(adm)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            output_logits = output_logits.detach().cpu().numpy()

            # prediction med set
            out_list, sorted_predict = sequence_output_process(
                output_logits, [voc_size[2], voc_size[2] + 1]
            )
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(
            np.array(y_gt),
            np.array(y_pred),
            np.array(y_pred_prob),
            np.array(y_pred_label),
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        # llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="data/ddi_A_final.pkl")

    # llprint(
    #     "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
    #         ddi_rate,
    #         np.mean(ja),
    #         np.mean(prauc),
    #         np.mean(avg_p),
    #         np.mean(avg_r),
    #         np.mean(avg_f1),
    #         med_cnt / visit_cnt,
    #     )
    # )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():

    # load data
    data_path = "data/records_final.pkl"
    voc_path = "data/voc_final.pkl"
    sym_large_dict = dill.load(open("data/diag_code2index.pkl", "rb"))
    med_large_dict = dill.load(open("data/atc_code2index.pkl", "rb"))
    sym_ontology = dill.load(open("data/diag_ontology.pkl", "rb"))
    med_ontology = dill.load(open("data/atc_ontology.pkl", "rb"))
    device = torch.device("cuda:{}".format(args.cuda))

    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    if args.sparse:
        data_test = get_sparse_test(data,
                                    data_test, 
                                    voc,
                                    sparse_percentage=args.sparse_percentage, 
                                    tolerance=args.tolerance, 
                                    sparse_field=args.sparse_field)

    END_TOKEN = voc_size[2] + 1
    is_sparse = f"_SPARSE_{args.sparse_percentage}_TOL_{int(args.tolerance)}_{args.sparse_field}" if args.sparse else ""
    # is_taxo = "_taxo" if args.taxomedrec else ""
    is_pro_taxo = "" if ((args.pro_taxo and args.embd_mode=="taxo") or args.embd_mode!="taxo") else "_no_pro_taxo"
    # if args.embd_mode == "taxo":
    #     if args.pro_taxo:
    #         is_pro_taxo = ""
    #     else:
    #         is_pro_taxo = "_no_pro_taxo"
    # else:
    #     is_pro_taxo = ""
    if args.embd_mode != "random":
        taxo_diag_embd = torch.load(f"data/sym_embd_{args.embd_mode}.pt").to(device)
        taxo_med_embd = torch.load(f"data/med_embd_{args.embd_mode}.pt").to(device)
    else:
        taxo_diag_embd = None
        taxo_med_embd = None
    model = Leap(voc_size, device=device, embd_mode=args.embd_mode, taxo_diag_embd=taxo_diag_embd, taxo_med_embd=taxo_med_embd,diag_ontology=sym_ontology,
        diag_large_dict=sym_large_dict,
        diag_small_dict=voc["diag_voc"].word2idx,
        med_ontology=med_ontology,
        med_large_dict=med_large_dict,
        med_small_dict=voc["med_voc"].word2idx,)
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

    if args.test:
        best_epoch = dill.load(open(f"saved/{model_name}/history_{model_name}_{args.embd_mode}{is_pro_taxo}.pkl", "rb"))["best_epoch"]
        resume_path = f"saved/{model_name}/Epoch_{best_epoch}_{args.embd_mode}{is_pro_taxo}.model"
        model.load_state_dict(torch.load(open(resume_path, "rb")))
        model.to(device=device)
        tic = time.time()
        result = []
        for _ in range(10):
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 0.8), replace=True
            )
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, test_sample, voc_size, 0, sparse_test=args.sparse
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)
        out_json_str = {
            "ddi": f"${mean[0]:.4f} \pm {std[0]:.4f}$",
            "ja": f"${mean[1]:.4f} \pm {std[1]:.4f}$",
            "avg_f1": f"${mean[2]:.4f} \pm {std[2]:.4f}$",
            "prauc": f"${mean[3]:.4f} \pm {std[3]:.4f}$",
            "avg_med": f"${mean[4]:.4f} \pm {std[4]:.4f}$"
        }
        json.dump(out_json_str, open(f"saved/{model_name}/test_result_{args.embd_mode}{is_sparse}{is_pro_taxo}.json", "w"))
        print("test time: {}".format(time.time() - tic))
        return

    model.to(device=device)
    print("parameters", get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = args.epochs
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):
            for adm in input:

                loss_target = adm[2] + [END_TOKEN]
                output_logits = model(adm)
                loss = F.cross_entropy(
                    output_logits, torch.LongTensor(loss_target).to(device)
                )
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, data_eval, voc_size, epoch
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    f'Epoch_{epoch}_{args.embd_mode}{is_pro_taxo}.model'), 
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))
        history["best_epoch"] = best_epoch

        dill.dump(
            history,
            open(
                os.path.join(
                    "saved", 
                    args.model_name, 
                    f"history_{args.model_name}_{args.embd_mode}{is_pro_taxo}.pkl"
                ),
                "wb",
            ),
        )


def fine_tune(fine_tune_name=""):

    # load data
    data_path = "data/records_final.pkl"
    voc_path = "data/voc_final.pkl"
    device = torch.device("cpu:0")

    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    ddi_A = dill.load(open("data/ddi_A_final.pkl", "rb"))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    # data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(
        torch.load(open(os.path.join("saved", args.model_name, fine_tune_name), "rb"))
    )
    model.to(device)

    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=args.lr)
    ddi_rate_record = []

    EPOCH = 60
    for epoch in range(EPOCH):
        loss_record = []
        start_time = time.time()
        random_train_set = [random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(
                    output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1]
                )

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))
                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(random_train_set)))

        if K_flag:
            print()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, data_test, voc_size, epoch
            )

    # test
    torch.save(
        model.state_dict(),
        open(os.path.join("saved", args.model_name, "final.model"), "wb"),
    )


if __name__ == "__main__":
    main()
    # fine_tune(fine_tune_name='Epoch_1_JA_0.2765_DDI_0.1158.model')
