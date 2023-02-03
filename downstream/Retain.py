import torch
import torch.nn as nn
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
import dill
import time
import argparse
import json
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

import sys

sys.path.append("..")
from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, get_sparse_test

torch.manual_seed(1203)
model_name = "Retain"
# resume_path = "Epoch_50_JA_0.4952_DDI_0.08157.model"

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False, help="test mode")
parser.add_argument("--sparse", action="store_true", default=False, help="test mode")
parser.add_argument("--pro_taxo", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
# parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--embd_mode", default="taxo", help="test mode")
parser.add_argument("--epochs", type=int, default=60, help="coefficient of P signal")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=0, help="which cuda")
parser.add_argument("--sparse_percentage", type=float, default=0.3, help="percentage of sparse stuff")
parser.add_argument("--tolerance", type=int, default=1, help="percentage of sparse stuff")
parser.add_argument("--sparse_field", type=str, default="med", help="percentage of sparse stuff")

args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch, sparse_test=False):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0
    
    if sparse_test:
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for step, input in enumerate(data_eval):
            target_output = model(input)
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[-1][2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        # llprint("\rtest step: {} / {}".format(step, len(data_eval)))
    else:
        for step, input in enumerate(data_eval):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
            for adm_idx, adm in enumerate(input):
                target_output = model(input[: adm_idx + 1])

                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)

                # prediction prod
                target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
                y_pred_prob.append(target_output)

                # prediction med set
                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)

                # prediction label
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

            smm_record.append(y_pred_label)
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
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
    pro_large_dict = dill.load(open("data/proc_code2index.pkl", "rb"))
    sym_ontology = dill.load(open("data/diag_ontology.pkl", "rb"))
    pro_ontology = dill.load(open("data/proc_ontology.pkl", "rb"))
    device = torch.device("cuda:{}".format(args.cuda))

    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    if args.sparse:
        data_test = get_sparse_test(data,
                                    data_test, 
                                    voc,
                                    sparse_percentage=args.sparse_percentage, 
                                    tolerance=args.tolerance, 
                                    sparse_field=args.sparse_field)
    data_eval = data[split_point + eval_len :]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    if args.embd_mode!="random":
        taxo_diag_embd = torch.load(f"data/sym_embd_{args.embd_mode}.pt").to(device)
        taxo_pro_embd = torch.load(f"data/pro_embd_{args.embd_mode}.pt").to(device)
        taxo_med_embd = torch.load(f"data/med_embd_{args.embd_mode}.pt").to(device)
    else:
        taxo_diag_embd = None
        taxo_pro_embd = None
        taxo_med_embd = None
    model = Retain(voc_size, device=device, embd_mode=args.embd_mode, pro_taxo=args.pro_taxo, taxo_diag_embd=taxo_diag_embd, taxo_pro_embd=taxo_pro_embd, taxo_med_embd=taxo_med_embd, diag_ontology=sym_ontology,
        diag_large_dict=sym_large_dict,
        diag_small_dict=voc["diag_voc"].word2idx,
        proc_ontology=pro_ontology,
        proc_large_dict=pro_large_dict,
        proc_small_dict=voc["pro_voc"].word2idx,)
    # model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, args.resume_path), 'rb')))
    is_sparse = f"_SPARSE_{args.sparse_percentage}_TOL_{int(args.tolerance)}_{args.sparse_field}" if args.sparse else ""
    # some
    is_pro_taxo = "" if ((args.pro_taxo and args.embd_mode!="random") or args.embd_mode=="random") else "_no_pro_taxo"
    if args.test:
        best_epoch = dill.load(open(f"saved/{model_name}/history_{model_name}_{args.embd_mode}{is_pro_taxo}.pkl", "rb"))["best_epoch"]
        resume_path = f"saved/{model_name}/Epoch_{best_epoch}_{args.embd_mode}{is_pro_taxo}.model"
        model.load_state_dict(torch.load(open(resume_path, "rb")))
        model.to(device=device)
        tic = time.time()
        
        result = []
        for _ in range(10):
            start_test_time = time.time()
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 0.8), replace=True
            )
            # test_sample = data_test
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, test_sample, voc_size, 0, sparse_test=args.sparse
            )
            end_test_time = time.time()
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med, end_test_time-start_test_time])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        out_json_str = {
            "ddi": f"${mean[0]:.4f} \pm {std[0]:.4f}$",
            "ja": f"${mean[1]:.4f} \pm {std[1]:.4f}$",
            "avg_f1": f"${mean[2]:.4f} \pm {std[2]:.4f}$",
            "prauc": f"${mean[3]:.4f} \pm {std[3]:.4f}$",
            "avg_med": f"${mean[4]:.4f} \pm {std[4]:.4f}$",
            "avg_time": f"${mean[5]:.1f} \pm {std[5]:.4f}$"
        }
        print(out_json_str)
        json.dump(out_json_str, open(f"saved/{model_name}/test_result_{args.embd_mode}{is_pro_taxo}{is_sparse}.json", "w"))
        print("test time: {}".format(time.time() - tic))
        return

    print("parameters", get_n_params(model))
    optimizer = Adam(model.parameters(), args.lr)
    model.to(device=device)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = args.epochs
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):
            # if len(input) < 2:
            #     continue

            loss = 0
            # for i in range(1, len(input)):
            #     target = np.zeros((1, voc_size[2]))
            #     target[:, input[i][2]] = 1

            #     output_logits = model(input[:i])
            #     loss += F.binary_cross_entropy_with_logits(
            #         output_logits, torch.FloatTensor(target).to(device)
            #     )
            for i in range(0, len(input)):
                target = np.zeros((1, voc_size[2]))
                target[:, input[i][2]] = 1

                output_logits = model(input[:i+1])
                loss += F.binary_cross_entropy_with_logits(
                    output_logits, torch.FloatTensor(target).to(device)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
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
                    f"Epoch_{epoch}_{args.embd_mode}{is_pro_taxo}.model",
                ),
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


if __name__ == "__main__":
    main()
