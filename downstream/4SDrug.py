import torch
import torch.nn.functional as F
from util import multi_label_metric, ddi_rate_score, get_sparse_test
import numpy as np
from util import PKLSet
from tqdm import trange, tqdm
from models import FourSDrug
from models import RAdam
import argparse
import os
import time
import dill
import itertools
import json


if torch.cuda.is_available():
    torch.cuda.set_device(0)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument("--embd_mode", default="taxo", help="test mode")
    parser.add_argument("--sparse", action="store_true", default=False, help="test mode")
    parser.add_argument("--sparse_field", type=str, default="med", help="percentage of sparse stuff")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='MIMIC3')
    parser.add_argument("--sparse_percentage", type=float, default=0.3, help="percentage of sparse stuff")
    parser.add_argument("--tolerance", type=float, default=1, help="percentage of sparse stuff")

    return parser.parse_known_args()


def evaluate(model, test_loader, n_drugs, device="cpu"):
    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    for step, adm in enumerate(test_loader):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        syms, drugs = torch.tensor(adm[0]).to(device), torch.tensor(adm[2]).to(device)
        # print(syms, drugs)
        # print(syms.shape, drugs.shape)
        scores = model.evaluate(syms, device=device)
        # scores = 2 * torch.softmax(scores, dim=-1) - 1

        y_gt_tmp = np.zeros(n_drugs)
        y_gt_tmp[drugs.cpu().numpy()] = 1
        y_gt.append(y_gt_tmp)

        result = torch.sigmoid(scores).detach().cpu().numpy()
        y_pred_prob.append(result)
        y_pred_tmp = result.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred.append(y_pred_tmp)

        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1
        med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt),
                                                                                 np.array(y_pred),
                                                                                 np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
    # print(y_pred_label)
    ddi_rate = ddi_rate_score(smm_record, path='data/ddi_A_final.pkl')
    # ddi_rate = 0
    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), 1.0 * med_cnt / visit_cnt, ddi_rate


def run():
    args, unknown = parse_args()
    print(args)
    # is_taxo = "_taxo" if args.taxomedrec else ""
    is_sparse = f"_SPARSE_{args.sparse_percentage}_TOL_{int(args.tolerance)}_{args.sparse_field}" if args.sparse else ""
    # config = Config()
    # voc = dill.load(open("/home/wtan0047/ar57/medication/baselines/4SDrug/datasets/MIMIC3/voc_final.pkl", "rb"))
    voc = dill.load(open("data/voc_final.pkl", "rb"))
    sym_voc, med_voc = voc['diag_voc'], voc['med_voc']
    sym_count = [0 for _ in range(len(sym_voc.idx2word))]
    med_count = [0 for _ in range(len(med_voc.idx2word))]

    # original_train_set = dill.load(open("/home/wtan0047/ar57/medication/baselines/4SDrug/datasets/MIMIC3/data_train.pkl", "rb"))
    # original_eval_set = dill.load(open("/home/wtan0047/ar57/medication/baselines/4SDrug/datasets/MIMIC3/data_eval.pkl", "rb"))
    # original_test_set = dill.load(open("/home/wtan0047/ar57/medication/baselines/4SDrug/datasets/MIMIC3/data_test.pkl", "rb"))
    # all_set = original_train_set+original_eval_set+original_test_set
    all_records = dill.load(open("data/records_final.pkl", "rb"))
    all_set = list(itertools.chain.from_iterable(all_records))
    
    for adm in all_set:
        syms, meds = adm[0], adm[2]
        for sym in syms:
            sym_count[sym] += 1
        for med in meds:
            med_count[med] += 1
            
    sym_sort = np.argsort(sym_count)
    med_sort = np.argsort(med_count)
    
    # test_sym_percentage = args.sparse_percentage
    if args.sparse_field == "med":
        test_field = med_sort[:int(args.sparse_percentage*len(med_sort))]
    else:
        test_field = sym_sort[:int(args.sparse_percentage*len(sym_sort))]
    # print("There are ", int(test_sym_percentage*len(med_sort)), " test syndromes")
    # tolerence = args.tolerance
    
    split_point = int(len(all_records) * 2 / 3)
    data_train = list(itertools.chain.from_iterable(all_records[:split_point]))
    eval_len = int(len(all_records[split_point:]) / 2)
    data_test = list(itertools.chain.from_iterable(all_records[split_point:split_point + eval_len]))
    data_eval = list(itertools.chain.from_iterable(all_records[split_point+eval_len:]))
    
    dill.dump(data_train, open("data/4S/data_train.pkl", "wb"))
    dill.dump(data_eval, open("data/4S/data_eval.pkl", "wb"))
    # all_test = pklSet.data_eval
    # print(len(pklSet.data_eval))
    if args.sparse_field == "med":
        sparse_test = [adm for adm in data_test if any(f in test_field for f in adm[2])]
    else:
        sparse_test = [adm for adm in data_test if any(f in test_field for f in adm[0])]
    # sparse_test = [adm for adm in data_test if len([sym for sym in adm[2] if sym in test_sym]) >= tolerence]
    
    # print(len(sparse_test))
    # print(len(pklSet.data_eval))
    # print(len(pklSet.sym_train))
    if len(sparse_test) == 0:
        print("No test set")
        return (None, None, None)
    
    pklSet = PKLSet(args.batch_size)
    if args.sparse:
        pklSet.data_test = sparse_test
    else:
        pklSet.data_test = data_test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    # device = torch.device("cpu")
    
    if args.embd_mode != "random":
        taxo_diag_embd = torch.load(f"data/sym_embd_{args.embd_mode}.pt")
        taxo_med_embd = torch.load(f"data/med_embd_{args.embd_mode}.pt")
    else:
        taxo_diag_embd = None
        taxo_med_embd = None
    model = FourSDrug(pklSet.n_sym, 
                  pklSet.n_drug, 
                  torch.FloatTensor(pklSet.ddi_adj).to(device), 
                  pklSet.sym_sets,
                  torch.tensor(pklSet.drug_multihots).to(device),
                  args.embedding_dim,
                  embd_mode=args.embd_mode,
                  taxo_diag_embd=taxo_diag_embd,
                  taxo_med_embd=taxo_med_embd).to(device)
    # model.load_state_dict(torch.load('best_ja_at15.pt', map_location=device))
    optimizer = RAdam(model.parameters(), lr=args.lr)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("total number of parameters: ", tot_params)
    if args.test:
        model.load_state_dict(torch.load(f'saved/4SDrug/best_ja_at15_BATCH{args.batch_size}_{args.embd_mode}.pt', map_location=device))
        test_ja, test_prauc, test_avg_p, test_avg_r, test_avg_f1, test_avg_med, test_ddi_rate = [[] for _ in range(7)]
        print(pklSet.data_test[0])
        for _ in range(10):
            idx = list(range(len(pklSet.data_test)))
            test_idx = np.random.choice(
                idx, round(len(pklSet.data_test) * 0.8), replace=True
            )
            test_sample = [pklSet.data_test[i] for i in test_idx]
            start_time = time.time()
            ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = evaluate(model, test_sample, pklSet.n_drug, device)
            end_time = time.time()
            test_ja.append(ja)
            test_prauc.append(prauc)
            test_avg_p.append(avg_p)
            test_avg_r.append(avg_r)
            test_avg_f1.append(avg_f1)
            test_avg_med.append(avg_med)
            test_ddi_rate.append(ddi_rate)
        # tune.report(ja=ja, prauc=prauc, avg_med=avg_med, ddi_rate=ddi_rate, avg_f1=avg_f1)
        # print('-' * 89)
        # print(
        #     '| best ja {:5.4f} | prauc {:5.4f} | avg_p {:5.4f} | avg_recall {:5.4f} | '
        #     'avg_f1 {:5.4f} | avg_med {:5.4f} | ddi_rate {:5.4f}'.format(ja, prauc, avg_p,
        #                                                                  avg_r,
        #                                                                  avg_f1, avg_med, ddi_rate))
        # print('-' * 89)
        results = [[np.mean(r), np.std(r)]for r in [test_ja, test_prauc, test_avg_p, test_avg_r, test_avg_f1, test_avg_med, test_ddi_rate]]
        results_json = {
            "ja": f"${np.mean(test_ja):.4f} \pm {np.std(test_ja):.4f}$", 
            # "test_prauc":f"${np.mean(test_prauc):.4f} \pm {np.std(test_prauc):.4f}$",
            # "test_avg_p":f"${np.mean(test_avg_p):.4f} \pm {np.std(test_avg_p):.4f}$",
            # "test_avg_r":f"${np.mean(test_avg_r):.4f} \pm {np.std(test_avg_r):.4f}$",
            "avg_f1":f"${np.mean(test_avg_f1):.4f} \pm {np.std(test_avg_f1):.4f}$",
            "ddi":f"${np.mean(test_ddi_rate):.4f} \pm {np.std(test_ddi_rate):.4f}$",
            "avg_med":f"${np.mean(test_avg_med):.4f} \pm {np.std(test_avg_med):.4f}$"
        }
        print(results_json)
        json.dump(results_json, open(f"saved/4SDrug/test_result_{args.embd_mode}{is_sparse}.json", "w"))
        return results_json
    else:
        best_ja = -np.inf
        best_ddi = np.inf
        best_f1 = -np.inf
        for epoch in trange(args.epochs):
            losses, set_idx = 0.0, 0
            model.train()

            for step, (syms, drugs, similar_idx) in tqdm(enumerate(zip(pklSet.sym_train, pklSet.drug_train, pklSet.similar_sets_idx))):
                syms, drugs, similar_idx = torch.tensor(syms).to(device), torch.tensor(drugs).to(device), torch.tensor(similar_idx).to(device)
                # print(syms)
                model.zero_grad()
                optimizer.zero_grad()
                scores, bpr, loss_ddi = model(syms, drugs, similar_idx, device)
                # scores = 2 * torch.softmax(scores, dim=-1) - 1

                sig_scores = torch.sigmoid(scores)
                scores_sigmoid = torch.where(sig_scores == 0, torch.tensor(1.0).to(device), sig_scores)

                bce_loss = F.binary_cross_entropy_with_logits(scores, drugs)
                entropy = -torch.mean(sig_scores * (torch.log(scores_sigmoid) - 1))
                loss = bce_loss + 0.5 * entropy + args.alpha * bpr + args.beta * loss_ddi
                losses += loss.item() / syms.shape[0]
                loss.backward()
                optimizer.step()
                set_idx += 1

            start = time.time()
            ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = evaluate(model, pklSet.data_eval, pklSet.n_drug, device)
            print('-' * 89)
            print(
                '| end of epoch{:3d}| training time{:5.4f} | ja {:5.4f} | prauc {:5.4f} | avg_p {:5.4f} | avg_recall {:5.4f} | '
                'avg_f1 {:5.4f} | avg_med {:5.4f} | ddi_rate {:5.4f}'.format(epoch, time.time() - start, ja, prauc, avg_p, avg_r,
                                                                            avg_f1, avg_med, ddi_rate))
            print('-' * 89)

            if ja > best_ja:
                torch.save(model.state_dict(), f'saved/4SDrug/best_ja_at15_BATCH{args.batch_size}_{args.embd_mode}.pt')
                best_ja = ja
            if ddi_rate < best_ddi:
                best_ddi = ddi_rate
            if avg_f1 > best_f1:
                best_f1 = avg_f1

    

if __name__ == '__main__':
    # ray.init(local_mode=True)
    # config = {
    #     "test_sym_percentage": tune.choice([0.1*i for i in range(1, 10)]),
    #     "tolerence": tune.choice([0, 1, 2, 3, 4, 5])
    # }
    
    # TODO: put lowest n-% of symptoms in test set
    # test_sym_percentage = [0.1*i for i in range(3, 10)]
    
    # # If n(n_sym in test_sym) > tolerence, then add to test set
    # tolenrences = [i for i in range(1, 10)]
    
    # best_perfs = []
    # for test_sym, tolerence in itertools.product(test_sym_percentage, tolenrences):
    #     config = {
    #         "test_sym_percentage": test_sym,
    #         "tolerence": tolerence
    #     }
    #     results = train(config)
    #     best_perfs.append((test_sym, tolerence, *results))
    #     print(best_perfs)
    #     with open("/home/wtan0047/ar57/medication/baselines/4SDrug//best_perfs_100epoch_64_medi.json", "w") as f:
    #         json.dump(best_perfs, f, indent=1)
        
    run()
    # print(results)
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=200,
    #     grace_period=1,
    #     reduction_factor=2)
    
    # reporter = CLIReporter(
    #     # parameter_columns=["l1", "l2", "lr", "batch_size"],
    #     metric_columns=["epoch", "ja", "prauc", "avg_med", "ddi_rate", "avg_f1"])
    
    # result = tune.run(partial(train),
    #                   resources_per_trial={"cpu": 2},
    #                   config=config,
    #                   num_samples=4,
    #                   scheduler=scheduler,
    #                   progress_reporter=reporter,
    #                   checkpoint_at_end=True,
    #                   )
    # train(config)