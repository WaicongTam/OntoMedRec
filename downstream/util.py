from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
# import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
# from collections import Counter
from rdkit import Chem
from collections import defaultdict
import torch
import itertools
warnings.filterwarnings('ignore')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1203)
    x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1203)
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    """生成最终正确的序列，output_logits表示每个位置的prob，filter_token代表SOS与END"""
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]  # 每个位置上按概率的降序排序

    out_list = []   # 生成的结果
    break_flag = False
    for i in range(len(pind)):
        # 顺序遍历pind上所有值
        # break_flag来判断是否退出sentence生成的循环
        if break_flag:
            break
        # 每个位置上是按降序排序好的结果
        for j in range(pind.shape[1]):
            label = pind[i][j]
            # 如果遇到了SOS或者END，就表示句子over了
            if label in filter_token:
                break_flag = True
                break
            # 如果遇到了未出现过的，就继续生成
            # 否则就继续看下一个概率较大的药
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    # 将out_list中按照概率的高低将所有药物排序？
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            # try:
            #     all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
            # except:
            #     continue
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    # p_1 = precision_at_k(y_gt, y_label, k=1)
    # p_3 = precision_at_k(y_gt, y_label, k=3)
    # p_5 = precision_at_k(y_gt, y_label, k=5)
    
    f1 = f1(y_gt, y_pred)
    try:
        prauc = precision_auc(y_gt, y_prob)
    except ValueError:
        prauc = float(0.0)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def sequence_metric_v2(y_gt, y_pred, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    # try:
    #     auc = roc_auc(y_gt, y_prob)
    # except ValueError:
    #     auc = 0
    # p_1 = precision_at_k(y_gt, y_label, k=1)
    # p_3 = precision_at_k(y_gt, y_label, k=3)
    # p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    # prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, path='data/ddi_A_final.pkl'):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):
    
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    for index, atc3 in med_voc.items():

        smilesList = list(molecule[atc3])
        """Create each data with the above defined functions."""
        counter = 0  # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
                )
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                    fingerprints = np.append(fingerprints, 1)

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except:
                continue

        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    N_fingerprint = len(fingerprint_dict)
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        if item > 0:
            average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)



def output_flatten(labels, logits, seq_length, m_length_matrix, med_num, END_TOKEN, device, training=True, testing=False, max_len=20):
    '''
    labels: [batch_size, visit_num, medication_num]
    logits: [batch_size, visit_num, max_med_num, medication_vocab_size]
    '''
    # 将最终多个维度的结果展开
    batch_size, max_seq_length = labels.size()[:2]
    assert max_seq_length == max(seq_length)
    whole_seqs_num = seq_length.sum().item()
    if training:
        whole_med_sum = sum([sum(buf) for buf in m_length_matrix]) + whole_seqs_num # 因为每一个seq后面会多一个END_TOKEN

        # 将结果展开，然后用库函数进行计算
        labels_flatten = torch.empty(whole_med_sum).to(device)
        logits_flatten = torch.empty(whole_med_sum, med_num).to(device)

        start_idx = 0
        for i in range(batch_size): # 每个batch
            for j in range(seq_length[i]):  # seq_length[i]指这个batch对应的seq数目
                for k in range(m_length_matrix[i][j]+1):  # m_length_matrix[i][j]对应seq中med的数目
                    if k==m_length_matrix[i][j]:    # 最后一个label指定为END_TOKEN
                        labels_flatten[start_idx] = END_TOKEN
                    else:
                        labels_flatten[start_idx] = labels[i, j, k]
                    logits_flatten[start_idx, :] = logits[i, j, k, :]
                    start_idx += 1
        return labels_flatten, logits_flatten
    else:
        # 将结果按照adm展开，然后用库函数进行计算
        labels_flatten = []
        logits_flatten = []

        start_idx = 0
        for i in range(batch_size): # 每个batch
            for j in range(seq_length[i]):  # seq_length[i]指这个batch对应的seq数目
                labels_flatten.append(labels[i,j,:m_length_matrix[i][j]].detach().cpu().numpy())
                
                if testing:
                    logits_flatten.append(logits[j])  # beam search目前直接给出了预测结果
                else:
                    logits_flatten.append(logits[i,j,:max_len,:].detach().cpu().numpy())     # 注意这里手动定义了max_len
                # cur_label = []
                # cur_seq_length = []
                # for k in range(m_length_matrix[i][j]+1):  # m_length_matrix[i][j]对应seq中med的数目
                #     if k==m_length_matrix[i][j]:    # 最后一个label指定为END_TOKEN
                #         continue
                #     else:
                #         labels_flatten[start_idx] = labels[i, j, k]
                #     logits_flatten[start_idx, :] = logits[i, j, k, :med_num]
                #     start_idx += 1
        return labels_flatten, logits_flatten


def print_result(label, prediction):
    '''
    label: [real_med_num, ]
    logits: [20, med_vocab_size]
    '''
    label_text = " ".join([str(x) for x in label])
    predict_text = " ".join([str(x) for x in prediction])
    
    return "[GT]\t{}\n[PR]\t{}\n\n".format(label_text, predict_text)


def get_sparse_test(data_all, data_test, voc, sparse_field="med", sparse_percentage=0.3, tolerance=1):
    all_syms_count = [0 for _ in range(len(voc["diag_voc"].idx2word))]
    all_meds_count = [0 for _ in range(len(voc["med_voc"].idx2word))]

    all_admission_record = list(itertools.chain.from_iterable(data_all))
    all_sym_record = list(itertools.chain.from_iterable([x[0] for x in all_admission_record]))
    all_med_record = list(itertools.chain.from_iterable([x[2] for x in all_admission_record]))

    for sym in all_sym_record:
        all_syms_count[sym] += 1

    for med in all_med_record:
        all_meds_count[med] += 1
        
    sym_sort = np.argsort(all_syms_count)
    med_sort = np.argsort(all_meds_count)

    if sparse_field == "med":
        test_field = med_sort[:int(sparse_percentage*len(med_sort))]
    elif sparse_field == "sym":
        test_field = med_sort[:int(sparse_percentage*len(sym_sort))]
    sparse_patients = []

    for patient in data_test:
        for adm_idx, admission in enumerate(patient):
            field_in_test = sum([1 if m in test_field else 0 for m in admission[0 if sparse_field=="sym" else 2]])
            if field_in_test >= tolerance:
                sparse_patients.append(patient[:adm_idx+1])
    
    return sparse_patients


class PKLSet(object):
    def __init__(self, batch_size):
        self.eval_path = 'data/4S/data_eval.pkl'
        self.voc_path = "data/voc_final.pkl"
        self.ddi_adj_path = 'data/ddi_A_final.pkl'
        self.ddi_adj = dill.load(open(self.ddi_adj_path, 'rb'))
        # self.ddi_adj = 0
        self.sym_train, self.drug_train, self.data_eval = self.check_file(batch_size)
        self.sym_sets, self.drug_multihots = self.mat_train_data()
        self.similar_sets_idx = self.find_similae_set_by_ja(self.sym_train)
        self.data_test = None
        # self.sym_counts = self.count_sym(self.sym_train)

    def check_file(self, batch_size):
        sym_path = f'data/4S/sym_train_{batch_size}.pkl'
        drug_path = f'data/4S/drug_train_{batch_size}.pkl'
        # if os.path.exists(sym_path):
        self.gen_batch_data(batch_size)
        return self.load_data(sym_path, drug_path)

    def load_data(self, sym_path, drug_path):
        sym_train, drug_train = dill.load(open(sym_path, 'rb')), dill.load(open(drug_path, 'rb'))
        data_eval = dill.load(open(self.eval_path, 'rb'))
        voc = dill.load(open(self.voc_path, 'rb'))
        sym_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

        self.n_sym, self.n_drug = len(sym_voc.idx2word), len(med_voc.idx2word)
        print("num symptom: {}, num drug: {}".format(self.n_sym, self.n_drug))
        return sym_train, drug_train, data_eval

    def count_sym(self):
        train_path = 'data/4S/data_train.pkl'
        data = dill.load(open(train_path, 'rb'))
        countings = np.zeros(self.n_sym)
        for adm in data:
            syms, drugs = adm[0], adm[2]
            countings[syms] += 1
        return countings

    def mat_train_data(self):
        train_path = 'data/4S/data_train.pkl'
        data_train = dill.load(open(train_path, 'rb'))
        sym_sets, drug_sets_multihot = [], []
        for adm in data_train:
            syms, drugs = adm[0], adm[2]
            sym_sets.append(syms)
            drug_multihot = np.zeros(self.n_drug)
            drug_multihot[drugs] = 1
            drug_sets_multihot.append(drug_multihot)
        return sym_sets, drug_sets_multihot

    def gen_batch_data(self, batch_size):
        voc = dill.load(open(self.voc_path, 'rb'))
        sym_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

        self.n_sym, self.n_drug = len(sym_voc.idx2word), len(med_voc.idx2word)
        sym_count = self.count_sym()
        size_dict, drug_dict = {}, {}
        sym_sets, drug_sets = [], []
        s_set_num = 0

        train_path = 'data/4S/data_train.pkl'
        data = dill.load(open(train_path, 'rb'))
        for adm in data:
            syms, drugs = adm[0], adm[2]
            sym_sets.append(syms)
            drug_sets.append(drugs)
            s_set_num += 1

        for adm in data:
            syms, drugs = adm[0], adm[2]
            drug_multihot = np.zeros(self.n_drug)
            drug_multihot[drugs] = 1
            if size_dict.get(len(syms)):
                size_dict[len(syms)].append(syms)
                drug_dict[len(syms)].append(drug_multihot)
            else:
                size_dict[len(syms)] = [syms]
                drug_dict[len(syms)] = [drug_multihot]

        keys, count = list(size_dict.keys()), 0
        keys.sort()
        new_s_set, new_d_set = [], []
        for size in keys:
            if size <= 1: continue
            for (syms, drugs) in zip(size_dict[size], drug_dict[size]):
                syms = np.array(syms)
                cnt, del_nums = torch.from_numpy(sym_count[syms]), int(max(1, len(syms) * 0.2))
                if del_nums == 1:
                    del_idx = torch.multinomial(cnt, len(syms) - del_nums)
                    remained = syms[del_idx.numpy()]
                    remained = remained.tolist()
                    new_s_set.append(remained)
                    new_d_set.append(drugs)
                else:
                    for _ in range(min(del_nums, 3)):
                        del_num = np.random.randint(1, del_nums)
                        del_idx = torch.multinomial(cnt, len(syms) - del_num)
                        remained = syms[del_idx.numpy()]
                        remained = remained.tolist()
                        new_s_set.append(remained)
                        new_d_set.append(drugs)

        for (remained, drugs) in zip(new_s_set, new_d_set):
            if size_dict.get(len(remained)) is None:
                count += 1
                size_dict[len(remained)] = [remained]
                drug_dict[len(remained)] = [drugs]
            elif remained not in size_dict[len(remained)]:
                count += 1
                size_dict[len(remained)].append(remained)
                drug_dict[len(remained)].append(drugs)

        sym_train, drug_train = [], []
        keys = list(size_dict.keys())
        keys.sort()
        for size in keys:
            num_size = len(size_dict[size])
            batch_num, start_idx = num_size // batch_size, 0
            if num_size % batch_size != 0: batch_num += 1
            for i in range(batch_num):
                if i == batch_num:
                    syms, drugs = size_dict[size][start_idx:], drug_dict[size][start_idx:]
                else:
                    syms, drugs = size_dict[size][start_idx:start_idx + batch_size], drug_dict[size][
                                                                                     start_idx:start_idx + batch_size]
                    start_idx += batch_size
                sym_train.append(syms)
                drug_train.append(drugs)

        with open(f'data/4S/sym_train_{batch_size}.pkl', 'wb') as f:
            dill.dump(sym_train, f)

        with open(f'data/4S/drug_train_{batch_size}.pkl', 'wb') as f:
            dill.dump(drug_train, f)

    def find_similae_set_by_ja(self, sym_train):
        similar_sets = [[] for _ in range(len(sym_train))]
        for i in range(len(sym_train)):
            for j in range(len(sym_train[i])):
                similar_sets[i].append(j)

        for idx, sym_batch in enumerate(sym_train):
            if len(sym_batch) <= 2 or len(sym_batch[0]) <= 2: continue
            batch_sets = [set(sym_set) for sym_set in sym_batch]
            for i in range(len(batch_sets)):
                max_intersection = 0
                for j in range(len(batch_sets)):
                    if i == j: continue
                    if len(batch_sets[i] & batch_sets[j]) > max_intersection:
                        max_intersection = len(batch_sets[i] & batch_sets[j])
                        similar_sets[idx][i] = j

        return similar_sets
