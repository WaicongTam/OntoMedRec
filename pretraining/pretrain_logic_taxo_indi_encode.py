from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ltn
import numpy as np
import networkx as nx
import dill
import random
import itertools
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
import os


parser = argparse.ArgumentParser(description="Run")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--embd_mode', type=str, default="random")
parser.add_argument('--scorer', type=str, default="neural")
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--k', type=int, default=16)
parser.add_argument('--p_exist', type=int, default=2)
parser.add_argument('--p_all', type=int, default=2)
args, unknown = parser.parse_known_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TaxoScorer(nn.Module):
    def __init__(self, layer_sizes=[128, 64, 32, 8, 4, 1]):
        super(TaxoScorer, self).__init__()
        self.elu = nn.ELU()
        self.linear_layers = nn.ModuleList([nn.Linear(layer_sizes[i-1], layer_sizes[i]) for i in range(1, len(layer_sizes))])    
    
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        for layer in self.linear_layers[:-1]:
            x = self.elu(torch.tanh(layer(x)))
        x = torch.sigmoid(self.linear_layers[-1](x))
        return x 
    

class IndicationScorer(TaxoScorer):
    def __init__(self, layer_sizes=[128, 64, 32, 8, 4, 1]):
        super(IndicationScorer, self).__init__(layer_sizes=layer_sizes)
        self.med_transform = nn.Linear(int(layer_sizes[0]/2), int(layer_sizes[0]/2))
        self.sym_transform = nn.Linear(int(layer_sizes[0]/2), int(layer_sizes[0]/2))
    
    def forward(self, med, sym):
        med = self.med_transform(med)
        sym = self.sym_transform(sym)
        x = torch.cat([med, sym], dim=1)
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
        x = torch.sigmoid(self.linear_layers[-1](x))
        return x 
        

class LTNDataLoader:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=True):
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch_points = self.data[idxlist[start_idx:end_idx]]

            yield batch_points


def collate_fn_single_taxo(batch_edges):
    nodes = []
    for node_1, node_2 in batch_edges:
        nodes.append(node_1)
        nodes.append(node_2)
    nodes  = list(set(nodes))
    return batch_edges, nodes


class SingleTaxoCollate():
    def __init__(self, graph):
        self.graph = graph
    
    def __call__(self, batch_edges):
        nodes = []
        for node_1, node_2 in batch_edges:
            nodes.append(node_1)
            nodes.append(node_2)
        nodes  = list(set(nodes))
        return batch_edges, nodes


class SubgraphSampler():
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size
        
    def __len__(self):
        return self.graph.get_number_of_nodes()-1
    
    def sample(self, n):
        root = self.graph.get_root()
        siblings = self.graph.get_siblings(n)
        parent = self.graph.get_parent(n)
        ancestors = self.graph.get_descending_path(root, n)[:-2]
        children = self.graph.get_sucessors(n)
        parent_edges = [(parent, n)] + [(n, child) for child in children] + [(parent, s) for s in siblings]
        if len(ancestors) > 1:
            parent_edges += [(ancestors[x], ancestors[x+1]) for x in range(len(ancestors)-1)]
        ancestor_edges = [(a, n) for a in ancestors]
        sibling_edges = list(itertools.combinations(siblings+[n], 2))
        nodes = siblings + [parent] + [n] + children + ancestors
        return nodes, parent_edges, ancestor_edges, sibling_edges
        
    def __iter__(self):
        # indices = self.graph.get_non_root_nodes()
        # np.random.shuffle(indices)
        # for n in indices:
        #     root = self.graph.get_root()
        #     siblings = self.graph.get_siblings(n)
        #     parent = self.graph.get_parent(n)
        #     ancestors = self.graph.get_descending_path(root, n)[:-2]
        #     children = self.graph.get_sucessors(n)
        #     parent_edges = [(parent, n)] + [(n, child) for child in children] + [(parent, s) for s in siblings]
        #     if len(ancestors) > 1:
        #         parent_edges += [(ancestors[x], ancestors[x+1]) for x in range(len(ancestors)-1)]
        #     ancestor_edges = [(a, n) for a in ancestors]
        #     sibling_edges = list(itertools.combinations(siblings+[n], 2))
        #     nodes = siblings + [parent] + [n] + children + ancestors 
        #     yield nodes, parent_edges, ancestor_edges, sibling_edges
        indices = self.graph.get_non_root_nodes()
        np.random.shuffle(indices)

        for _, start_idx in enumerate(range(0, len(indices), self.batch_size)):
            end_idx = min(start_idx+self.batch_size, len(indices))
            batch_nodes = indices[start_idx:end_idx]
            sampled = [self.sample(n) for n in batch_nodes]
            batch_node_set = list(set(itertools.chain.from_iterable([x[0] for x in sampled])))
            batch_parent_set = list(set(itertools.chain.from_iterable([x[1] for x in sampled])))
            yield batch_node_set, batch_parent_set

class MediSubgraphSampler():
    def __init__(self, medi, med_graph, sym_graph):
        self.medi = medi
        self.med_graph = med_graph
        self.sym_graph = sym_graph
    
    def __len__(self):
        return len(self.medi)
    
    def __iter__(self):
        indices = list(range(len(self.medi)))
        np.random.shuffle(indices)
        for n in indices:
            med, sym = self.medi[n]
            similar_meds = [med for med, sym in self.medi if sym==sym]
            sym_subgraph = self.sym_graph.get_descending_subgraph(sym) 
            
            

class MediCollate():
    def __init__(self, sym_graph):
        self.sym_graph = sym_graph
    
    def __call__(self, batch_edges):
        med_nodes = []
        sym_nodes = []
        sym_children = []
        
        for med, sym in batch_edges:
            med_nodes.append(med)
            sym_nodes.append(sym)
        
        return batch_edges, med_nodes, sym_nodes
    

def train_one_ontology(epoch,
                       node_loader,
                       encoder,
                       optimiser,
                       parent_scorer,
                       ancestor_scorer,
                       sibling_scorer,
                       p,
                       writer,
                       name,
                       SatAgg,
                       Forall,
                       Not,
                       Implies,
                       And):
    epoch_loss = 0
    
    for batch_idx, (nodes, parent_edges, ancestor_edges, sibling_edges) in enumerate(node_loader):
        optimiser.zero_grad()
        x_ = ltn.Variable("x", torch.stack([n.value for n in [encoder[n] for n in nodes]]).to(DEVICE))
        y_ = ltn.Variable("y", torch.stack([n.value for n in [encoder[n] for n in nodes]]).to(DEVICE))
        z_ = ltn.Variable("z", torch.stack([n.value for n in [encoder[n] for n in nodes]]).to(DEVICE))
        sat_agg = SatAgg(
            # Parental relation is non-reflexive
            Forall(x_, 
                    Not(parent_scorer(x_, x_)),
                    p=p),
            
            # Ancestor relation is non-reflexive
            Forall(x_, 
                    Not(ancestor_scorer(x_, x_)),
                    p=p),
            
            # Sibling relation is non-reflexive
            Forall(x_,
                    Not(sibling_scorer(x_, x_)),
                    p=p),
            
            # Parental relation is single direction
            Forall([x_, y_], 
                    Implies(parent_scorer(x_, y_), 
                            Not(parent_scorer(y_, x_))),
                    p=p),
            
            # Ancestor relation is single direction
            Forall([x_, y_], 
                    Implies(ancestor_scorer(x_, y_), 
                            Not(ancestor_scorer(y_, x_))),
                    p=p),
            
            # Sibling relation is commutative
            Forall([x_, y_],
                    Implies(sibling_scorer(x_, y_),
                            sibling_scorer(y_, x_)),
                    p=p),
            
            # Parent of parent is ancestor
            Forall([x_, y_, z_],
                    Implies(And(parent_scorer(x_, y_), 
                                parent_scorer(y_, z_)), 
                            ancestor_scorer(x_, z_)),
                    p=p),
            
            # The parent of ancestor is ancestor
            Forall([x_, y_, z_],
                    Implies(And(parent_scorer(x_, y_), 
                                ancestor_scorer(y_, z_)), 
                            ancestor_scorer(x_, z_)),
                    p=p),
            
            # The ancestor of parent is ancestor
            Forall([x_, y_, z_],
                    Implies(And(ancestor_scorer(x_, y_), 
                                parent_scorer(y_, z_)), 
                            ancestor_scorer(x_, z_)),
                    p=p),
            
            # Sibling relationship definition
            Forall([x_, y_, z_],
                    Implies(And(parent_scorer(x_, y_),
                                parent_scorer(x_, z_)),
                            sibling_scorer(y_, z_)),
                    p=p),
            
            # Every node has at least one parent, if not the parent node of the sampled node
            
            
            # Sibling relationship transitivity
            Forall([x_, y_, z_],
                    Implies(And(sibling_scorer(x_, y_), 
                                sibling_scorer(y_, z_)),
                            sibling_scorer(x_, z_)),
                    p=p),
            
            # Existing parental relationships
            SatAgg(*[parent_scorer(encoder[x], encoder[y]) for x, y in parent_edges]),
            
            # Negative samples of parental relationships
            SatAgg(*[Not(parent_scorer(encoder[x], encoder[y])) for x in nodes for y in nodes if (x, y) not in parent_edges]),
        )
        loss = 1. - sat_agg
        epoch_loss += loss.item()
        writer.add_scalar(f"{name}Loss / step", loss.item(), epoch*len(node_loader)+batch_idx)
        loss.backward()
        optimiser.step()
    writer.add_scalar(f"{name}Loss / epoch", epoch_loss/len(node_loader), epoch)  
    torch.cuda.empty_cache()
    
    

def pretrain_subgraph_sampler(config):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(f"tensorboard/omr/{current_time}_{config['d_model']}_{config['embd_mode']}_{config['scorer']}_{config['lr']}")
    model_save_dir = f"saved/pretraining/omr/{current_time}_{config['d_model']}_{config['embd_mode']}_{config['scorer']}_{config['lr']}"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    writer.add_text(tag="config", text_string=str(config))
    sym_graph = dill.load(open(f"src/pretraining/input/processed/diag_ontology.pkl", "rb"))
    med_graph = dill.load(open(f"src/pretraining/input/processed/atc_ontology.pkl", "rb"))
    pro_graph = dill.load(open(f"src/pretraining/input/processed/proc_ontology.pkl", "rb"))
    sym_nodes = list(sym_graph.get_nodes())
    pro_nodes = list(pro_graph.get_nodes())
    med_nodes = list(med_graph.get_nodes())
    indications = dill.load(open("src/pretraining/input/processed/medi_drug_diag_pairs.pkl", "rb"))
    if config['embd_mode'] != "random":
        sym_pretrained_weights = dill.load(open(f"src/pretraining/input/processed/diag_embd_{config['embd_mode']}.pkl", "rb"))
        pro_pretrained_weights = dill.load(open(f"src/pretraining/input/processed/proc_embd_{config['embd_mode']}.pkl", "rb"))
        med_pretrained_weights = dill.load(open(f"src/pretraining/input/processed/atc_embd_{config['embd_mode']}.pkl", "rb"))
    else:
        sym_pretrained_weights = torch.rand((len(sym_nodes), config["d_model"]))
        pro_pretrained_weights = torch.rand((len(pro_nodes), config["d_model"]))
        med_pretrained_weights = torch.rand((len(med_nodes), config["d_model"]))
    # encoder = pretrained_weights
    sym_encoder = [ltn.Constant(sym_pretrained_weights[i, :], trainable=True) for i in range(len(sym_nodes))]
    pro_encoder = [ltn.Constant(pro_pretrained_weights[i, :], trainable=True) for i in range(len(pro_nodes))]
    med_encoder = [ltn.Constant(med_pretrained_weights[i, :], trainable=True) for i in range(len(med_nodes))]
    
    d_embd = 768 if config['embd_mode'] != "random" else config["d_model"]     
    layer_sizes = [2*d_embd, d_embd, int(d_embd/2), int(d_embd/4), int(d_embd/8), int(d_embd/16), 1]
    sym_parent_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    sym_ancestor_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    sym_sibling_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    pro_parent_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    pro_ancestor_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    pro_sibling_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    med_parent_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    med_ancestor_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    med_sibling_scorer = ltn.Predicate(TaxoScorer(layer_sizes).to(DEVICE))
    indication_scorer = ltn.Predicate(IndicationScorer(layer_sizes).to(DEVICE))
    
    ind_collate = MediCollate(sym_graph)  
    med_node_loader = SubgraphSampler(med_graph, batch_size=config["batch_size"])
    sym_node_loader = SubgraphSampler(sym_graph, batch_size=config["batch_size"])
    pro_node_loader = SubgraphSampler(pro_graph, batch_size=config["batch_size"])
    indication_loader = DataLoader(indications, batch_size=config["batch_size"], shuffle=True, collate_fn=ind_collate)
    
    sym_params = [n.value for n in sym_encoder]+list(sym_parent_scorer.parameters())+list(sym_ancestor_scorer.parameters())+list(sym_sibling_scorer.parameters()) 
    pro_params = [n.value for n in pro_encoder]+list(pro_parent_scorer.parameters())+list(pro_ancestor_scorer.parameters())+list(pro_sibling_scorer.parameters()) 
    med_params = [n.value for n in med_encoder]+list(med_parent_scorer.parameters())+list(med_ancestor_scorer.parameters())+list(med_sibling_scorer.parameters()) 
    indi_params = list(indication_scorer.parameters())
    
    sym_optimiser = torch.optim.Adam(sym_params, lr=config["lr"])
    med_optimiser = torch.optim.Adam(med_params, lr=config["lr"])
    pro_optimiser = torch.optim.Adam(pro_params, lr=config["lr"])
    ind_optimiser = torch.optim.Adam(sym_params+med_params+indi_params, lr=config["lr"])
    
    SatAgg = ltn.fuzzy_ops.SatAgg()
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=config["p_exist"]), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=config["p_all"]), quantifier="f")
    print("START TRAINING")
    best_sat = 0
    for epoch in range(config["n_epochs"]):
        epoch_sym_loss = 0.0
        epoch_pro_loss = 0.0
        epoch_med_loss = 0.0
        epoch_ind_loss = 0.0
        if epoch <= 4:
            p_temp = 2
        elif epoch<= 9:
            p_temp = 4
        else:
            p_temp = 6

        for batch_idx, (nodes, parent_edges) in enumerate(sym_node_loader):
            sym_optimiser.zero_grad()
            x_ = ltn.Variable("x", torch.stack([n.value for n in [sym_encoder[n] for n in nodes]]).to(DEVICE))
            y_ = ltn.Variable("y", torch.stack([n.value for n in [sym_encoder[n] for n in nodes]]).to(DEVICE))
            z_ = ltn.Variable("z", torch.stack([n.value for n in [sym_encoder[n] for n in nodes]]).to(DEVICE))
            sat_agg = SatAgg(
                # Parental relation is non-reflexive
                Forall(x_, 
                       Not(sym_parent_scorer(x_, x_)),
                       p=p_temp),
                
                # Ancestor relation is non-reflexive
                Forall(x_, 
                       Not(sym_ancestor_scorer(x_, x_)),
                       p=p_temp),
                
                # Sibling relation is non-reflexive
                Forall(x_,
                       Not(sym_sibling_scorer(x_, x_)),
                       p=p_temp),
                
                # Parental relation is single direction
                Forall([x_, y_], 
                       Implies(sym_parent_scorer(x_, y_), 
                               Not(sym_parent_scorer(y_, x_))),
                       p=p_temp),
                
                # Ancestor relation is single direction
                Forall([x_, y_], 
                       Implies(sym_ancestor_scorer(x_, y_), 
                               Not(sym_ancestor_scorer(y_, x_))),
                       p=p_temp),
                
                # Sibling relation is commutative
                Forall([x_, y_],
                       Implies(sym_sibling_scorer(x_, y_),
                               sym_sibling_scorer(y_, x_)),
                       p=p_temp),
                
                # Parent of parent is ancestor
                Forall([x_, y_, z_],
                       Implies(And(sym_parent_scorer(x_, y_), 
                                   sym_parent_scorer(y_, z_)), 
                               sym_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # The parent of ancestor is ancestor
                Forall([x_, y_, z_],
                       Implies(And(sym_parent_scorer(x_, y_), 
                                   sym_ancestor_scorer(y_, z_)), 
                               sym_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # The ancestor of parent is ancestor
                Forall([x_, y_, z_],
                       Implies(And(sym_ancestor_scorer(x_, y_), 
                                   sym_parent_scorer(y_, z_)), 
                               sym_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # Sibling relationship definition
                Forall([x_, y_, z_],
                       Implies(And(sym_parent_scorer(x_, y_),
                                   sym_parent_scorer(x_, z_)),
                               sym_sibling_scorer(y_, z_)),
                       p=p_temp),
                
                # Every node has at least one parent, if not the parent node of the sampled node
                
                
                # Sibling relationship transitivity
                Forall([x_, y_, z_],
                       Implies(And(sym_sibling_scorer(x_, y_), 
                                   sym_sibling_scorer(y_, z_)),
                               sym_sibling_scorer(x_, z_)),
                       p=p_temp),
                
                # Existing parental relationships
                SatAgg(*[sym_parent_scorer(sym_encoder[x], sym_encoder[y]) for x, y in parent_edges]),
                
                # Negative samples of parental relationships
                SatAgg(*[Not(sym_parent_scorer(sym_encoder[x], sym_encoder[y])) for x in nodes for y in nodes if (x, y) not in parent_edges]),
            )
            loss = 1. - sat_agg
            epoch_sym_loss += loss.item()
            writer.add_scalar("SymLoss / step", loss.item(), epoch*int(len(sym_node_loader)/config["batch_size"])+batch_idx)
            # writer.add_scalar("SymLR / step", sym_scheduler.get_last_lr()[-1], epoch*len(sym_node_loader)+batch_idx)
            loss.backward()
            sym_optimiser.step()
            # sym_scheduler.step()
        writer.add_scalar("SymLoss / epoch", epoch_sym_loss/len(sym_node_loader), epoch)  
        torch.cuda.empty_cache()
        for batch_idx, (nodes, parent_edges) in enumerate(pro_node_loader):
            pro_optimiser.zero_grad()
            x_ = ltn.Variable("x", torch.stack([n.value for n in [pro_encoder[n] for n in nodes]]).to(DEVICE))
            y_ = ltn.Variable("y", torch.stack([n.value for n in [pro_encoder[n] for n in nodes]]).to(DEVICE))
            z_ = ltn.Variable("z", torch.stack([n.value for n in [pro_encoder[n] for n in nodes]]).to(DEVICE))
            sat_agg = SatAgg(
                # Parental relation is non-reflexive
                Forall(x_, 
                       Not(pro_parent_scorer(x_, x_)),
                       p=p_temp),
                
                # Ancestor relation is non-reflexive
                Forall(x_, 
                       Not(pro_ancestor_scorer(x_, x_)),
                       p=p_temp),
                
                # Sibling relation is non-reflexive
                Forall(x_,
                       Not(pro_sibling_scorer(x_, x_)),
                       p=p_temp),
                
                # Parental relation is single direction
                Forall([x_, y_], 
                       Implies(pro_parent_scorer(x_, y_), 
                               Not(pro_parent_scorer(y_, x_))),
                       p=p_temp),
                
                # Ancestor relation is single direction
                Forall([x_, y_], 
                       Implies(pro_ancestor_scorer(x_, y_), 
                               Not(pro_ancestor_scorer(y_, x_))),
                       p=p_temp),
                
                # Sibling relation is commutative
                Forall([x_, y_],
                       Implies(pro_sibling_scorer(x_, y_),
                               pro_sibling_scorer(y_, x_)),
                       p=p_temp),
                
                # Parent of parent is ancestor
                Forall([x_, y_, z_],
                       Implies(And(pro_parent_scorer(x_, y_), 
                                   pro_parent_scorer(y_, z_)), 
                               pro_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # The parent of ancestor is ancestor
                Forall([x_, y_, z_],
                       Implies(And(pro_parent_scorer(x_, y_), 
                                   pro_ancestor_scorer(y_, z_)), 
                               pro_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # The ancestor of parent is ancestor
                Forall([x_, y_, z_],
                       Implies(And(pro_ancestor_scorer(x_, y_), 
                                   pro_parent_scorer(y_, z_)), 
                               pro_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # Sibling relationship definition
                Forall([x_, y_, z_],
                       Implies(And(pro_parent_scorer(x_, y_),
                                   pro_parent_scorer(x_, z_)),
                               pro_sibling_scorer(y_, z_)),
                       p=p_temp),
                
                # Sibling relationship transitivity
                Forall([x_, y_, z_],
                       Implies(And(pro_sibling_scorer(x_, y_), 
                                   pro_sibling_scorer(y_, z_)),
                               pro_sibling_scorer(x_, z_)),
                       p=p_temp),
                
                # Existing parental relationships
                SatAgg(*[pro_parent_scorer(pro_encoder[x], pro_encoder[y]) for x, y in parent_edges]),
                
                # Negative samples of parental relationships
                SatAgg(*[Not(pro_parent_scorer(pro_encoder[x], pro_encoder[y])) for x in nodes for y in nodes if (x, y) not in parent_edges]),
            )
            loss = 1. - sat_agg
            epoch_pro_loss += loss.item()
            writer.add_scalar("ProLoss / step", loss.item(), epoch*int(len(pro_node_loader)/config["batch_size"])+batch_idx)
            loss.backward()
            pro_optimiser.step()  
        writer.add_scalar("ProLoss / epoch", epoch_pro_loss/len(pro_node_loader), epoch)
        torch.cuda.empty_cache()
        torch.save(torch.cat([n.value.unsqueeze(0) for n in pro_encoder],
                             dim=0),
                   f"{model_save_dir}/pro_embd.pt")
        for batch_idx, (nodes, parent_edges) in enumerate(med_node_loader):
            med_optimiser.zero_grad()
            x_ = ltn.Variable("x", torch.stack([n.value for n in [med_encoder[n] for n in nodes]]).to(DEVICE))
            y_ = ltn.Variable("y", torch.stack([n.value for n in [med_encoder[n] for n in nodes]]).to(DEVICE))
            z_ = ltn.Variable("z", torch.stack([n.value for n in [med_encoder[n] for n in nodes]]).to(DEVICE))
            sat_agg = SatAgg(
                # Parental relation is non-reflexive
                Forall(x_, 
                       Not(med_parent_scorer(x_, x_)),
                       p=p_temp),
                
                # Ancestor relation is non-reflexive
                Forall(x_, 
                       Not(med_ancestor_scorer(x_, x_)),
                       p=p_temp),
                
                # Sibling relation is non-reflexive
                Forall(x_,
                       Not(med_sibling_scorer(x_, x_)),
                       p=p_temp),
                
                # Parental relation is single direction
                Forall([x_, y_], 
                       Implies(med_parent_scorer(x_, y_), 
                               Not(med_parent_scorer(y_, x_))),
                       p=p_temp),
                
                # Ancestor relation is single direction
                Forall([x_, y_], 
                       Implies(med_ancestor_scorer(x_, y_), 
                               Not(med_ancestor_scorer(y_, x_))),
                       p=p_temp),
                
                # Sibling relation is commutative
                Forall([x_, y_],
                       Implies(med_sibling_scorer(x_, y_),
                               med_sibling_scorer(y_, x_)),
                       p=p_temp),
                
                # Parent of parent is ancestor
                Forall([x_, y_, z_],
                       Implies(And(med_parent_scorer(x_, y_), 
                                   med_parent_scorer(y_, z_)), 
                               med_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # The parent of ancestor is ancestor
                Forall([x_, y_, z_],
                       Implies(And(med_parent_scorer(x_, y_), 
                                   med_ancestor_scorer(y_, z_)), 
                               med_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # The ancestor of parent is ancestor
                Forall([x_, y_, z_],
                       Implies(And(med_ancestor_scorer(x_, y_), 
                                   med_parent_scorer(y_, z_)), 
                               med_ancestor_scorer(x_, z_)),
                       p=p_temp),
                
                # Sibling relationship definition
                Forall([x_, y_, z_],
                       Implies(And(med_parent_scorer(x_, y_),
                                   med_parent_scorer(x_, z_)),
                               med_sibling_scorer(y_, z_)),
                       p=p_temp),
                
                # # Negative samples for parent relationship
                # Forall([x_, y_, z_],
                #        Implies(And(med_sibling_scorer(x_, y_),
                #                    med_parent_scorer(x_, z_)),
                #                Not(med_parent_scorer(y_, z_)))),
                
                # Sibling relationship transitivity
                Forall([x_, y_, z_],
                       Implies(And(med_sibling_scorer(x_, y_), 
                                   med_sibling_scorer(y_, z_)),
                               med_sibling_scorer(x_, z_)),
                       p=p_temp),
                
                # Existing parental relationships
                SatAgg(*[med_parent_scorer(med_encoder[x], med_encoder[y]) for x, y in parent_edges]),
                
                # Negative samples of parental relationships
                SatAgg(*[Not(med_parent_scorer(med_encoder[x], med_encoder[y])) for x in nodes for y in nodes if (x, y) not in parent_edges]),
            )
            loss = 1. - sat_agg
            epoch_med_loss += loss.item()
            writer.add_scalar("MedLoss / step", loss.item(), epoch*int(len(med_node_loader)/config["batch_size"])+batch_idx)
            loss.backward()
            med_optimiser.step()
        writer.add_scalar("MedLoss / epoch", epoch_med_loss/len(med_node_loader), epoch)
        torch.cuda.empty_cache()
        # for batch_idx, (batch_edges, med_nodes, sym_nodes) in enumerate(indication_loader):
        #     ind_optimiser.zero_grad()
        #     sat_agg = SatAgg(
        #         SatAgg(*[indication_scorer(med_encoder[x], sym_encoder[y]) for x, y in batch_edges]),
        #     )
        #     loss = 1. - sat_agg
        #     epoch_ind_loss += loss.item()
        #     writer.add_scalar("IndLoss / step", loss.item(), epoch*len(indication_loader)+batch_idx)
        #     loss.backward()
        #     ind_optimiser.step()
            
        #     if sat_agg > best_sat:
        #         best_sat = sat_agg
        #         torch.save(torch.cat([n.value.unsqueeze(0) for n in sym_encoder], dim=0), 
        #                 f"{model_save_dir}/sym_embd.pt")
        #         torch.save(torch.cat([n.value.unsqueeze(0) for n in med_encoder], dim=0), 
        #                 f"{model_save_dir}/med_embd.pt")
        #         torch.save(torch.cat([n.value.unsqueeze(0) for n in pro_encoder], dim=0), 
        #                 f"{model_save_dir}/pro_embd.pt")
        ind_optimiser.zero_grad()
        sat_agg = SatAgg(
            SatAgg(*[indication_scorer(med_encoder[x], sym_encoder[y]) for x, y in indications]),
        )
        loss = 1. - sat_agg
        epoch_ind_loss += loss.item()
        writer.add_scalar("IndLoss", loss.item(), epoch)
        loss.backward()
        ind_optimiser.step()
        if sat_agg > best_sat:
            best_sat = sat_agg
            torch.save(torch.cat([n.value.unsqueeze(0) for n in sym_encoder], dim=0), 
                    f"{model_save_dir}/sym_embd.pt")
            torch.save(torch.cat([n.value.unsqueeze(0) for n in med_encoder], dim=0), 
                    f"{model_save_dir}/med_embd.pt")

        # writer.add_scalar("IndLoss / epoch", epoch_ind_loss/len(indication_loader), epoch)
        print(f"End of Epoch#{epoch+1}")
    

def collate_fn_medi(batch_edges):
    med_nodes = []
    sym_nodes = []
    
    for med, sym in batch_edges:
        med_nodes.append(med)
        sym_nodes.append(sym)
        
    return batch_edges, med_nodes, sym_nodes        


if __name__ == "__main__":
    config = {"n_epochs" :args.n_epochs,
              "k": args.k,
              "lr": args.lr,
              "batch_size": args.batch_size,
              "p_exist": args.p_exist,
              "p_all": args.p_all,
              "embd_mode": args.embd_mode,
              "d_model": args.d_model,
              "scorer": args.scorer}
    # pretrain_old(config=config)
    pretrain_subgraph_sampler(config=config)
    