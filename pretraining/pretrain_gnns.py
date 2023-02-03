from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.inits import glorot, zeros, uniform
from torch_geometric.utils import scatter, softmax, add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill
import networkx as nx
import argparse
from datetime import datetime
import os
from pretrain_logic_taxo_indi_encode import SubgraphSampler, IndicationScorer, TaxoScorer
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Run")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--embd_mode', type=str, default="random")
parser.add_argument('--scorer', type=str, default="neural")
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--gnn', type=str, default="gat")
args, unknown = parser.parse_known_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GNN(nn.Module):
    def __init__(self, ontology, dropout=0.3, device="cuda"):
        super(GNN, self).__init__()
        self.edges = np.array(list(ontology.get_edges())).transpose()
        self.edges = torch.LongTensor(self.edges).to(device)
        self.x = torch.eye(ontology.get_number_of_nodes()).to(device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self):
        return self.dropout(self.conv(self.x, self.edges))
        

class GCN(GNN):
    def __init__(self, ontology, in_channels, out_channels, aggr='add', dropout=0.3, device="cuda"):
        super().__init__(ontology, dropout, device)
        self.conv = GCNConv(in_channels,
                            out_channels,
                            aggr=aggr)
    

class GAT(GNN):
    def __init__(self, ontology, in_channels, out_channels, dropout=0.3, device="cuda"):
        super(GAT, self).__init__(ontology, dropout, device)
        self.conv = GATConv(in_channels,
                            out_channels,
                            heads=1)
    

class GNNScorer(TaxoScorer):
    def __init__(self, layer_sizes=[128, 64, 32, 8, 4, 1]):
        super().__init__(layer_sizes)
    
    def forward(self, node_pairs_embd):
        x = torch.cat([node_pairs_embd[:,0,:], node_pairs_embd[:,1,:]], dim=1)
        for layer in self.linear_layers[:-1]:
            x = self.elu(torch.tanh(layer(x)))
        x = torch.sigmoid(self.linear_layers[-1](x))
        return x 
    

class NodeLinkPredictor(nn.Module):
    def __init__(self, gnn=None, layer_sizes=[128, 64, 32, 8, 4, 1]):
        super(NodeLinkPredictor, self).__init__()
        self.gnn = gnn
        self.scorer = GNNScorer(layer_sizes=layer_sizes)
    
    def forward(self, node_pairs):
        node_pairs_embd = self.gnn()[node_pairs, :]
        link_score = self.scorer(node_pairs_embd)
        return link_score    
    
    def get_node_embeddings(self):
        self.eval()
        return self.gnn().detach().cpu()


class IndicationPredictor(nn.Module):
    def __init__(self, med_gnn, sym_gnn, layer_sizes=[128, 64, 32, 8, 4, 1]):
        super(IndicationPredictor, self).__init__()
        self.med_gnn = med_gnn
        self.sym_gnn = sym_gnn
        self.med_transform = nn.Linear(int(layer_sizes[0]/2), int(layer_sizes[0]/2))
        self.sym_transform = nn.Linear(int(layer_sizes[0]/2), int(layer_sizes[0]/2))
        self.scorer = GNNScorer(layer_sizes=layer_sizes)
        
    def forward(self, indi_pairs):
        med_embd = self.med_gnn()[indi_pairs[0], :]
        sym_embd = self.sym_gnn()[indi_pairs[1], :]
        med = self.med_transform(med_embd).unsqueeze(0).unsqueeze(0)
        sym = self.sym_transform(sym_embd).unsqueeze(0).unsqueeze(0)
        x = torch.cat([med, sym], dim=1)
        link_score = self.scorer(x)
        return link_score
    
    def get_node_embeddings(self):
        self.eval()
        return self.med_gnn().detach().cpu(), self.sym_gnn().detach().cpu()
        

def pretrain_gnn(config):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(f"tensorboard/{config['gnn']}/{current_time}_{config['d_model']}_{config['embd_mode']}_{config['scorer']}_{config['lr']}")
    model_save_dir = f"saved/pretraining/{config['gnn']}/{current_time}_medi_{config['d_model']}_{config['embd_mode']}_{config['scorer']}_{config['lr']}"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    writer.add_text(tag="config", 
                    text_string=str(config))
    sym_graph = dill.load(open(f"src/pretraining/input/processed/diag_ontology.pkl", "rb"))
    pro_graph = dill.load(open(f"src/pretraining/input/processed/proc_ontology.pkl", "rb"))
    med_graph = dill.load(open(f"src/pretraining/input/processed/atc_ontology.pkl", "rb"))
    indications = dill.load(open("src/pretraining/input/processed/medi_drug_diag_pairs.pkl", "rb"))
    d_embd = 768 if config['embd_mode'] != "random" else config["d_model"]
    layer_sizes = [2*d_embd, d_embd, int(d_embd/2), int(d_embd/4), int(d_embd/8), int(d_embd/16), 1]
    sym_node_loader = SubgraphSampler(sym_graph,
                                      batch_size=config["batch_size"])
    pro_node_loader = SubgraphSampler(pro_graph, 
                                      batch_size=config["batch_size"])
    med_node_loader = SubgraphSampler(med_graph, 
                                      batch_size=config["batch_size"])
    if config["gnn"]=="gat":
        sym_gnn = GAT(sym_graph,
                      in_channels=sym_graph.get_number_of_nodes(), 
                      out_channels=d_embd, 
                      device=DEVICE).to(DEVICE) 
        pro_gnn = GAT(pro_graph,
                      in_channels=pro_graph.get_number_of_nodes(),
                      out_channels=d_embd,
                      device=DEVICE).to(DEVICE)
        med_gnn = GAT(med_graph,
                      in_channels=med_graph.get_number_of_nodes(), 
                      out_channels=d_embd,
                      device=DEVICE).to(DEVICE)
    elif config["gnn"]=="gcn":
        sym_gnn = GCN(sym_graph,
                      in_channels=sym_graph.get_number_of_nodes(), 
                      out_channels=d_embd,
                      device=DEVICE).to(DEVICE) 
        pro_gnn = GCN(pro_graph,
                      in_channels=pro_graph.get_number_of_nodes(),
                      out_channels=d_embd,
                      device=DEVICE).to(DEVICE)
        med_gnn = GCN(med_graph,
                      in_channels=med_graph.get_number_of_nodes(), 
                      out_channels=d_embd,
                      device=DEVICE).to(DEVICE)
    sym_parent_scorer = NodeLinkPredictor(sym_gnn,
                                          layer_sizes).to(DEVICE)
    pro_parent_scorer = NodeLinkPredictor(pro_gnn,
                                          layer_sizes).to(DEVICE)
    med_parent_scorer = NodeLinkPredictor(med_gnn,
                                          layer_sizes).to(DEVICE)
    indication_scorer = IndicationPredictor(med_gnn,
                                            sym_gnn,
                                            layer_sizes).to(DEVICE)
    
    sym_optimiser = torch.optim.Adam(list(sym_parent_scorer.parameters()),
                                     lr=config["lr"])
    med_optimiser = torch.optim.Adam(list(pro_parent_scorer.parameters()),
                                     lr=config["lr"])
    pro_optimiser = torch.optim.Adam(list(med_parent_scorer.parameters()),
                                     lr=config["lr"])
    ind_optimiser = torch.optim.Adam(list(indication_scorer.parameters()),
                                     lr=config["lr"])
    sym_loss = nn.BCELoss()
    pro_loss = nn.BCELoss()
    med_loss = nn.BCELoss()
    ind_loss = nn.BCELoss()
    best_ind_loss = len(indications)
    best_pro_loss = len(pro_node_loader)*config["batch_size"]
    for epoch in range(config["n_epochs"]):
        epoch_sym_loss = 0.0
        epoch_pro_loss = 0.0
        epoch_med_loss = 0.0
        epoch_ind_loss = 0.0
        sym_parent_scorer.train()
        pro_parent_scorer.train()
        med_parent_scorer.train()
        indication_scorer.train()
        sym_optimiser.zero_grad()
        med_optimiser.zero_grad()
        pro_optimiser.zero_grad()
        ind_optimiser.zero_grad()
        for batch_idx, (_, parent_edges) in enumerate(sym_node_loader):
            parent_edges_tensor = torch.LongTensor(parent_edges).to(DEVICE)
            label =  torch.ones(parent_edges_tensor.shape[0]).unsqueeze(-1).to(DEVICE)
            link_score = sym_parent_scorer(parent_edges_tensor)
            loss = sym_loss(link_score, label)
            loss.backward()
            sym_optimiser.step()
            epoch_sym_loss += loss.item()
            writer.add_scalar("SymLoss / step", loss.item(), epoch*int(len(sym_node_loader)/config["batch_size"])+batch_idx)
        writer.add_scalar("SymLoss / epoch", epoch_sym_loss, epoch)
        for batch_idx, (_, parent_edges) in enumerate(pro_node_loader):
            parent_edges_tensor = torch.LongTensor(parent_edges).to(DEVICE)
            label =  torch.ones(parent_edges_tensor.shape[0]).unsqueeze(-1).to(DEVICE)
            link_score = pro_parent_scorer(parent_edges_tensor)
            loss = pro_loss(link_score, label)
            loss.backward()
            pro_optimiser.step()
            epoch_pro_loss += loss.item()
            writer.add_scalar("ProLoss / step", loss.item(), epoch*int(len(pro_node_loader)/config["batch_size"])+batch_idx)
        writer.add_scalar("ProLoss / epoch", epoch_pro_loss, epoch)
        if epoch_pro_loss < best_pro_loss:
            best_pro_loss = epoch_pro_loss
            torch.save(pro_parent_scorer.get_node_embeddings(), f"{model_save_dir}/pro_embd.pt")
        for batch_idx, (_, parent_edges) in enumerate(med_node_loader):
            parent_edges_tensor = torch.LongTensor(parent_edges).to(DEVICE)
            label =  torch.ones(parent_edges_tensor.shape[0]).unsqueeze(-1).to(DEVICE)
            link_score = med_parent_scorer(parent_edges_tensor)
            loss = med_loss(link_score, label)
            loss.backward()
            med_optimiser.step()
            epoch_med_loss += loss.item()
            writer.add_scalar("MedLoss / step", loss.item(), epoch*int(len(med_node_loader)/config["batch_size"])+batch_idx)
        writer.add_scalar("MedLoss / epoch", epoch_med_loss, epoch)
        for idx, indi_pair in enumerate(indications):
            indi_tensor = torch.LongTensor(indi_pair).to(DEVICE)
            label = torch.ones(1).unsqueeze(-1).to(DEVICE)
            indi_score = indication_scorer(indi_tensor)
            loss = ind_loss(indi_score, label)
            loss.backward()
            ind_optimiser.step()
            writer.add_scalar("IndLoss / step", loss.item(), epoch*len(indications)+idx)
            epoch_ind_loss += loss.item()
        writer.add_scalar("IndLoss / epoch", epoch_ind_loss, epoch)
        if epoch_ind_loss < best_ind_loss:
            best_ind_loss = epoch_ind_loss
            torch.save(indication_scorer.get_node_embeddings()[0], f"{model_save_dir}/med_embd.pt")
            torch.save(indication_scorer.get_node_embeddings()[1], f"{model_save_dir}/sym_embd.pt")       
        

if __name__ == "__main__":
    config = {"n_epochs" :args.n_epochs,
              "lr": args.lr,
              "batch_size": args.batch_size,
              "embd_mode": args.embd_mode,
              "d_model": args.d_model,
              "scorer": args.scorer,
              "gnn": args.gnn}
    # pretrain_old(config=config)
    pretrain_gnn(config=config)