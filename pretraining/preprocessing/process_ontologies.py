from typing import Dict, List, Tuple
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import dill


class Ontology():
    def __init__(self, name:str):
        self.name = name
        self.graph = nx.DiGraph()
    
    def get_number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def get_number_of_edges(self) -> int:
        return self.graph.number_of_edges()
    
    def get_nodes(self) -> List:
        return self.graph.nodes()
    
    def get_edges(self) -> List:
        return self.graph.edges()

    def add_node(self, node, **kwargs):
        self.graph.add_node(node, **kwargs)
    
    def add_edge(self, start, end):
        self.graph.add_edge(start, end)
        
    def add_nodes_from_list(self, node_list):
        self.graph.add_nodes_from(node_list)

    def add_edges_from_list(self, edge_list):
        self.graph.add_edges_from(edge_list)
    
    def is_valid_dag(self):
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_node_depth(self, node: int) -> int:
        depth = len(self.get_descending_path(self.get_root(), node)) - 1
        
        return depth
    
    def draw_graph(self, filepath="output"):
        nx.draw(self.graph,
                node_size=0.1,
                arrows=False,
                width=0.2, 
                font_size=8,
                alpha=0.6)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.savefig(f"{filepath}/{self.name}.pdf", format='pdf', dpi=1000)
        
    def get_lca(self, node:int, other:int) -> int:
        node_predecessors = self.get_descending_path(ancestor=self.get_root(), node=node)
        other_predecessors = self.get_descending_path(ancestor=self.get_root(), node=other)
        lca = 0
        # zip() will clip the predecessing path
        # and the constant update of nearest_predecessor
        # will make sure it is the shortest path
        for node_pred, other_pred in zip(node_predecessors, other_predecessors):
            if node_pred == other_pred:
                lca = node_pred
        
        return lca
        
    def get_smallest_subgraph(self, node:int, other:int) -> nx.DiGraph:
        lca = self.get_lca(node=node, other=other)
        lca_to_node_path = self.get_descending_path(ancestor=lca, node=node)
        lca_to_other_path = self.get_descending_path(ancestor=lca, node=other)
        smallest_subgraph =  nx.subgraph_view(self.graph,
                                              filter_node=lambda x: x in set(lca_to_node_path+lca_to_other_path))
        
        assert nx.is_directed_acyclic_graph(smallest_subgraph) 
        
        return smallest_subgraph
    
    def get_descending_subgraph(self, node: int) -> nx.DiGraph:
        successors = list(nx.dfs_preorder_nodes(self.graph, source=node))
        descending_subgraph = nx.subgraph_view(self.graph, filter_node=lambda x: x in successors)
        assert nx.is_directed_acyclic_graph(descending_subgraph)
        return descending_subgraph
    
    def get_root(self):
        root = None
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                root =  node
        return root
    
    def get_leaf_nodes(self) -> List:
        leaf_nodes = []
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0:
                leaf_nodes.append(node)
        return leaf_nodes
    
    def get_non_root_nodes(self) -> List:
        non_root = []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) > 0:
                non_root.append(node)
        return non_root
    
    def get_parent(self, node: int) -> int:
        return list(self.graph.predecessors(node))[0]
    
    def get_nodes_at_level(self, level: int) -> List:
        nodes = [node for node in self.graph.nodes() if self.get_node_depth(node)==level]
        return nodes
    
    def is_leaf_node(self, node: int) -> bool:
        return self.graph.out_degree(node) == 0
    
    def get_sucessors(self, node: int) -> List:
        return list(self.graph.successors(node))
    
    def get_descending_path(self, ancestor: int, node: int) -> List:
        return nx.shortest_path(self.graph, ancestor, node)
    
    def get_immediate_predecessor(self, node: int) -> List:
        return list(self.graph.predecessors(node))

    def get_siblings(self, node: int) -> List:
        parent = self.get_parent(node)
        siblings = self.get_sucessors(parent)
        return [s for s in siblings if s != node]
        
    def relabel_to_consecutive(self):
        relabelled = nx.convert_node_labels_to_integers(self.graph)
        self.graph = relabelled


def preprocess_ontology(file_path: str) -> Tuple[Dict, List, List, List]:
    raw_ontology = pd.read_csv(file_path, 
                               index_col=False,)
    # Renaming these columns for easier indexing
    raw_ontology.rename(columns={"Class ID": "ID", 
                                 "Preferred Label": "NAME", 
                                 "Parents": "PARENT"}, 
                        inplace=True)
    
    # Drop irrelevant columns
    raw_ontology.drop(columns=["Synonyms", "Definitions", "CUI", "Semantic Types", "Semantic type UMLS property"], 
                      inplace=True)
    
    # Drop na in these key columns
    raw_ontology.dropna(subset=["PARENT", "ID", "NAME", "Obsolete"], inplace=True)
    
    # Drop obsolete entries
    raw_ontology = raw_ontology[~raw_ontology["Obsolete"] == True]
    
    # Drop the "Obsolete" column
    raw_ontology.drop(columns=["Obsolete"], inplace=True)
    
    # Seperate the actual code of the nodes
    raw_ontology["ID"] = raw_ontology["ID"].apply(lambda x: x.split("/")[-1].strip())
    raw_ontology["PARENT"] = raw_ontology["PARENT"].apply(lambda x: x.split("/")[-1].strip())
    
    # Turn node name to lowercase
    raw_ontology["NAME"] = raw_ontology["NAME"].apply(lambda x: x.lower())
    
    # Get rid of the nodes that do not belong to the ontology
    raw_ontology = raw_ontology[~raw_ontology["ID"].str.startswith("T")]
    
    # Initialised three conversion variables
    code2index = {"owl#Thing": 0}
    index2code = ["owl#Thing"]
    index2name = ["owl#Thing"]
    
    # Populate these four conversions
    for index, (id, name) in enumerate(zip(raw_ontology["ID"], raw_ontology["NAME"])):
        code2index[id] = index + 1
        index2code.append(id)
        index2name.append(name)
    
    # Get the list of edges
    edges = [(code2index[parent], code2index[child]) for child, parent in zip(raw_ontology["ID"], raw_ontology["PARENT"])]
    
    return code2index, index2code, index2name, edges


if __name__ == "__main__":
    atc_code2index, atc_index2code, atc_index2name, atc_edges  = preprocess_ontology("src/pretraining/input/raw/ontologies/ATC.csv")
    icd_code2index, icd_index2code, icd_index2name, icd_edges = preprocess_ontology("src/pretraining/input/raw/ontologies/ICD9CM.csv")
    
    print(f"The ATC ontology has {len(atc_code2index)} nodes and {len(atc_edges)} edges")
    print(f"The ICD ontology has {len(icd_code2index)} nodes and {len(icd_edges)} edges")
    
    atc = Ontology(name="atc")
    icd = Ontology(name="icd")
    
    for index, (code, name) in enumerate(zip(atc_index2code, atc_index2name)):
        atc.add_node(index, code=code, name=name)
    atc.add_edges_from_list(atc_edges)
    for index, (code, name) in enumerate(zip(icd_index2code, icd_index2name)):
        icd.add_node(index, code=code, name=name)
    icd.add_edges_from_list(icd_edges)
    
    # Test if the graphs have been correctly constructed
    assert atc.get_number_of_nodes() == len(atc_code2index)
    assert atc.get_number_of_edges() == len(atc_edges)
    assert icd.get_number_of_nodes() == len(icd_code2index)
    assert icd.get_number_of_edges() == len(icd_edges)
    assert atc.is_valid_dag()
    assert icd.is_valid_dag()
    
    # Test node attribute
    assert atc.graph.nodes[100] == {"code" : atc_index2code[100], "name": atc_index2name[100]}
    assert icd.graph.nodes[100] == {"code" : icd_index2code[100], "name": icd_index2name[100]}
    
    # Test root nodes
    assert atc.graph.nodes[atc.get_root()] == {'code': 'owl#Thing', 'name': 'owl#Thing'}
    assert icd.graph.nodes[icd.get_root()] == {'code': 'owl#Thing', 'name': 'owl#Thing'}
    
    # Get predecessors
    print(atc.get_descending_path(0, 4876))
    print(atc.get_immediate_predecessor(4876))
    
    # Get successors
    assert len(atc.get_sucessors(4876)) == 0
    
    # Get subgraphs
    labels = {index: code for index, code in enumerate(atc_index2code)}
    print(atc.get_smallest_subgraph(atc_code2index["D02AB"], atc_code2index["D01BA"]).nodes)
    smallest_subgraph = atc.get_smallest_subgraph(atc_code2index["C01AA01"],
                                                  atc_code2index["D01AA01"])
    node_labels = nx.get_node_attributes(smallest_subgraph, "name")
    # nx.draw(smallest_subgraph, 
    #         with_labels=True, 
    #         labels=node_labels)
    # plt.show()
    
    # Test the get depth function 
    for index, node in enumerate(atc_index2code):
        if node == "owl#Thing":
            assert atc.get_node_depth(index) == 0
        elif len(node) == 1:
            assert atc.get_node_depth(index) == 1
        elif len(node) == 3:
            assert atc.get_node_depth(index) == 2
        elif len(node) == 4:
            assert atc.get_node_depth(index) == 3
        elif len(node) == 5:
            assert atc.get_node_depth(index) == 4
        elif len(node) == 7:
            assert atc.get_node_depth(index) == 5
            
    # Save the ontologies and conversion tables
    dill.dump(atc, open("src/pretraining/input/processed/atc_ontology.pkl", "wb"))
    dill.dump(atc_code2index, open("src/pretraining/input/processed/atc_code2index.pkl", "wb"))
    dill.dump(atc_index2code, open("src/pretraining/input/processed/atc_index2code.pkl", "wb"))
    dill.dump(atc_index2name, open("src/pretraining/input/processed/atc_index2name.pkl", "wb"))
    dill.dump(icd, open("src/pretraining/input/processed/icd_ontology.pkl", "wb"))
    dill.dump(icd_code2index, open("src/pretraining/input/processed/icd_code2index.pkl", "wb"))
    dill.dump(icd_index2code, open("src/pretraining/input/processed/icd_index2code.pkl", "wb"))
    dill.dump(icd_index2name, open("src/pretraining/input/processed/icd_index2name.pkl", "wb"))
    
    # Seperate the diagnosis and procedure taxonomy
    proc = Ontology(name="proc")
    diag = Ontology(name="diag")
    proc_subontology = icd.get_descending_subgraph(icd_code2index["00-99.99"])
    proc.graph = proc_subontology
    diag_subontology = icd.graph.copy()
    diag_subontology.remove_nodes_from(proc_subontology.nodes)
    diag.graph = diag_subontology
    assert len(diag.graph.nodes)+len(proc.graph.nodes)==len(icd.graph.nodes)
    assert nx.is_directed_acyclic_graph(diag.graph)
    assert nx.is_directed_acyclic_graph(proc.graph)
    
    # Relabel these two taxonomies to save space
    proc.relabel_to_consecutive()
    diag.relabel_to_consecutive()
    
    # Get the code2index, index2code, index2name for these two taxonomies
    diag_code2index = {code: index for index, code in nx.get_node_attributes(diag.graph, "code").items()}
    diag_index2code = [code for idx, code in nx.get_node_attributes(diag.graph, "code").items()]
    diag_index2name = [name for idx, name in nx.get_node_attributes(diag.graph, "name").items()]
    proc_code2index = {code: index for index, code in nx.get_node_attributes(proc.graph, "code").items()}
    proc_index2code = [code for idx, code in nx.get_node_attributes(proc.graph, "code").items()]
    proc_index2name = [name for idx, name in nx.get_node_attributes(proc.graph, "name").items()]
    
    # Save these two taxonomies and conversion tables
    dill.dump(diag, open("src/pretraining/input/processed/diag_ontology.pkl", "wb"))
    dill.dump(diag_code2index, open("src/pretraining/input/processed/diag_code2index.pkl", "wb"))
    dill.dump(diag_index2code, open("src/pretraining/input/processed/diag_index2code.pkl", "wb"))
    dill.dump(diag_index2name, open("src/pretraining/input/processed/diag_index2name.pkl", "wb"))
    dill.dump(proc, open("src/pretraining/input/processed/proc_ontology.pkl", "wb"))
    dill.dump(proc_code2index, open("src/pretraining/input/processed/proc_code2index.pkl", "wb"))
    dill.dump(proc_index2code, open("src/pretraining/input/processed/proc_index2code.pkl", "wb"))
    dill.dump(proc_index2name, open("src/pretraining/input/processed/proc_index2name.pkl", "wb"))
    
    
    # max_diag_depth = max([diag.get_node_depth(i) for i in diag_code2index.values()])
    # max_proc_depth = max([proc.get_node_depth(i) for i in proc_code2index.values()])
    # print(f"Diagnosis ontology has the max depth of {max_diag_depth}")
    # print(f"Procedure ontology has the max depth of {max_proc_depth}")
    
    # # diag_node_at_levels = [len([index for index in diag_code2index.values() if diag.get_node_depth(index) == level]) for level in range(1, max_diag_depth+1)]
    # # proc_node_at_levels = [len([index for index in proc_code2index.values() if proc.get_node_depth(index) == level]) for level in range(1, max_proc_depth+1)]
    
    # # print(diag_node_at_levels)
    # # print(proc_node_at_levels)
    
    # # print(len(diag.get_leaf_nodes()))
    # # print(len(proc.get_leaf_nodes()))
    
    # diag_in_ehr = pd.read_csv("input/processed/DIAGNOSES.csv")["ICD9_CODE"].unique()
    # proc_in_ehr =pd.read_csv("input/processed/PROCEDURES.csv")["ICD9_CODE"].unique()
    
    # print(len(diag_in_ehr))
    # print(len([diag_code for diag_code in diag_in_ehr if not diag.is_leaf_node(diag_code)]))
    # print(len(proc_in_ehr))
    # print(len([proc_code for proc_code in proc_in_ehr if not proc.is_leaf_node(proc_code)]))
    
    atc3_above_codes = [idx for code, idx in atc_code2index.items() if len(code)<=4] + [0]
    atc4_above_codes = [idx for code, idx in atc_code2index.items() if len(code)<=5] + [0]
    
    atc_3_graph = atc.graph.subgraph(atc3_above_codes)
    atc_4_graph = atc.graph.subgraph(atc4_above_codes)
    print(len(atc_3_graph.nodes))
    print(len(atc_3_graph.edges))
    print(len(atc_4_graph.nodes))
    print(len(atc_4_graph.edges))
    
    atc3 = Ontology(name="atc3")
    atc3.graph = atc_3_graph
    atc3.relabel_to_consecutive()
    atc3_code2index = {code: index for index, code in nx.get_node_attributes(atc3.graph, "code").items()}
    atc3_index2code = [code for idx, code in nx.get_node_attributes(atc3.graph, "code").items()]
    atc3_index2name = [name for idx, name in nx.get_node_attributes(atc3.graph, "name").items()]
    dill.dump(atc3, open("src/pretraining/input/processed/atc3_ontology.pkl", "wb"))
    dill.dump(atc3_code2index, open("src/pretraining/input/processed/atc3_code2index.pkl", "wb"))
    dill.dump(atc3_index2code, open("src/pretraining/input/processed/atc3_index2code.pkl", "wb"))
    dill.dump(atc3_index2name, open("src/pretraining/input/processed/atc3_index2name.pkl", "wb"))
    
    atc4 = Ontology(name="atc4")
    atc4.graph = atc_4_graph
    atc4.relabel_to_consecutive()
    atc4_code2index = {code: index for index, code in nx.get_node_attributes(atc4.graph, "code").items()}
    atc4_index2code = [code for idx, code in nx.get_node_attributes(atc4.graph, "code").items()]
    atc4_index2name = [name for idx, name in nx.get_node_attributes(atc4.graph, "name").items()]
    dill.dump(atc4, open("src/pretraining/input/processed/atc4_ontology.pkl", "wb"))
    dill.dump(atc4_code2index, open("src/pretraining/input/processed/atc4_code2index.pkl", "wb"))
    dill.dump(atc4_index2code, open("src/pretraining/input/processed/atc4_index2code.pkl", "wb"))
    dill.dump(atc4_index2name, open("src/pretraining/input/processed/atc4_index2name.pkl", "wb"))
    
    atc3_codes_idx = [idx for code, idx in atc_code2index.items() if len(code)==4]
    atc4_codes_idx = [idx for code, idx in atc_code2index.items() if len(code)==5]
    atc5_codes_idx = [idx for code, idx in atc_code2index.items() if len(code)==7]
    dill.dump(atc3_codes_idx, open("src/pretraining/input/processed/atc3_codes_idx.pkl", "wb"))
    dill.dump(atc4_codes_idx, open("src/pretraining/input/processed/atc4_codes_idx.pkl", "wb"))
    dill.dump(atc5_codes_idx, open("src/pretraining/input/processed/atc5_codes_idx.pkl", "wb"))