import pandas as pd
from rdkit import Chem
import itertools
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from data_processing.compound import get_mol2vec_features, get_mol_features
from gensim.models import word2vec
# from Bio.PDB import PDBParser
import os

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# normal
# dti_train = pd.read_csv('data/Davis/split_data/train.csv')
# dti_test = pd.read_csv('data/Davis/split_data/test.csv')
# dti_val = pd.read_csv('data/Davis/split_data/valid.csv')

# pro-level sequence-iden
dti_train = pd.read_csv('data/data/davis_dataset/train.csv')
dti_test = pd.read_csv('data/data/davis_dataset/test.csv')
dti_val = pd.read_csv('data/data/davis_dataset/valid.csv')

df_drugs_smiles = pd.read_csv('data/data/all_smiles.csv')

batch_size = 32

drug_id_mol_graph_tup = [(smiles, Chem.MolFromSmiles(smiles.strip())) for smiles in df_drugs_smiles['smiles']]

ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


def get_bipartite_graph(num_drug_atoms, num_protein_atoms):
    # 创建二部图的边列表，不需要添加偏移量
    x1 = np.arange(0, num_drug_atoms)
    x2 = np.arange(0, num_protein_atoms)
    edge_list = torch.LongTensor(np.meshgrid(x1, x2))
    edge_list = torch.stack([edge_list[0].reshape(-1), edge_list[1].reshape(-1)], dim=0)

    return edge_list

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
            'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
            'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
            ]) + [atom.GetDegree() / 10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_atom_features(atom, mode='one_hot'):
    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature


def get_mol_edge_list_and_feat_mtx(mol_graph):
    try:
        features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
        features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
        _, features = zip(*features)
        features = torch.stack(features)

        edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
        undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list

        return undirected_edge_list.T, features
    except Exception as e:
        print(f"Error processing molecule: {e}")
        return None


MOL_EDGE_LIST_FEAT_MTX = {smiles: get_mol_edge_list_and_feat_mtx(mol)
                                for smiles, mol in drug_id_mol_graph_tup}
MOL_EDGE_LIST_FEAT_MTX = {smiles: mol for smiles, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_residue_centroids(structure):
    centroids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):  # 以α碳为代表，适用于标准氨基酸
                    centroid = residue['CA'].get_coord()
                    centroids.append(centroid)
    return np.array(centroids)

def calculate_edges_3d(centroids, threshold=5.0):
    edges_forward = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < threshold:
                edges_forward.append([i, j])  # 添加边 i 到 j

    # 将边复制并反转，以创建反向边
    edges_backward = [[j, i] for i, j in edges_forward]

    # 合并边并转换为NumPy数组
    edges = np.array(edges_forward + edges_backward).T
    return edges


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class ProteinFeatureManager():
    def __init__(self, data_path):
        map_data = pd.read_csv(os.path.join(data_path, 'mapping.csv'))
        sequence_to_id = {}
        if 'DB' in data_path:
            for seq, uniprot in zip(list(map_data['sequences']), list(map_data['uniprot'])):
                sequence_to_id[seq] = uniprot
        else:
            for seq, uniprot in zip(list(map_data['sequences']), list(map_data['uniprot_davis'])):
                sequence_to_id[seq] = uniprot
        self.sequence_to_id = sequence_to_id

        with open(os.path.join(data_path, 'protein_embedding/bert_embedding_Nongram.pkl'), 'rb') as f:
            self.bert_embed_dict = pickle.load(f)
        
        self.data_path = data_path
        self.seq_to_node = {}
        for seq in self.sequence_to_id:
            try:
                self.seq_to_node[seq] = np.load(os.path.join(self.data_path, 'protein_node_features', self.sequence_to_id[seq] + '.npy'))
            except:
                print(seq)

        self.seq_to_map = {}
        for seq in self.sequence_to_id:
            try:
                self.seq_to_map[seq] = np.load(os.path.join(self.data_path, 'protein_contact_map', self.sequence_to_id[seq] + '.npy'))
            except:
                print(seq)

        self.seq_to_contact = {}
        for seq in self.sequence_to_id:
            try:
                self.seq_to_contact[seq] = np.load(os.path.join(self.data_path, 'pdb_map_davis_700', self.sequence_to_id[seq] + '.npy'))
            except:
                print(seq)

    def get_node_features(self, sequence):
        return self.seq_to_node[sequence]

    def get_contact_map(self, sequence):
        return self.seq_to_map[sequence]

    # def get_contact_map(self, sequence):
    #     return np.load(os.path.join(self.data_path, 'pdb_map_davis_700_adj', self.sequence_to_id[sequence] + '.npy'))

    def get_pdb_map(self, sequence):
        return np.load(os.path.join(self.data_path, 'pdb_map_davis_700', self.sequence_to_id[sequence] + '.npy'))

    def get_pretrained_embedding(self, sequence):
        bert_embed = self.bert_embed_dict[sequence]
        return bert_embed


protein_feature_manager = ProteinFeatureManager(r'data/Davis')

class DrugDataset(Dataset):
    def __init__(self, compound_smiles, 
                protein_sequences, labels,
                mol_edge_list_feat_mtx, protein_feature_manager, threshold):
        self.compound_smiles = compound_smiles
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.protein_feature_manager = protein_feature_manager
        self.mol2vec_model = word2vec.Word2Vec.load("model_300dim.pkl")
        self.atom_dim = 34
        self.mol_edge_list_feat_mtx = mol_edge_list_feat_mtx
        self.threshold = threshold
        self.smiles_to_cosine = {}
        self.sequence_to_cosine = {}
        #----------------------------------------------------

    def __len__(self):
        return len(self.labels)

    # def generate_cosine_similarity_matrix(self):
    #     df = pd.read_csv('data/data/all_smiles.csv')
    #     smiles_list = df['smiles'].tolist()

    #     df_protein = pd.read_csv('data/Davis/all_proteins.csv')
    #     sequences_list = df_protein['sequence'].tolist()

    #     # 生成嵌入向量
    #     embeddings = []
    #     for smile in smiles_list:
    #         embedding = get_mol2vec_features(self.mol2vec_model, smile)
    #         avg_embedding = np.mean(embedding, axis=0)
    #         embeddings.append(avg_embedding)
    #     cosine_matrix = cosine_similarity(embeddings)

    #     embeddings_protein = []
    #     for sequence in sequences_list:
    #         embedding = self.protein_feature_manager.get_pretrained_embedding(sequence)
    #         avg_embedding = np.mean(embedding, axis=0)
    #         embeddings_protein.append(avg_embedding)
    #     cosine_matrix_protein = cosine_similarity(embeddings_protein)

    #     self.smiles_to_cosine = {smile: cosine_matrix[i] for i, smile in enumerate(smiles_list)}
    #     self.sequence_to_cosine = {sequence: cosine_matrix_protein[i] for i, sequence in enumerate(sequences_list)}

    def __getitem__(self, idx):
        compound_smile = self.compound_smiles[idx]
        protein_sequence = self.protein_sequences[idx]
        label = self.labels[idx]

        compound_node_features, compound_adj_matrix, _ = get_mol_features(compound_smile, self.atom_dim)
        compound_word_embedding = get_mol2vec_features(self.mol2vec_model, compound_smile)
        # print('compound_word_embedding:', compound_word_embedding.shape)

        protein_node_features = self.protein_feature_manager.get_node_features(protein_sequence)
        protein_contact_map = self.protein_feature_manager.get_contact_map(protein_sequence)
        protein_seq_embedding = self.protein_feature_manager.get_pretrained_embedding(protein_sequence)

        drug_edges, drug_nodes = self.mol_edge_list_feat_mtx[compound_smile]

        node_features = protein_node_features

        pdb_graph_edge = self.protein_feature_manager.get_pdb_map(protein_sequence)
        pdb_graph_edge = torch.tensor(pdb_graph_edge, dtype=torch.long)

        sample = {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_WORD_EMBEDDING': compound_word_embedding,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MAP': protein_contact_map,
            'PROTEIN_EMBEDDING': protein_seq_embedding,
            'SEQUENCE': protein_sequence,
            'compound_smile': compound_smile,
            'drug_edges': torch.tensor(drug_edges),
            'drug_nodes': torch.tensor(drug_nodes),
            'protein_sequence': protein_sequence,
            'node_features': torch.tensor(node_features, dtype=torch.float),
            'pdb_graph': pdb_graph_edge,
            'label': torch.tensor(label, dtype=torch.float)
            #'COSINE_SIMILARITY_EMBEDDING': torch.tensor(cosine_embedding, dtype=torch.float),
            #'COSINE_SEQUENCE_EMBEDDING': torch.tensor(cosine_embedding_sequence, dtype=torch.float)
        }

        return sample

    def collate_fn(self, batch):

        batch_size = len(batch)
        compound_node_nums = [item['COMPOUND_NODE_FEAT'].shape[0] for item in batch]
        protein_node_nums = [item['PROTEIN_NODE_FEAT'].shape[0] for item in batch]
        max_compound_len = max(compound_node_nums)
        max_protein_len = max(protein_node_nums)

        compound_node_features = torch.zeros((batch_size, max_compound_len, batch[0]['COMPOUND_NODE_FEAT'].shape[1]))
        compound_adj_matrix = torch.zeros((batch_size, max_compound_len, max_compound_len))
        compound_word_embedding = torch.zeros(
            (batch_size, max_compound_len, batch[0]['COMPOUND_WORD_EMBEDDING'].shape[1]))
        protein_node_features = torch.zeros((batch_size, max_protein_len, batch[0]['PROTEIN_NODE_FEAT'].shape[1]))
        protein_contact_map = torch.zeros((batch_size, max_protein_len, max_protein_len))
        protein_seq_embedding = torch.zeros((batch_size, max_protein_len, batch[0]['PROTEIN_EMBEDDING'].shape[1]))
        labels, seqs = list(), list()

        drug_graphs = []
        protein_graphs = []
        bipartite_graphs = []

        #smiles_avg = []
        #sequences_avg = []

        for i, item in enumerate(batch):
            v = item['COMPOUND_NODE_FEAT']
            compound_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['COMPOUND_ADJ']
            compound_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

            v = item['COMPOUND_WORD_EMBEDDING']
            compound_word_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)

            v = item['PROTEIN_NODE_FEAT']
            protein_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['PROTEIN_MAP']
            protein_contact_map[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

            v = item['PROTEIN_EMBEDDING']
            # if self.args.pretrained == 1 and self.args.objective == 'classification':
            v[:, 358] = (v[:, 358] - 7.8) / 6.5
            protein_seq_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)[:max_protein_len, :]

            labels.append(item['label'])
            seqs.append(item['SEQUENCE'])

            drug_graph = Data(x=item['drug_nodes'].float(), edge_index=item['drug_edges'].long())
            protein_graph = Data(x=item['node_features'].float(), edge_index=item['pdb_graph'])
            drug_graphs.append(drug_graph)
            protein_graphs.append(protein_graph)

            drug_features = item['drug_nodes'].float()
            protein_features = item['node_features'].float()

            num_drug_atoms = drug_features.size(0)
            num_protein_atoms = protein_features.size(0)

            bipartite_edge_index = get_bipartite_graph(num_drug_atoms, num_protein_atoms)
            bipartite_data = BipartiteData(edge_index=bipartite_edge_index, x_s=drug_features, x_t=protein_features)
            bipartite_graphs.append(bipartite_data)


        compound_node_nums = torch.LongTensor(compound_node_nums)
        protein_node_nums = torch.LongTensor(protein_node_nums)
        labels = torch.tensor(labels).type(torch.float32)

        sequences_batch = []

        drug_batch = Batch.from_data_list(drug_graphs)
        protein_batch = Batch.from_data_list(protein_graphs)
        bipartite_batch = Batch.from_data_list(bipartite_graphs)

        #smiles_avg_tensor = torch.stack(smiles_avg).to(device)
        #sequences_avg_tensor = torch.stack(sequences_avg).to(device)

        batch_data = {
            'COMPOUND_NODE_FEAT': compound_node_features.to(device),
            'COMPOUND_ADJ': compound_adj_matrix.to(device),
            'COMPOUND_WORD_EMBEDDING': compound_word_embedding.to(device),
            'COMPOUND_NODE_NUM': compound_node_nums.to(device),
            'PROTEIN_NODE_FEAT': protein_node_features.to(device),
            'PROTEIN_MAP': protein_contact_map.to(device),
            'PROTEIN_EMBEDDING': protein_seq_embedding.to(device),
            'PROTEIN_NODE_NUM': protein_node_nums.to(device),
            'labels': labels.to(device),
            'SEQUENCE': sequences_batch,
            'drug_graphs': drug_batch.to(device),
            'protein_graphs': protein_batch.to(device),
            'bipartite_graphs': bipartite_batch.to(device),
            #'smiles_avg': smiles_avg_tensor.to(device),
            #'sequences_avg': sequences_avg_tensor.to(device)
        }

        return batch_data

d_train, t_train, y_train = dti_train['COMPOUND_SMILES'], dti_train['PROTEIN_SEQUENCE'], dti_train['REG_LABEL']
d_test, t_test, y_test = dti_test['COMPOUND_SMILES'], dti_test['PROTEIN_SEQUENCE'], dti_test['REG_LABEL']
d_val, t_val, y_val = dti_val['COMPOUND_SMILES'], dti_val['PROTEIN_SEQUENCE'], dti_val['REG_LABEL']

train_dataset = DrugDataset(d_train, t_train, y_train, MOL_EDGE_LIST_FEAT_MTX, protein_feature_manager, 0.5)
test_dataset = DrugDataset(d_test, t_test, y_test, MOL_EDGE_LIST_FEAT_MTX, protein_feature_manager, 0.5)
val_dataset = DrugDataset(d_val, t_val, y_val, MOL_EDGE_LIST_FEAT_MTX, protein_feature_manager, 0.5)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)

train_std, train_mean = np.var(y_train), np.mean(y_train)
