# 只用分子的模型图和蛋白质的预测接触图
from torch import nn
from torch_geometric.nn import (GATConv,
                                GCNConv,
                                GINConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import torch
import numpy as np
import math

drop_out_rating = 0.3
n_heads = 3

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# model_type = 'gat'

class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim, mode):
        super().__init__()
        self.input_dim = input_dim
        if mode in [0, 1,2,3, 7 ,8, 9]:
            self.intra = GATConv(input_dim, 32, 2)
        elif mode == 4:
            print('GCN')
            self.intra = GCNConv(input_dim, 32 * 2)
        elif mode == 5:
            print('GIN')
            self.intra = GINConv(nn.Linear(input_dim, 32 * 2))

    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep, attn = self.intra(input_feature, edge_index, return_attention_weights=True)
        return intra_rep, attn

class BilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
        """
        x1: (batch_size, dim1)
        x2: (batch_size, dim2)
        return: (batch_size, out_dim)
        """
        out = self.bilinear(x1, x2)
        out = self.activation(out)
        return out

class InterGraphAttention(nn.Module):
    def __init__(self, input_dim, mode):
        super().__init__()
        self.input_dim = input_dim
        self.mode = mode
        if mode in [0,1,2,3,7,8,9]:
            self.inter = GATConv((input_dim, input_dim), 32, 2)
        elif mode == 4:
            print('gcn')
            self.inter1 = GCNConv(input_dim, 32 * 2)
            self.inter2 = GCNConv(input_dim, 32 * 2)
        elif mode == 5:
            print('gin')
            self.inter1 = GINConv(nn.Linear(input_dim, 32 * 2))
            self.inter2 = GINConv(nn.Linear(input_dim, 32 * 2))

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)

        if self.mode in [0,1,2,3,8,9]:
            t_rep = self.inter((h_input, t_input), edge_index)
            h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])
        elif self.mode in [4,5]:
            x1 = torch.cat((h_input, t_input))
            x2 = torch.cat((t_input, h_input))
            h_rep = self.inter1(x1, edge_index)[:h_input.shape[0]]
            t_rep = self.inter1(x2, edge_index[[1, 0]])[:t_input.shape[0]]
        return h_rep, t_rep

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, heads, tails, alpha_scores):
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        scores = heads @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        Args:
            embed_size: 输入向量的维度（d_model）
            heads: 注意力头的数量
        """
        super(SimpleSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # 每个头的维度

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义 Q, K, V 的线性变换层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # 最终输出层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, embed_size]
        Returns:
            注意力加权后的输出，形状为 [batch_size, seq_len, embed_size]
        """
        batch_size, seq_len, embed_size = x.shape

        # 1. 分割输入到多个头（reshape）
        x = x.reshape(batch_size, seq_len, self.heads, self.head_dim)  # [batch, seq_len, heads, head_dim]

        # 2. 计算 Q, K, V
        queries = self.queries(x)  # [batch, seq_len, heads, head_dim]
        keys = self.keys(x)        # [batch, seq_len, heads, head_dim]
        values = self.values(x)    # [batch, seq_len, heads, head_dim]

        # 3. 计算注意力分数 (Q * K^T)
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])  # [batch, heads, seq_len, seq_len]

        # 4. 缩放 + Softmax
        scaling = self.embed_size ** (1/2)
        attention = F.softmax(energy / scaling, dim=-1)  # [batch, heads, seq_len, seq_len]

        # 5. 注意力加权求和 (Attention * V)
        out = torch.einsum("bhqk,bkhd->bqhd", [attention, values])  # [batch, seq_len, heads, head_dim]

        # 6. 合并多头输出
        out = out.reshape(batch_size, seq_len, self.heads * self.head_dim)  # [batch, seq_len, embed_size]

        # 7. 通过全连接层
        out = self.fc_out(out)
        return out
    
class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features_drug, in_features_protein, head_out_feats, mode, attn=False):
        super().__init__()
        self.n_heads = n_heads
        self.in_features_drug = in_features_drug
        self.in_features_protein = in_features_protein
        self.out_features = head_out_feats

        if mode in [0, 1, 2, 3, 7, 8, 9]:
            self.feature_conv_drug = GATConv(in_features_drug, head_out_feats, n_heads)
            self.feature_conv_protein = GATConv(in_features_protein, head_out_feats, n_heads)
        elif mode == 4:
            print('GCN')
            self.feature_conv_drug = GCNConv(in_features_drug, head_out_feats * n_heads)
            self.feature_conv_protein = GCNConv(in_features_protein, head_out_feats * n_heads)
        elif mode == 5:
            print('GIN')
            linear1 = nn.Linear(in_features_drug, head_out_feats * n_heads)
            self.feature_conv_drug = GINConv(linear1)
            linear2 = nn.Linear(in_features_protein, head_out_feats * n_heads)
            self.feature_conv_protein = GINConv(linear2)
            
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads, mode)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads, mode)

        if not attn:
            self.readout_1 = SAGPooling(head_out_feats*n_heads, min_score=-1)
            self.readout_2 = SAGPooling(head_out_feats*n_heads, min_score=-1)
        else:
            self.readout_1 = SAGPooling(head_out_feats*n_heads, min_score=0.3)
            self.readout_2 = SAGPooling(head_out_feats*n_heads, min_score=0.3)

    def forward(self, h_data, t_data, b_graph):
        h_data.x = self.feature_conv_drug(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv_protein(t_data.x, t_data.edge_index)

        h_intraRep, _ = self.intraAtt(h_data)
        t_intraRep, attn = self.intraAtt(t_data)

        if b_graph is not None:
            h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)
            h_rep = torch.cat([h_intraRep, h_interRep], 1)
            t_rep = torch.cat([t_intraRep, t_interRep], 1)
        else:
            h_rep = torch.cat([h_intraRep, h_intraRep], 1)
            t_rep = torch.cat([t_intraRep, t_intraRep], 1)

        h_data.x = h_rep
        t_data.x = t_rep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, h_perm, h_att_scores = self.readout_1(h_data.x,h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, t_perm, t_att_scores = self.readout_2(t_data.x,t_data.edge_index, batch=t_data.batch)


        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        pool_info = {
            "drug_perm": h_perm,
            "drug_score": h_att_scores,
            "drug_pool_batch": h_att_batch,
            "drug_full_batch": h_data.batch,

            "protein_perm": t_perm,
            "protein_score": t_att_scores,
            "protein_pool_batch": t_att_batch,
            "protein_full_batch": t_data.batch,
        }

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb, pool_info

class My_Model(nn.Module):
    def __init__(self, in_features_drug, in_features_protein, hidd_dim, kge_dim, heads_out_feat_params, blocks_params, attn=False):
        super(My_Model, self).__init__()

        self.in_features_drug = in_features_drug
        self.in_features_protein = in_features_protein
        self.hidd_dim = hidd_dim
        self.n_blocks = len(blocks_params)
        self.kge_dim = kge_dim

        self.initial_norm_drug = LayerNorm(self.in_features_drug)
        self.initial_norm_protein = LayerNorm(self.in_features_protein)
        
        out_dim = 1

        self.mol2vec_embedding_dim = 300
        self.mol2vec_fc = nn.Linear(self.mol2vec_embedding_dim, self.mol2vec_embedding_dim)
        self.mol_attn = SimpleSelfAttention(self.mol2vec_embedding_dim, 4)

        self.embedding_dim = 768
        self.hid_dim = 64
        self.tape_fc = nn.Linear(self.embedding_dim, self.hid_dim * 4)
        self.tape_attn = SimpleSelfAttention(self.hid_dim * 4, 4)

        self.out = nn.Sequential(
                nn.Linear(self.mol2vec_embedding_dim + self.hid_dim * 4 + 4 * hidd_dim, len(blocks_params) * hidd_dim),
                nn.ReLU(),
                nn.Linear(len(blocks_params) * hidd_dim, hidd_dim),
                nn.ReLU(),
                nn.Linear(hidd_dim, hidd_dim // 4),
                nn.ReLU(),
                nn.Linear(hidd_dim // 4, out_dim)
        )

        self.blocks = []
        self.net_norms = ModuleList()

        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features_drug, in_features_protein, head_out_feats, 0, attn)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features_drug = head_out_feats * n_heads
            in_features_protein = head_out_feats * n_heads

    def forward(self, drug_graphs, protein_graphs, bi_graphs, word_embedding, protein_tape, task='repr'):
        h_data, t_data, b_graph = drug_graphs, protein_graphs, bi_graphs
        h_data.x = self.initial_norm_drug(h_data.x)
        t_data.x = self.initial_norm_protein(t_data.x)
        protein_attn_list = []
        
        mol2vec_embedding = self.mol2vec_fc(word_embedding)
        mol2vec_embedding = self.mol_attn(mol2vec_embedding).mean(1)
        
        tape_embedding = self.tape_fc(protein_tape)
        tape_embedding = self.tape_attn(tape_embedding).mean(1)
        
        repr_h = []
        repr_t = []

        # Separate processing for drugs and proteins
        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            protein_attn_list.append(out[4])

            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = torch.cat([repr_h[:, 0, :], repr_h[:, -1, :]], -1)
        # print(kge_heads.shape)
        kge_tails = torch.cat([repr_t[:, 0, :], repr_t[:, -1, :]], -1)

        features = torch.cat([kge_heads, kge_tails, mol2vec_embedding, tape_embedding], -1)

        scores = self.out(features)
        if task=='logic':
            scores = self.sig(scores)
        if task == 'attn':
            return scores, protein_attn_list
        return scores