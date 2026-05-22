import os
import torch
from data_process import train_loader, test_loader, val_loader, train_std, train_mean, TOTAL_ATOM_FEATS
from model_upload import My_Model
from torch.optim import Adam
import torch.nn as nn
# from data_process_bi import TOTAL_ATOM_FEATS
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from rdkit import RDLogger
# import pandas as pd
import random
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore')
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_all_seed(42)

epo_num = 10
learning_rate = 0.0002

dim_drug = TOTAL_ATOM_FEATS
dim_protein = 64
hidd_dim = 128
kge_dim = 128

model = My_Model(dim_drug, dim_protein, hidd_dim, kge_dim, heads_out_feat_params=[64, 64, 64, 64], blocks_params=[2, 2, 2, 2]).to(device)

def train(model):
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    mse = 1000
    rmse = 1000  # 计算均方根误差（RMSE）
    r2 = 0

    pearson_corr= 0
    spearman_corr= 0
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("total params:", total)
    print("trainable params:", trainable)

    with tqdm(total=epo_num) as pbar:
        for epoch in range(epo_num):
            model.train()
            num_batches = len(train_loader)
            for batch_idx, batch_data in enumerate(train_loader):
                drug_graphs = batch_data['drug_graphs'].to(device)
                protein_graphs = batch_data['protein_graphs'].to(device)
                bi_graphs = batch_data['bipartite_graphs'].to(device)
                protein_tape = batch_data['PROTEIN_EMBEDDING'].to(device)
                word_embedding = batch_data['COMPOUND_WORD_EMBEDDING'].to(device)
                labels = batch_data['labels'].to(device).float()
                optimizer.zero_grad()

                outputs = model(drug_graphs, protein_graphs, bi_graphs, word_embedding, protein_tape)  # Adjust depending on model's forward method
                labels = (labels - train_mean) / train_std
                loss = criterion(outputs.squeeze(), labels)

                loss.backward()
                optimizer.step()

                # 计算当前进度
                progress = int(batch_idx / num_batches * 100)
                # print(f'epoch:{epoch}---progress:{progress}%')
                
                pbar.set_postfix(epoch=epoch, progress=f'{progress}%', train_loss=loss.item(), mse=mse, rmse=rmse, r2=r2, pear=pearson_corr, spear=spearman_corr)

            model.eval()
            with torch.no_grad():
                all_labels = []
                all_outputs = []
                all_predictions = []
                all_scores = []
                for batch_idx, batch_data in enumerate(val_loader):
                    drug_graphs = batch_data['drug_graphs'].to(device)
                    protein_graphs = batch_data['protein_graphs'].to(device)
                    bi_graphs = batch_data['bipartite_graphs'].to(device)
                    protein_tape = batch_data['PROTEIN_EMBEDDING'].to(device)
                    word_embedding = batch_data['COMPOUND_WORD_EMBEDDING'].to(device)
                    labels = batch_data['labels'].to(device).float()  # 注意标签类型改为float

                    outputs = model(drug_graphs, protein_graphs, bi_graphs, word_embedding, protein_tape).squeeze(-1)
                    outputs = outputs * train_std + train_mean
                    all_outputs = np.array(all_outputs)
                    all_labels = np.array(all_labels)
                    # 计算均方误差（MSE）和 R^2 值
                    mse_epoch = mean_squared_error(all_labels, all_outputs)
                    rmse_epoch = np.sqrt(mse)  # 计算均方根误差（RMSE）
                    r2 = r2_score(all_labels, all_outputs)

                    # 计算Pearson和Spearman相关系数
                    pearson_corr_epoch, _ = pearsonr(all_labels, all_outputs)
                    spearman_corr_epoch, _ = spearmanr(all_labels, all_outputs)
                    if mse_epoch < mse:
                    # if pearson_corr_epoch > pearson_corr:
                        mse = mse_epoch
                        rmse = rmse_epoch
                        pearson_corr = pearson_corr_epoch
                        spearman_corr = spearman_corr_epoch
                        print('Save Checkpoint')
                        torch.save(model.state_dict(), f'./runs2/model_0_pro.pth')
                    pbar.set_postfix(epoch=epoch, train_loss=loss.item(), mse=mse, rmse=rmse, r2=r2, pear=pearson_corr, spear=spearman_corr)
                    pbar.update()
                    
                    print('epoch:',epoch)
                    print('MSE:', mse)
                    print('RMSE:', rmse)
                    print('r2:', r2)
                    print('pearson_corr:', pearson_corr)
                    print('spearman_corr:', spearman_corr)

    print("Saved trained model")

def test(model):
    model = My_Model(dim_drug, dim_protein, hidd_dim, kge_dim, heads_out_feat_params=[64, 64, 64, 64], blocks_params=[2,2,2,2],attn=False).to(device)
    model.load_state_dict(torch.load(f'./runs2/model_0_pro.pth', map_location=device))
    model.eval()

    trace_loader = test_loader
    with torch.no_grad():
        all_labels = []
        all_outputs = []

        for batch_idx, batch_data in enumerate(trace_loader):
            drug_graphs = batch_data['drug_graphs'].to(device)
            protein_graphs = batch_data['protein_graphs'].to(device)
            bi_graphs = batch_data['bipartite_graphs'].to(device)
            protein_tape = batch_data['PROTEIN_EMBEDDING'].to(device)
            word_embedding = batch_data['COMPOUND_WORD_EMBEDDING'].to(device)
            labels = batch_data['labels'].to(device).float()  # 注意标签类型改为float

            outputs = model(drug_graphs, protein_graphs, bi_graphs, word_embedding, protein_tape).squeeze(-1)
            outputs = outputs * train_std + train_mean
            all_labels.extend(labels.detach().cpu().tolist())
            all_outputs.extend(outputs.detach().cpu().tolist())

        print(type(all_labels[0]), type(all_outputs[0]))
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        mse = mean_squared_error(all_labels, all_outputs)
        rmse = np.sqrt(mse)  # 计算均方根误差（RMSE）
        r2 = r2_score(all_labels, all_outputs)

        # np.save('pvalue/pred.npy', all_outputs)
        # np.save('pvalue/y.npy', all_labels)

        pearson_corr, _ = pearsonr(all_labels, all_outputs)
        spearman_corr, _ = spearmanr(all_labels, all_outputs)
        print(all_labels.shape)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('r2:', r2)
        print('pearson_corr:', pearson_corr)
        print('spearman_corr:', spearman_corr)

test(model)