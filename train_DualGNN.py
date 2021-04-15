import os
import pdb
import argparse
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DataLoad import MyDataset
from Model import DualGNN
from lightgcn import LightGCN
from LRGCCF import LRGCCF
from Model_MMGCN import MMGCN
from model_VBPR import VBPR_net
from Model_stargcn import Stargcn
from evaluation import test_eval
from evaluation import train_eval
from parse import parse_args
import networkx as nx
import sys
import random
from collections import Counter
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
def readTrainSparseMatrix(set_matrix,is_user,u_d,i_d):
    user_items_matrix_i=[]
    user_items_matrix_v=[] 
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
def readTrainSparseMatrix_all(u_i,i_u,u_d,i_d,user_num):
    all_matrix_i=[]
    all_matrix_v=[]

    d_i=u_d
    d_j=i_d
    for i in u_i:
        len_set = len(u_i[i])
        for j in u_i[i]:
            all_matrix_i.append([i, j+user_num])
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            # 1/sqrt((d_i+1)(d_j+1))
            all_matrix_v.append(d_i_j)  # (1./len_set)

    d_i=i_d
    d_j=u_d
    for i in i_u:
        len_set = len(i_u[i])
        for j in i_u[i]:
            all_matrix_i.append([i+user_num, j])
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            # 1/sqrt((d_i+1)(d_j+1))
            all_matrix_v.append(d_i_j)  # (1./len_set)

    all_matrix_i=torch.cuda.LongTensor(all_matrix_i)
    all_matrix_v=torch.cuda.FloatTensor(all_matrix_v)
    return torch.sparse.FloatTensor(all_matrix_i.t(), all_matrix_v)

def random_pick(some_list, probabilities):
    x = np.random.uniform(0,1)
    cumulative_probability = 0.0
    i = 0
    # pdb.set_trace()
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
        i += 1
    return i
def random_walk(start_node,user_co_dict,step,num_traces,back_pro):
    traces = []
    # pdb.set_trace()
    for i in range(num_traces):
        p_previous_node = -2
        previous_node = -1
        current_node = start_node
        # trace = []
        for j in range(step):
            if j >0:
                x = np.random.uniform(0, 1)
                if x < back_pro:
                    current_node = previous_node
            if len(user_co_dict[current_node])<4:
                pdb.set_trace()
            if p_previous_node == start_node:
                p_previous_node = previous_node
                previous_node = current_node
                current_node = start_node
            co_occur_index = user_co_dict[current_node][2]
            co_occur_prob = user_co_dict[current_node][3]
            # pdb.set_trace()
            pick_index = random_pick(co_occur_index,co_occur_prob)
            if len(user_co_dict[current_node][0][co_occur_index[pick_index]:co_occur_index[pick_index + 1]]) == 0:
                u_sample = user_co_dict[current_node][0][co_occur_index[pick_index]]
                traces.append(u_sample)

                p_previous_node = previous_node
                previous_node = current_node
                current_node = u_sample
                continue
            u_sample = np.random.choice(
                user_co_dict[current_node][0][co_occur_index[pick_index]:co_occur_index[pick_index + 1]])
            traces.append(u_sample)

            p_previous_node = previous_node
            previous_node = current_node
            current_node = u_sample
        # traces[i] = trace
        # pdb.set_trace()
    return traces
class Net:
    def __init__(self, args,sparse_graph):
        ##########################################################################################################################################
        self.device = torch.device("cuda:{0}".format(args.device) if torch.cuda.is_available() else "cpu")
        ##########################################################################################################################################
        seed = args.seed
        
        # seed = np.random.randint(0,200000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.varient = args.varient
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r#l_r#
        self.weight_decay = args.weight_decay#weight_decay#
        self.drop_rate = args.dropnode
        self.batch_size = args.batch_size
        self.construction = args.construction
        self.num_traces = args.traces
        self.back_pro = args.backprob
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.K = args.sampling
        self.dataset = args.dataset
        self.cold_start = args.cold_start
        self.dim_latent = args.dim_latent
        self.aggr_mode = args.aggr_mode#aggr_mode#
        self.user_aggr_mode = args.user_aggr_mode
        self.num_layer = args.num_layer
        self.has_id = args.has_id
        self.test_recall = []
        self.test_ndcg = []
        ##########################################################################################################################################
        if self.dataset =='Movielens':
            self.num_user = 55485
            self.num_item = 5986
        elif self.dataset =='tiktok':
            self.num_user = 36656
            self.num_item = 76085
        elif self.dataset =='tiktok_new':
            self.num_user = 32309
            self.num_item = 66456
        elif self.dataset =='cold_movie':
            self.num_user = 55485
            self.num_item = 5986
            self.num_cold_item = 867
        print('Data loading ...')
        self.train_dataset = MyDataset('../Data/'+self.dataset+'/', self.num_user, self.num_item,self.dataset,self.model_name)
        path = '../Data/'+self.dataset+'/'
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if args.dataset == 'Movielens':
            self.edge_index = np.load('../Data/Movielens/train.npy')
            self.user_graph_dict = np.load('../Data/Movielens/user_graph_dict.npy', allow_pickle=True).item()
            self.user_graph_dict_rw = np.load('../Data/Movielens/user_graph_dict_randomwalk.npy',allow_pickle=True).item()
            self.val_dataset = np.load('../Data/Movielens/val_full.npy', allow_pickle=True)
            self.test_dataset = np.load('../Data/Movielens/test_full.npy', allow_pickle=True)
            self.sparse_graph = sparse_graph
            self.v_feat = np.load('../Data/Movielens/FeatureVideo_normal.npy',allow_pickle=True)
            self.a_feat = np.load('../Data/Movielens/FeatureAudio_avg_normal.npy',allow_pickle=True)
            self.t_feat = np.load('../Data/Movielens/FeatureText_stl_normal.npy',allow_pickle=True)
            self.user_item_dict = np.load('../Data/Movielens/user_item_dict.npy', allow_pickle=True).item()
            # self.training_user_set = np.load('../Data/Movielens/user_item_dict_lr.npy', allow_pickle=True).item()
            # self.training_item_set = np.load('../Data/Movielens/item_user_dict_lr.npy', allow_pickle=True).item()
            # self.item_user_dict = np.load('./Data/Movielens/item_user_dict.npy', allow_pickle=True).item()
            # self.final_embed = torch.load('./Data/Movielens/movielens_final_emb.pt')
        elif args.dataset == 'tiktok':
            self.edge_index = np.load('../Data/tiktok/train.npy')
            self.user_graph_dict = np.load('../Data/tiktok/user_graph_dict_randomwalk.npy',allow_pickle=True).item()
            self.val_dataset = np.load('../Data/tiktok/val_full.npy',allow_pickle=True)
            self.test_dataset = np.load('../Data/tiktok/test_full.npy',allow_pickle=True)
            self.v_feat = torch.load('../Data/tiktok/feat_v.pt')
            self.a_feat = torch.load('../Data/tiktok/feat_a.pt')
            self.t_feat = torch.load('../Data/tiktok/feat_t.pt')
            self.user_item_dict = np.load('../Data/tiktok/user_item_dict.npy', allow_pickle=True).item()
            #self.final_embed = torch.load('./Data/tiktok/tiktok_final_emb.pt')
        elif args.dataset == 'tiktok_new':
            self.edge_index = np.load('../Data/tiktok_new/train_tik.npy')
            # pdb.set_trace()
            self.sparse_graph = sparse_graph
            self.user_graph_dict = np.load('../Data/tiktok_new/user_graph_dict.npy',allow_pickle=True).item()
            self.val_dataset = np.load('../Data/tiktok_new/val_full_tik.npy',allow_pickle=True)
            self.test_dataset = np.load('../Data/tiktok_new/test_full_tik.npy',allow_pickle=True)
            self.v_feat = torch.load('../Data/tiktok_new/feat_v_tik.pt')
            self.a_feat = torch.load('../Data/tiktok_new/feat_a_tik.pt')
            self.t_feat = torch.load('../Data/tiktok_new/feat_t_tik.pt')
            self.user_item_dict = np.load('../Data/tiktok_new/user_item_dict_tik.npy', allow_pickle=True).item()
            self.training_user_set = np.load('../Data/tiktok_new/user_item_dict_lr.npy', allow_pickle=True).item()
            self.training_item_set = np.load('../Data/tiktok_new/item_user_dict_lr.npy', allow_pickle=True).item()
        elif args.dataset =='cold_movie':
            self.edge_index = np.load(path+'final_cold_movie_train.npy')
            self.user_graph_dict = np.load(path+'user_graph_dict_cold_movie.npy',allow_pickle=True).item()
            self.val_dataset_warm_cold = np.load(path+'val_movie_warm+cold.npy',allow_pickle=True)
            self.test_dataset_warm_cold = np.load(path+'test_movie_warm+cold.npy',allow_pickle=True)
            self.val_dataset_warm = np.load(path+'val_movie_warm.npy',allow_pickle=True)
            self.test_dataset_warm = np.load(path+'test_movie_warm.npy',allow_pickle=True)
            self.val_dataset_cold = np.load(path+'val_movie_cold.npy',allow_pickle=True)
            self.test_dataset_cold = np.load(path+'test_movie_cold.npy',allow_pickle=True)
            self.v_feat = np.load(path+'v_feat_movie_cold.npy')
            self.a_feat = np.load(path+'a_feat_movie_cold.npy')
            self.t_feat = np.load(path+'t_feat_movie_cold.npy')
            self.user_item_dict = np.load(path+'user_item_dict_cold_mov.npy', allow_pickle=True).item()
            self.item_user_dict = np.load(path+'item_user_dict_cold_mov.npy', allow_pickle=True).item()
        print('Data has been loaded.')
        if self.model_name == 'LRGCCF':
            u_d=readD(self.training_user_set,self.num_user)
            i_d=readD(self.training_item_set,self.num_item)
            d_i_train=u_d
            d_j_train=i_d
            sparse_u_i=readTrainSparseMatrix(self.training_user_set,True,u_d,i_d)
            sparse_i_u=readTrainSparseMatrix(self.training_item_set,False,u_d,i_d)
            # sparse_all=readTrainSparseMatrix_all(self.training_user_set,self.training_item_set,u_d,i_d,self.num_user)
        
        # pdb.set_trace()
        # attn_u_mov = torch.load('./results/final_mov_30/u_prefer_weight.pt')
        # attn_u_mov_sm = torch.softmax(attn_u_mov,dim=1)
        # top_index_mov = torch.topk(torch.max(attn_u_mov_sm,dim=1).values,k=12).indices
        # top_index_mov_list = top_index_mov.tolist()
        # random.shuffle(top_index_mov_list)
        # attn_u_mov_final_10 = attn_u_mov_sm[top_index_mov_list]
        
        # attn_u_tik = torch.load('./results/final_tik_each/u_prefer_weight.pt')
        # attn_u_tik_sm = torch.softmax(attn_u_tik,dim=1)
        # # top_index_tik = torch.topk(torch.max(attn_u_tik_sm,dim=1).values,k=10).indices
        # top_index_tik = torch.topk(attn_u_tik_sm,dim=0,k=10).indices[:4]
        # top_index_tik_list = top_index_tik.view(1,-1).tolist()[0]
        # random.shuffle(top_index_tik_list)
        # attn_u_tik_final_10 = attn_u_tik_sm[top_index_tik_list]
        
        # # attn_u_mov_final_10 = torch.softmax(attn_u_mov_final_10,dim=1)
        # # attn_u_tik_final_10 = torch.softmax(attn_u_tik_final_10,dim=1)
        
        # draw_pic_list = [attn_u_mov_final_10,attn_u_tik_final_10]
        # self.draw_pic(draw_pic_list)
        
        # new_edge = []
        # for item in self.edge_index:
        #     user,item = item
        #     new_edge.append([int(user),int(item)])
        # pdb.set_trace()
        # matrix_list.clear()

        ##########################################################################################################################################
        # asdsa = []
        # for i in range(self.num_user):
        #     if len(self.user_graph_dict[i][1])!=0:
        #         asdsa.append(max(self.user_graph_dict[i][1]))
        # # pdb.set_trace()
        # user_topk = torch.topk(torch.tensor(asdsa),20).indices
        # sum=0
        # dim_list_max = []
        # dim_list = []
        # dim_same = []
        # dim_samea = []
        # for i in range(20):
        #     user_emb = self.final_embed[user_topk[i]]
        #     user_embs = self.final_embed[self.user_graph_dict[int(user_topk[0])][0]]
        #     dis = -torch.abs(user_embs-user_emb)
        #     for d in range(dis.shape[1]):
        #         top_list = torch.topk(dis.t()[d],25)
        #         top_values = top_list.values
        #         dim_list_max.append(torch.min(top_values))
        #         dim_list.append(top_list)
        #         dim_same.append(set(np.array(torch.tensor(top_list.indices.cpu()))))
        #     for k in range(dis.shape[1]):
        #         for j in range(dis.shape[1]):
        #             if (len(dim_same[k].intersection(dim_same[j]))>10) and (k!=j):
        #                 dim_samea.append((k,j))
        #     print('sadasdsad')
        #     pdb.set_trace()


# user_embs[list(dim_same[56].intersection(dim_same[37]))].cpu()
# self.draw_pic([user_embs[list(dim_same[56].intersection(dim_same[37]))].cpu().detach().numpy()])
# user_list = [user_emb.unsqueeze(0).cpu().detach().numpy(),user_embs[list(dim_same[56].intersection(dim_same[37]))].cpu().detach().numpy(),np.random.rand(11,64)]
        # for i in range(self.num_item):
        #     item_emb = self.final_embed[i+self.num_user]
        #     user_embs = self.final_embed[self.item_user_dict[i]]
        #     dis = -torch.abs(user_embs-item_emb)

        #     for i in range(dis.shape[1]):
        #         top_list = torch.topk(dis.t()[i],100)
        #         top_values = top_list.values
        #         dim_list_max.append(torch.min(top_values))
        #         dim_list.append(top_list)
        #         dim_same.append(set(np.array(torch.tensor(top_list.indices.cpu()))))
        #     for i in range(dis.shape[1]):
        #         for j in range(dis.shape[1]):
        #             if len(dim_same[i].intersection(dim_same[j]))>10:
        #                 dim_samea.append((i,j))
        #         # print('sadasdsad')
        #         pdb.set_trace()
        

        # self.item_user_dict = {}
        # for i in range(self.num_item):
        #     self.item_user_dict[i] = []
        # for data in self.edge_index:
        #     user,item = data
        #     self.item_user_dict[item-self.num_user].append(user)
        # for i in range(len(self.item_user_dict)):
        #     sum = sum+ len(self.item_user_dict[i])

        # pdb.set_trace()
        self.features = [self.v_feat, self.a_feat, self.t_feat]
        if args.model_name =='DualGNN':
            self.model = DualGNN(self.features, self.edge_index,self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.construction, self.num_layer, self.has_id, self.dim_latent,self.weight_decay,self.drop_rate,self.K, self.user_item_dict,self.dataset,self.cold_start,device = self.device).to(self.device)
        if self.model_name == 'LightGCN':
            self.model = LightGCN(self.features, self.sparse_graph,self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat, self.num_layer, self.has_id, self.dim_latent,self.weight_decay,self.dropout , self.user_item_dict,device=self.device).to(self.device)
        if self.model_name == 'MMGCN':
            self.model = MMGCN(self.v_feat, self.a_feat, self.t_feat, self.t_feat, self.edge_index, self.batch_size, self.num_user, self.num_item, 'mean', 'False', 2, True, self.user_item_dict, self.weight_decay, self.dim_latent,self.dataset,self.drop_rate,device=self.device).to(self.device)
        if self.model_name == 'VBPR':
            self.model = VBPR_net(self.num_user, self.num_item,self.weight_decay, self.dim_latent, self.features,self.user_item_dict,self.dataset,device=self.device).to(self.device)
        if self.model_name == 'Stargcn':
            self.model = Stargcn(self.features,self.sparse_graph, self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat,
                 self.num_layer, self.weight_decay, self.user_item_dict,self.dim_latent,device=self.device).to(self.device)
        if self.model_name == 'LRGCCF':
            self.model = LRGCCF(self.num_user, self.num_item, self.dim_latent,sparse_u_i,sparse_i_u,d_i_train,d_j_train,self.user_item_dict,self.weight_decay,device=self.device).to(self.device)
        # if self.model_name == 'LRGCCF':
            
        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}])
        ##########################################################################################################################################
    
    def draw_pic(self,user_list):
        fig, axs = plt.subplots(1,2)
        img_list = []
        pdb.set_trace()
        # pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
        #                     cmap=cm[col])
        # fig.colorbar(pcm, ax=ax)
        for i in range(len(user_list)):
            im = self.heatmap(user_list[i].cpu().detach().numpy(), ax=axs[i],
                                cmap="YlGn", cbarlabel="harvest [t/year]")
            img_list.append(im)
        
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
        vmin = min(image.min() for image in user_list)
        vmax = max(image.max() for image in user_list)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in img_list:
            im.set_norm(norm)
        fig.colorbar(img_list[0], ax=axs, orientation='horizontal', fraction=.1, pad=0.05, aspect=30)
        # fig.tight_layout()
        plt.savefig('./pic3.jpg')
        fig.savefig('test.eps',dpi=600)
        plt.show()
    def heatmap(self,data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        # ax.set_xticklabels(col_labels)
        # ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        # ax.tick_params(top=True, bottom=False,
        #                labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
        #          rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1))
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])
        return im
    def run(self,):
        # self.model.result_embed = torch.load('./results/top40_best_movie/result_emb.pt')
        # pdb.set_trace()
        max_precision = 0.0
        max_recall = 0.0
        max_NDCG = 0.0
        num_decreases = 0
        max_val=0
        while os.path.exists('./results/'+self.varient) == False :
            os.mkdir('./results/'+self.varient)
        result_log_path = './results/'+self.varient+f'/train_log_({args.l_r:.6f}_{args.weight_decay:.6f}_{args.sampling:.6f}_{args.dataset:.9s}_{self.num_traces}_{self.back_pro}_{self.seed}).txt'
        result_path = './results/'+self.varient+'/result_{0}_{1}_{2}.txt'.format(args.dataset,self.num_traces,self.back_pro)
        result_best_path = './results/'+self.varient+'/result_best_{0}_{1}_{2}.txt'.format(args.dataset,self.num_traces,self.back_pro)
        user_cons = 0
        print(args.l_r)
        print(args.weight_decay)
        print(args.dataset)
        
        with open(result_log_path, "a") as f:
            f.write(result_log_path)
            f.write("\n")
        for epoch in range(self.num_epoch):
            # ranklist_tra, ranklist_vt, ranklist_tt  = self.model.gene_ranklist(self.val_dataset,self.test_dataset)
            # train_eval(epoch, self.model, 'Train',ranklist_tra,args,result_log_path)
            # test_eval(epoch, self.model, self.val_dataset, 'Val',ranklist_vt,args,result_log_path)
            # test_precision_10, test_recall_10, test_ndcg_score_10 = test_eval(epoch, self.model, self.test_dataset, 'Test',ranklist_tt,args,result_log_path)
            self.model.train()
            user_graph, user_weight_matrix = self.topk_sample(self.K)
            # user_cons= self.constra_sample(self.K,user_graph)
            # user_graph,user_weight_matrix = self.user_randomwalk(self.K,self.num_traces,self.back_pro)
            # pdb.set_trace()
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))
            sum_loss = 0.0
            sum_reg_loss = 0.0

            
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                if self.model_name == 'DualGNN':
                    self.loss,reg_loss = self.model.loss(data ,user_graph,user_weight_matrix,user_cons=user_cons)
                else:
                    self.loss,reg_loss = self.model.loss(data)
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss
                sum_reg_loss += reg_loss
            print('avg_loss:',sum_loss/self.batch_size)
            print('avg_reg_loss:', sum_reg_loss / self.batch_size)
            # cov_p_abs = torch.max(self.model.cov_u_p,dim=1).values-torch.min(self.model.cov_u_p,dim=1).values
            # print('cov_max:',torch.max(cov_p_abs))
            # print('cov_min:',torch.min(cov_p_abs))
            # print('1/3 min',len(torch.where(cov_p_abs<(torch.max(cov_p_abs)-torch.min(cov_p_abs))/3)[0]))
            # print('1/3 max',len(torch.where(cov_p_abs>(torch.max(cov_p_abs)-torch.min(cov_p_abs))/3*2)[0]))
            pbar.close()
            if torch.isnan(sum_loss/self.batch_size):
                with open(result_path,'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} \t sampling:{2} \t SEED:{3} is Nan'.format(args.l_r, args.weight_decay,args.sampling,self.seed))
                break
            # self.model.result_embed = torch.load('./results/top40_best_movie/result_emb.pt')
            # ranklist_tra, ranklist_vt, ranklist_tt  = self.model.gene_ranklist(self.val_dataset,self.test_dataset)
            ranklist_tra, ranklist_vt, ranklist_tt  = self.model.gene_ranklist(self.val_dataset,self.test_dataset)
            # ranklist_tra, ranklist_vt, ranklist_tt  = self.model.gene_ranklist(self.val_dataset_warm,self.test_dataset_warm)
            # ranklist_tra, ranklist_vt, ranklist_tt  = self.model.gene_ranklist_cold(self.val_dataset_cold,self.test_dataset_cold)
            # pdb.set_trace()
            train_eval(epoch, self.model, 'Train',ranklist_tra,args,result_log_path)
            val_precision_10, val_recall_10, val_ndcg_score_10 =test_eval(epoch, self.model, self.val_dataset, 'Val',ranklist_vt,args,result_log_path,0)
            test_precision_10, test_recall_10, test_ndcg_score_10 = test_eval(epoch, self.model, self.test_dataset, 'Test',ranklist_tt,args,result_log_path,0)
            
            if self.model_name == 'DualGNN':
                if self.construction == 'weighted_sum':
                    attn_u = F.softmax(self.model.weight_u,dim=1)
                    attn_u = torch.squeeze(attn_u)


                    attn_u_max = torch.max(attn_u,0)
                    attn_u_max_num = torch.max(attn_u,0).indices[0]
                    attn_u_min = torch.min(attn_u,0)
                    with open(result_log_path, "a") as f:
                        f.write('---------------------------------attn_u_max: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
                                epoch, 10, float(attn_u_max[0][0]),float(attn_u_max[0][1]),float(attn_u_max[0][2])))  # 将字符串写入文件中
                        f.write("\n")
                    with open(result_log_path, "a") as f:
                        f.write('---------------------------------attn_u_num: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
                                epoch, 10, float(attn_u[attn_u_max_num][0]),float(attn_u[attn_u_max_num][1]),float(attn_u[attn_u_max_num][2])))  # 将字符串写入文件中
                        f.write("\n")
                    with open(result_log_path, "a") as f:
                        f.write('---------------------------------attn_u_min: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
                                epoch, 10, float(attn_u_min[0][0]),float(attn_u_min[0][1]),float(attn_u_min[0][2])))  # 将字符串写入文件中
                        f.write("\n")
            # with open(f'./train_log_({args.l_r:.6f}_{args.weight_decay:.6f}_{args.dropout:.6f}).txt', "a") as f:
            #      f.write('---------------------------------attn_i0: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
            #                epoch, 10, float(self.model.weight_i[0][0]),float(self.model.weight_i[0][1]),float(self.model.weight_i[0][2])))  # 将字符串写入文件中
            #      f.write("\n")
            self.test_recall.append(test_recall_10)
            self.test_ndcg.append(test_ndcg_score_10)
           # pdb.set_trace()
            if val_recall_10 > max_val:
               max_precision = test_precision_10
               max_recall = test_recall_10
               max_NDCG = test_ndcg_score_10
               max_val = val_recall_10
               num_decreases = 0
               best_embed = self.model.result_embed
            #    if self.construction == 'weighted_sum':
            #         best_attn = attn_u
            else:   
                if num_decreases >20 and self.model_name != 'Stargcn':
                    with open(result_path, 'a') as save_file:
                        save_file.write(
                            'lr: {0} \t Weight_decay:{1} \t sampling:{2}=====> Precision:{3} \t Recall:{4} \t NDCG:{5} \t SEED:{6}\r\n'.
                            format(args.l_r, args.weight_decay,args.sampling ,max_precision, max_recall, max_NDCG,self.seed))
                    # torch.save(best_attn,'./results/'+self.varient+'/u_prefer_weight.pt')
                    torch.save(best_embed,'./results/'+self.varient+'/result_emb.pt')
                    while os.path.exists(result_best_path) == False :
                        with open(result_best_path, 'a') as save_file:
                            save_file.write(
                                'Recall:{0}\r\n'.
                                format(max_recall)) 
                    # pdb.set_trace()
                    file = open(result_best_path)
                    maxs = file.readline()
                    maxvalue = float(maxs.strip('[Recall:\n]'))
                    # if max_recall >= maxvalue:
                    #     np.save('./results/'+self.varient+'/recall.npy',self.test_recall)
                    #     np.save('./results/'+self.varient+'/ndcg.npy',self.test_ndcg)
                    #     torch.save(self.model.result_embed,'./results/'+self.varient+'/result_emb.pt')
                    break
                else:
                    num_decreases += 1
            if epoch>990:
                    with open(result_path, 'a') as save_file:
                        save_file.write(
                            'lr: {0} \t Weight_decay:{1} \t sampling:{2}=====> Precision:{3} \t Recall:{4} \t NDCG:{5} \t SEED:{6}\r\n'.
                            format(args.l_r, args.weight_decay,args.sampling ,max_precision, max_recall, max_NDCG,self.seed))
                    file = open(result_best_path)
                    maxvalue = float(maxs.strip('[Recall:\n]'))
                    if max_recall >= maxvalue:
                        np.save('./results/'+self.varient+'/recall.npy',self.test_recall)
                        np.save('./results/'+self.varient+'/ndcg.npy',self.test_ndcg)
                        torch.save(self.model.result_embed,'./results/'+self.varient+'/result_emb.pt')
                    break
        return max_recall, max_precision, max_NDCG
    def constra_sample(self,k,user_graph):
        constra_users_sample = []
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(self.num_user):
            constra_user = []
            if len(self.user_graph_dict[i][0]) == 0:
                 constra_users_sample.append(tasike)
                 continue
            while len(constra_user)<k:
                neg_user = np.random.randint(0,self.num_user)
                if neg_user in user_graph[i]:
                    continue
                else:
                    constra_user.append(int(neg_user))
            constra_users_sample.append(constra_user)
        # pdb.set_trace()
        return constra_users_sample
    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    # pdb.set_trace()
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)
                
                # user_weight_matrix[i] = torch.tensor(user_graph_weight) / sum(user_graph_weight) #weighted
                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0) #softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k)/k #mean
                # pdb.set_trace()
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            # user_weight_matrix[i] = torch.tensor(user_graph_weight) / sum(user_graph_weight) #weighted
            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0) #softmax
            if self.user_aggr_mode == 'mean':
                # pdb.set_trace()
                user_weight_matrix[i] = torch.ones(k)/k #mean
            # user_weight_list.append(user_weight)
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix
    def user_randomwalk(self,k,num_traces,back_pro):
        tasike = []
        user_weight_matrix = torch.zeros(len(self.user_graph_dict_rw), k)
        for i in range(k):
            tasike.append(0)
        odd = []
        user_graph_index = []
        count = 0
        # for i in range(len(self.user_graph_dict)):
        #     if len(self.user_graph_dict[i][0]) !=0:
        #         # traces = random_walk(i,self.user_graph_dict,20,2)
        #         co_occur_dict = dict(Counter(self.user_graph_dict[i][1]))
        #         co_occur_list = list(co_occur_dict.keys())
        #         co_occur_sum = sum(co_occur_list)
        #         pick_pro = []
        #         co_occur_index = []
        #         index_sum = 0
        #         # pdb.set_trace()
        #         for item in co_occur_list:
        #             pick_pro.append(item / co_occur_sum)
        #             co_occur_index.append(index_sum)
        #             index_sum += co_occur_dict[item]
        #         co_occur_index.append(-1)
        #         # pick_index = random_pick(co_occur_list, pick_pro)
        #         self.user_graph_dict[i].append(co_occur_index) #pro_index
        #         self.user_graph_dict[i].append(pick_pro)
        #     else:
        #         continue
        pdb.set_trace()
        for i in range(len(self.user_graph_dict_rw)):
            user_graph_sample = []
            user_graph_weight = []
            
            # pdb.set_trace()
            if len(self.user_graph_dict_rw[i][0]) !=0:
                traces = random_walk(i,self.user_graph_dict_rw,k,num_traces,back_pro)
                # pdb.set_trace()
                traces_count = dict(Counter(traces))
                traces_co = sorted(traces_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
                for pair in traces_co[:k]:
                    # pdb.set_trace()
                    user_graph_sample.append(pair[0])
                    user_graph_weight.append(pair[1])
                # pdb.set_trace()
                # temp_lenth = len(user_graph_sample)
                # if temp_lenth<k:
                #     # pdb.set_trace()
                #     temp_tensor = F.softmax(torch.tensor(user_graph_weight,dtype=torch.float32), dim=0)
                #     user_weight_matrix[i][:temp_lenth] = temp_tensor #softmax
                #     user_graph_sample = user_graph_sample+tasike[temp_lenth:]
                while len(user_graph_sample) < k:
                    # pdb.set_trace()
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                else:
                    # pdb.set_trace()
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight,dtype=torch.float32), dim=0)
            else:
                # odd.append(i)
                user_graph_sample = tasike
                # continue
            user_graph_index.append(user_graph_sample)
            # if len(self.user_graph_dict[i][0]) < 20:
            #     user_graph_sample[i] = self.user_graph_dict[i][0]
            #     while len(user_graph_sample[i]) < 20:
            #         # user_graph_dict[i][0].append(user_graph_dict[i][0][0])
            #         if len(user_graph_sample[i]) != 0:
            #             user_graph_sample[i].append(random.choice(user_graph_sample[i]))
            #         else:
            #             odd.append(i)
            #             user_graph_sample[i] = tasike
            # else:
            #
            #     while len(user_graph_sample[i]) < 20:
            #         pick_index = random_pick(co_occur_list, pick_pro)
            #         if len(self.user_graph_dict[i][0][co_occur_index[pick_index]:co_occur_index[pick_index + 1]]) == 0:
            #             u_sample = self.user_graph_dict[i][0][co_occur_index[pick_index]]
            #             # if u_sample in user_graph_sample[i]:
            #             # pdb.set_trace()
            #             # continue
            #             user_graph_sample[i].append(u_sample);
            #             continue
            #         u_sample = random.choice(
            #             self.user_graph_dict[i][0][co_occur_index[pick_index]:co_occur_index[pick_index + 1]])
            #         user_graph_sample[i].append(u_sample)

                # user_graph_sample[i] = random.sample(self.user_graph_dict[i][0], 10)

            # user_graph_index.append(user_graph_sample[i])

        # nodelist = []
        # linklist = []
        #
        # for i in range(self.num_user):
        #     nodelist.append(i)
        # for i in range(self.num_user):
        #     for j in user_graph_index[i]:
        #         linklist.append([i,j])
        # for node in nodelist:
        #     G.add_node(node)
        #
        # for link in linklist:
        #     G.add_edge(link[0], link[1])

        # G = nx.Graph()
        # pdb.set_trace()
        # for node in nodelist:
        #     G.add_node(node)
        # for link in linklist:
        #     G.add_edge(link[0], link[1])
        # sum_sub =0
        # for sub_graph in nx.connected_components(G):
        #     sum_sub+=1
        # pdb.set_trace()
        # user_graph_sample[len(self.user_graph_dict) + 1] = odd
        # user_graph_index.append(odd)
        # pdb.set_trace()
        return user_graph_index,user_weight_matrix
if __name__ == '__main__':
    dimN = 100000
    dimM = 3300
    matrix_list = []
    args = parse_args()
    if args.dataset == 'tiktok_new':
        # sparse_graph = torch.load('../Data/tiktok_new/graph.pt')
        sparse_graph=None
    elif args.dataset == 'Movielens':
        # sparse_graph = torch.load('../Data/Movielens/graph.pt')
        sparse_graph=None
    device = torch.device("cuda:{0}".format(args.device) if torch.cuda.is_available() else "cpu")
    # sys.stdout = Logger(f'./train_log_({args.l_r:.6f}_{args.weight_decay:.6f}).txt')
    while(len(matrix_list) <args.blocks):
        try:
            matrix_list.append(torch.ones([dimN,dimM],device=device))
        except:
            continue
    # pdb.set_trace()
    matrix_list.clear()
    egcn = Net(args,sparse_graph)
    egcn.run()
    # except Exception as err:
    #     print(1,err)
    # else:
    #     print(2)


