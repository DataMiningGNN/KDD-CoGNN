import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from util import cmd_args as args, load_data, create_meta_episodes, euclidean_metric, cos_metric
import scipy as sp
import scipy.stats
import copy

from dagcn import DAGCN
from graphcnn import GraphCNN, Graph_reps_GCN
from network_layers import MLP, MLPClassifier, MLPTransformation, GraphConv, FTL_Classification

from classification_heads import ClassificationHead
from triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss
from Aggregate import aggregate_att, average_weights_dp, average_weights

import warnings
warnings.filterwarnings("ignore")


criterion = nn.CrossEntropyLoss()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def adjust_learning_rate(optimizer, epoch):
    if epoch < args.adjust_epochs:
        lr = args.learning_rate
    else:
        # lr = args.learning_rate * (0.2 ** (epoch // 50))
        lr = args.learning_rate  * args.adjust_lr_frac

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lamda(args, epoch):
    epoch = epoch + 1
    if epoch <= args.warm_up:
        lambda_cot = args.lambda_cot_max*math.exp(-5*(1-epoch/args.warm_up)**2)
    else:
        lambda_cot = args.lambda_cot_max
        
    return lambda_cot


def Jensen_Shannon_div(num_U_samples, U_p1, U_p2):
# the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a1 = 0.5 * (S(U_p1) + S(U_p2)) + 1e-8
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)
    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)
    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3))/num_U_samples


class Graph_Extractor(nn.Module):
    def __init__(self, device):
        super(Graph_Extractor, self).__init__()
        if args.gm == 'DAGCN':
            model = DAGCN
        else:
            print('unknown gm %s' % args.gm)
            sys.exit()

        args.attr_dim = 0

        self.gnn = model(latent_dim=args.DAGCN_latent_dim,
                     output_dim=args.out_dim,
                     num_node_feats=args.feat_dim,
                     num_edge_feats=0,
                     multi_h_emb_weight=args.multi_h_emb_weight,
                     max_k=args.max_k,
                     dropout=args.dropout,
                     max_block=args.max_block,
                     reg=args.reg)


        out_dim = args.out_dim
        if out_dim == 0:
            out_dim = self.gnn.dense_dim

        self.mlps = MLP(2, out_dim, args.hidden_dim*2, args.hidden_dim)
        self.device = device


    def forward(self, batch_graph, edge_feat = None):
        node_feat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        embed = self.mlps(embed)

        # relation_mat = cos_metric(embed, embed)
        relation_mat = euclidean_metric(embed, embed)
        relation_mat = F.sigmoid(relation_mat)

        return relation_mat, embed


def few_shot_classification(args, cls_head, support_reps, query_reps, s_labels, num_way, num_shot, device):
    if args.class_head == 'DiffCO':
        support_reps = support_reps.unsqueeze(0)
        query_reps   = query_reps.unsqueeze(0)
        s_labels     = s_labels.unsqueeze(0)

        query_output = cls_head(query_reps, support_reps, s_labels, num_way, num_shot, device, lambda_reg=args.Lambda_reg)
        query_output = query_output.squeeze(0)
        
    elif args.class_head == 'Proto':
        ##### Prototype network
        support_reps = support_reps.reshape(num_shot, num_way, -1).mean(dim=0)
        query_output = euclidean_metric(query_reps, support_reps)
        
    return query_output
            
    
def few_shot_test(args, Dagcn, Gin, Graph_reps_gcn, cls_head, episodes_idx, graphs_list, test_way, device, phase='target-test'):    
    ACC = []
    num_samples_episode = float(len(episodes_idx[0]))

    num_shot = args.test_shot
    num_way  = test_way

    num_support_samples = num_shot * num_way
    num_query_samples = float(num_samples_episode - num_support_samples)
    
    # selected_idx = []
    with torch.no_grad():
        for episode_idx in episodes_idx:
            episode = [graphs_list[idx] for idx in episode_idx]
            # episode_label = [g.label for g in episode]
            # print('episode_label:', episode_label)

            # for idx in episode_idx:
            #     if idx not in selected_idx:
            #         selected_idx.append(idx)
            
            Gin_relation_mat, Gin_embed = Gin(episode)
            Dagcn_relation_mat, Dagcn_embed = Dagcn(episode)          
                
            graph_representations_1 = Graph_reps_gcn(Gin_embed, Gin_relation_mat)
            graph_representations_2 = Graph_reps_gcn(Dagcn_embed, Dagcn_relation_mat)
                        
            s_labels = torch.arange(num_way).repeat(num_shot)
            s_labels = s_labels.type(torch.LongTensor).to(device)
            
            support_reps_1 = graph_representations_1[:num_support_samples]
            query_reps_1   = graph_representations_1[num_support_samples:]
            support_reps_2 = graph_representations_2[:num_support_samples]
            query_reps_2   = graph_representations_2[num_support_samples:]

            
            query_output_1 = few_shot_classification(args, cls_head, support_reps_1, query_reps_1, \
                                                     s_labels, num_way, num_shot, device)
            query_output_2 = few_shot_classification(args, cls_head, support_reps_2, query_reps_2, \
                                                     s_labels, num_way, num_shot, device)
                
            query_output = query_output_1 + query_output_2
                
            # query_labels = torch.LongTensor([graphs_list[idx].label for idx in episode_idx[num_support_samples:]]).to(device)
            query_labels = torch.arange(num_way).repeat(args.num_query)
            ## query_labels = query_labels.type(torch.cuda.LongTensor)
            query_labels = query_labels.type(torch.LongTensor).to(device)
            
            query_pred = query_output.max(1, keepdim=True)[1]
            correct = query_pred.eq(query_labels.view_as(query_pred)).sum().cpu().item()
            acc = correct / num_query_samples  ###########

            ACC.append(acc)

    # print('num of selected_idx:', len(selected_idx))
    
    m, h = mean_confidence_interval(ACC)
    print(phase, "accuracy: %f, h: %f" % (m, h))
    return m, h    


def meta_train(args, train_episodes_idx, test_episodes_idx, graphs_list, Dagcn, Gin, Graph_reps_gcn, cls_head, Dagcn_optimizer, Gin_optimizer, gcn_optimizer, \
               count_update_epoch, train_way, test_way, device, target_episodes_idx=None, target_graphs_list=None):

    loss_accum = 0.0

    num_samples = len(train_episodes_idx[0])
    num_support_samples = train_way * args.train_shot
    num_query_samples = float(num_samples) - num_support_samples
 
    ACC = []

    for train_episode_idx in train_episodes_idx:
        Dagcn_optimizer.zero_grad()
        Gin_optimizer.zero_grad()
        gcn_optimizer.zero_grad()

        train_episode = [graphs_list[idx] for idx in train_episode_idx]
        
        # train_episode_label = [g.label for g in train_episode]
        # print('train_episode_label:', train_episode_label)
        
        Gin_relation_mat, Gin_embed = Gin(train_episode)  #######
        Dagcn_relation_mat, Dagcn_embed = Dagcn(train_episode) ########
            
        graph_representations_1 = Graph_reps_gcn(Gin_embed, Gin_relation_mat)
        graph_representations_2 = Graph_reps_gcn(Dagcn_embed, Dagcn_relation_mat)

        # labels = torch.LongTensor([graph.label for graph in train_episode]).to(device)
        # query_labels = labels[train_episode_mask_inverse]
    
        new_labels = torch.arange(train_way).repeat(args.num_query + args.train_shot)
        # new_labels = new_labels.type(torch.cuda.LongTensor)
        new_labels = new_labels.type(torch.LongTensor).to(device)
        query_labels = new_labels[num_support_samples:]
        
        s_labels = new_labels[:num_support_samples]
        
        support_reps_1 = graph_representations_1[:num_support_samples]
        query_reps_1   = graph_representations_1[num_support_samples:]
        support_reps_2 = graph_representations_2[:num_support_samples]
        query_reps_2   = graph_representations_2[num_support_samples:]


        if args.com_support_triplet_l:
            # s_triplet_l_1    = batch_hard_triplet_loss(s_labels, support_reps_1, device, margin=args.triplet_m)
            s_triplet_l_1, _ = batch_all_triplet_loss(s_labels, support_reps_1, device, margin=args.triplet_m)
            # s_triplet_l_2    = batch_hard_triplet_loss(s_labels, support_reps_2, device, margin=args.triplet_m)
            s_triplet_l_2, _ = batch_all_triplet_loss(s_labels, support_reps_2, device, margin=args.triplet_m)

        query_output_1 = few_shot_classification(args, cls_head, support_reps_1, query_reps_1, \
                                                 s_labels, train_way, args.train_shot, device)
        query_output_2 = few_shot_classification(args, cls_head, support_reps_2, query_reps_2, \
                                                 s_labels, train_way, args.train_shot, device)
        

        query_output = query_output_1 + query_output_2
        loss = criterion(query_output, query_labels)

        # print('CE LOSS:', loss)
        # print('query_output:', query_output)
            
        query_pred = query_output.max(1, keepdim=True)[1]
        correct = query_pred.eq(query_labels.view_as(query_pred)).sum().cpu().item()
        acc = correct / float(num_query_samples)
        ACC.append(acc)
        
        # triplet_l_1 = batch_hard_triplet_loss(new_labels, Dagcn_embed, device, margin=args.triplet_m)
        triplet_l_1, _ = batch_all_triplet_loss(new_labels, Dagcn_embed, device, margin=args.triplet_m)
        # triplet_l_2 = batch_hard_triplet_loss(new_labels, Gin_embed, device, margin=args.triplet_m)
        triplet_l_2, _ = batch_all_triplet_loss(new_labels, Gin_embed, device, margin=args.triplet_m)
        
        if args.com_support_triplet_l:
            loss = args.CE_lambda * loss + (triplet_l_1 + triplet_l_2 + s_triplet_l_1 + s_triplet_l_2)
        else:
            loss = args.CE_lambda * loss + (triplet_l_1 + triplet_l_2)

        if args.JS_div:
            JS_loss = Jensen_Shannon_div(num_query_samples, query_output_1, query_output_2)
            loss = loss + JS_lamda * JS_loss
            # print('JS_loss:', JS_loss)
            
        loss.backward()


        # nn.utils.clip_grad_norm(Dagcn.parameters(), args.clip)
        nn.utils.clip_grad_norm(Gin.parameters(), args.clip)
        nn.utils.clip_grad_norm(Graph_reps_gcn.parameters(), args.clip)

        Dagcn_optimizer.step()
        Gin_optimizer.step()
        gcn_optimizer.step()
    
        loss = loss.detach().cpu().numpy()
        loss_accum += loss


    average_loss = loss_accum/len(train_episodes_idx)

    print('count_update_epoch:', count_update_epoch, "train loss: ", average_loss)
    m, h = mean_confidence_interval(ACC)
    print('train', "accuracy: %f, h: %f" % (m, h))

    return Dagcn.state_dict(), Gin.state_dict(), Graph_reps_gcn.state_dict()


def update_local_weights(args, Dagcn, Gin, Graph_reps_gcn, cls_head, user_data, global_round, train_way, test_way, device):
    Dagcn_optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, Dagcn.parameters()), \
                                       lr=args.learning_rate, weight_decay=args.weight_decay_value)
    Gin_optimizer   = torch.optim.Adam(filter(lambda p : p.requires_grad, Gin.parameters()), \
                                       lr=args.learning_rate, weight_decay=args.weight_decay_value)
    gcn_optimizer   = torch.optim.Adam(filter(lambda p : p.requires_grad, Graph_reps_gcn.parameters()), \
                                       lr=args.learning_rate, weight_decay=args.weight_decay_value)

    train_episodes_idx, test_episodes_idx, graphs_list = user_data[0], user_data[1], user_data[2]
                                                
    for epoch in range(args.num_epochs):
        count_update_epoch = global_round * args.num_epochs+ epoch
        if args.adjust_lr:
            adjust_learning_rate(Dagcn_optimizer, count_update_epoch)
            adjust_learning_rate(Gin_optimizer,   count_update_epoch)
            adjust_learning_rate(gcn_optimizer,   count_update_epoch)
            
        Dagcn_w, Gin_w, Graph_reps_gcn_w = \
            meta_train(args, train_episodes_idx, test_episodes_idx, graphs_list, Dagcn, Gin, \
                       Graph_reps_gcn, cls_head, Dagcn_optimizer, Gin_optimizer, gcn_optimizer, \
                       count_update_epoch, train_way, test_way, device)

    return [Dagcn_w, Gin_w, Graph_reps_gcn_w]


def test(args, episodes_idx, graphs_list, Dagcn, Gin, Graph_reps_gcn, FTL_Classifier, ft_epoch, device):

    num_test_samples = 0.0
    all_test_idx = []
    Correct = 0.0
    
    with torch.no_grad():
        Dagcn.eval()
        Gin.eval()
        Graph_reps_gcn.eval()
        FTL_Classifier.eval()

        for episode_idx in episodes_idx:
            all_test_idx += [i for i in episode_idx]
            episode = [graphs_list[idx] for idx in episode_idx]
            
            Gin_relation_mat, Gin_embed = Gin(episode)
            Dagcn_relation_mat, Dagcn_embed = Dagcn(episode)         
                
            graph_representations_1 = Graph_reps_gcn(Gin_embed, Gin_relation_mat) 
            graph_representations_2 = Graph_reps_gcn(Dagcn_embed, Dagcn_relation_mat)                                 
         
                
            Z_1 = FTL_Classifier(graph_representations_1)
            Z_2 = FTL_Classifier(graph_representations_2)

            Z = Z_1 + Z_2

            query_labels = torch.LongTensor([graphs_list[idx].label for idx in episode_idx]).to(device)

            pred = Z.max(1, keepdim=True)[1]
            correct = pred.eq(query_labels.view_as(pred)).sum().cpu().item()
            Correct += correct
        
            num_test_samples += len(episode)

        if ft_epoch == 0:
            if len(set(all_test_idx)) != len(all_test_idx):
                print('bug in test data')
                sys.exit()
            
    test_acc = Correct/float(num_test_samples)
    print('Correct/num_test_samples:{}/{}, test_acc:{}'.format(Correct, num_test_samples, test_acc))
    return test_acc


def Fine_tuning(args, train_episodes_idx, graphs_list, Dagcn, Gin, Graph_reps_gcn, FTL_Classifier, \
                Dagcn_optimizer, Gin_optimizer, gcn_optimizer, FTL_optimizer, ft_epoch, ft_criterion, device, test_episodes_idx=None):

    if args.JS_div:
        JS_lamda = adjust_lamda(args, ft_epoch)

    ft_loss_accum = 0.0
    Correct = 0.0
    count_samples = 0.0
    
    if not args.Fixed_weights:
        for param_group in Dagcn_optimizer.param_groups:
            print('Dagcn_optimizer lr:', param_group['lr'])
        for param_group in Gin_optimizer.param_groups:
            print('Gin_optimizer lr:', param_group['lr'])
        for param_group in gcn_optimizer.param_groups:
            print('gcn_optimizer lr:', param_group['lr'])
    for param_group in FTL_optimizer.param_groups:
        print('FTL_optimizer lr:', param_group['lr'])
    
    for train_episode_idx in train_episodes_idx:
        count_samples += float(len(train_episode_idx))

        if Dagcn_optimizer is not None:
            Dagcn_optimizer.zero_grad()
            Gin_optimizer.zero_grad()
            gcn_optimizer.zero_grad()
        FTL_optimizer.zero_grad()

        train_episode = [graphs_list[idx] for idx in train_episode_idx]
        
        Gin_relation_mat, Gin_embed = Gin(train_episode)  #######
        Dagcn_relation_mat, Dagcn_embed = Dagcn(train_episode) ########
            
        graph_representations_1 = Graph_reps_gcn(Gin_embed, Gin_relation_mat)
        graph_representations_2 = Graph_reps_gcn(Dagcn_embed, Dagcn_relation_mat)

        Z_1 = FTL_Classifier(graph_representations_1)
        Z_2 = FTL_Classifier(graph_representations_2)

        query_labels = torch.LongTensor([graphs_list[idx].label for idx in train_episode_idx]).to(device)
    
        Z = Z_1 + Z_2
        loss = ft_criterion(Z, query_labels)

        pred = Z.max(1, keepdim=True)[1]
        correct = pred.eq(query_labels.view_as(pred)).sum().cpu().item()
        Correct += correct
        
        if not args.Fixed_weights:
            # triplet_l_1 = batch_hard_triplet_loss(query_labels, Dagcn_embed, device, margin=args.triplet_m)
            triplet_l_1, _ = batch_all_triplet_loss(query_labels, Dagcn_embed, device, margin=args.triplet_m)
            # triplet_l_2 = batch_hard_triplet_loss(query_labels, Gin_embed, device, margin=args.triplet_m)
            triplet_l_2, _ = batch_all_triplet_loss(query_labels, Gin_embed, device, margin=args.triplet_m)
                
            loss = args.CE_lambda * loss + (triplet_l_1 + triplet_l_2)

        if args.JS_div:
            JS_loss = Jensen_Shannon_div(float(len(train_episode_idx)), Z_1, Z_2)
            loss = loss + JS_lamda * JS_loss
            
        loss.backward()


        if Dagcn_optimizer is not None:
            # nn.utils.clip_grad_norm(Dagcn.parameters(), args.clip)
            nn.utils.clip_grad_norm(Gin.parameters(), args.clip)
            nn.utils.clip_grad_norm(Graph_reps_gcn.parameters(), args.clip)  
        nn.utils.clip_grad_norm(FTL_Classifier.parameters(), args.clip)
    
        if Dagcn_optimizer is not None:
            Dagcn_optimizer.step()
            Gin_optimizer.step()
            gcn_optimizer.step()
        FTL_optimizer.step()
    
        loss = loss.detach().cpu().numpy()
        ft_loss_accum += loss
        
    
    average_loss = ft_loss_accum/len(train_episodes_idx)    
    acc = Correct/count_samples

    print("Fine_tuning loss: ", average_loss, "accuracy:", acc)
    
    test_acc = 0.0
    if test_episodes_idx != None:
        test_acc = test(args, test_episodes_idx, graphs_list, Dagcn, Gin, Graph_reps_gcn, FTL_Classifier, ft_epoch, device)

    return test_acc


if __name__ == '__main__':
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    NCI_Datasets = ['NCI1', 'NCI109', 'NCI-H23']


    COLORS_3_Datasets = ['COLORS_3_p0', 'COLORS_3_p1', 'COLORS_3_p2']

    Chemical_DataSets = ['MCF-7H', 'MOLT-4H', 'OVCAR-8H', 'PC-3H', 'SF-295H', 'SW-620H']


    if args.target_data in args.Local_train_data:
        print('ERROR: The target dataset is not allowed in training data')
        sys.exit()

    if args.target_data in NCI_Datasets:
        data_type = 'NCI_Datasets'
        if not os.path.exists('./pkl_datasets/' + data_type + '/'):
            os.makedirs('./pkl_datasets/' + data_type + '/')
        dataset_DIR = 'dataset/' + data_type
        for dataset in NCI_Datasets:
            data_pkl_file = './pkl_datasets/' + data_type + '/' + dataset + '.pkl'
            if not os.path.isfile(data_pkl_file):
                load_data(dataset_DIR, data_type, dataset)

    elif args.target_data in COLORS_3_Datasets:
        data_type = 'COLORS_3_Datasets'
        if not os.path.exists('./pkl_datasets/' + data_type + '/'):
            os.makedirs('./pkl_datasets/' + data_type + '/')
        dataset_DIR = 'dataset/' + data_type
        dataset = 'COLORS_3'
        data_pkl_file = './pkl_datasets/' + data_type + '/' + dataset + '.pkl'
        if not os.path.isfile(data_pkl_file):
            load_data(dataset_DIR, data_type, dataset)

    elif args.target_data in Chemical_DataSets:
        data_type = 'Chemical_DataSets'
        if not os.path.exists('./pkl_datasets/' + data_type + '/'):
            os.makedirs('./pkl_datasets/' + data_type + '/')
        dataset_DIR = 'dataset/' + data_type
        for dataset in Chemical_DataSets:
            data_pkl_file = './pkl_datasets/' + data_type + '/' + dataset + '.pkl'
            if not os.path.isfile(data_pkl_file):
                load_data(dataset_DIR, data_type, dataset)

    else:
        print('The dataset is not supported')
        sys.exit()


    pretrained_dir = './meta_pretrained_weights/Proposed_COGNN/' + data_type + '/'
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)

    if not os.path.exists('./acc_results/Proposed_COGNN/' + data_type + '/'):
        os.mkdir('./acc_results/Proposed_COGNN/' + data_type + '/')

    num_experiments = args.Num_experiments
    obtain_training_data=False
    Graphs_list = []

    data_pkl_file = './pkl_datasets/' + data_type + '/' + args.target_data + '.pkl'
    if args.FL_mode == 'few-shot':
        train_way, test_way = None, args.Test_way[0]
        _, Target_episodes_idx, target_graphs_list = create_meta_episodes(args.target_data, data_pkl_file, train_way, test_way, \
                                num_exps=num_experiments, obtain_training_data=False)
        print('global target data INFO:', args.target_data)
        print('--num_global_target_episodes, num_samples in an episode:', \
                            len(Target_episodes_idx[0]), len(Target_episodes_idx[0][0]))
    else:
        train_way, test_way = None, None
        Target_ft_idx, Target_ft_test_idx, target_graphs_list = create_meta_episodes(args.target_data, data_pkl_file, train_way, test_way, \
                                num_exps=num_experiments, obtain_training_data=False)

    Graphs_list.append(target_graphs_list)

    obtain_training_data=True
    Users_local_data = []
    num_use = 0

    for data_name in args.Local_train_data:
        data_pkl_file = './pkl_datasets/' + data_type + '/' + data_name + '.pkl'
        train_way, test_way = args.Train_way[num_use], args.Test_way[num_use+1]
        num_use += 1

        Train_episodes_idx, Test_episodes_idx, graphs_list = create_meta_episodes(data_name, data_pkl_file, train_way, test_way, \
                                num_exps=num_experiments, obtain_training_data=True)

        Users_local_data.append([Train_episodes_idx, Test_episodes_idx, graphs_list])
        Graphs_list.append(graphs_list)
        
        print('local data INFO:', data_name)
        print('--num_train_episodes, num_samples in an episode:', \
                           len(Train_episodes_idx[0]), len(Train_episodes_idx[0][0]))
        print('--num_test_episodes,  num_samples in an episode:', \
                           len(Test_episodes_idx[0]), len(Test_episodes_idx[0][0]))
        print('--number of graphs_list:', len(graphs_list))

    args.num_users =  len(args.Local_train_data)
    node_features_dim = args.feat_dim

    print('node_features dim:', node_features_dim, Graphs_list[0][0].node_features.shape[1])

    if args.feat_dim != Graphs_list[0][0].node_features.shape[1]:
        print('node_features dim some bug')
        sys.exit()        
    
    if args.sortpooling_k <= 1:
        num_nodes_sortpooling_k = 0.0
        if args.FL_mode == 'few-shot':
            for i in range(args.num_users):
                num_nodes_list = sorted([len(g.node_tags) for g in Graphs_list[i+1]])
                num_nodes_sortpooling_k += num_nodes_list[int(math.ceil(args.sortpooling_k * len(num_nodes_list))) - 1]
                
            args.sortpooling_k = int(num_nodes_sortpooling_k/args.num_users)
        else:
            for i in range(args.num_users + 1):
                num_nodes_list = sorted([len(g.node_tags) for g in Graphs_list[i]])
                num_nodes_sortpooling_k += num_nodes_list[int(math.ceil(args.sortpooling_k * len(num_nodes_list))) - 1]
                
            args.sortpooling_k = int(num_nodes_sortpooling_k/(args.num_users + 1.0))
           
        args.sortpooling_k = max(10, args.sortpooling_k)
    print('$$$$$$ k used in SortPooling is: ' + str(args.sortpooling_k))
    ### Be careful: After the server-centric provides 'args.sortpooling_k', \
    ###             each user only provides 'num_nodes_sortpooling_k' according to their own local data.
    ###             And 'target_data' can not be available in few-shot experiments

    if args.FL_mode == 'few-shot':
        f = open('./acc_results/Proposed_COGNN/' + data_type + '/'+args.target_data+'_'+str(args.test_shot)+'shot_'+ \
                 str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+'_acc_results.txt', 'w')
    else: 
        f = open('./acc_results/Proposed_COGNN/' + data_type + '/'+args.target_data+'_'+args.FL_mode+'_'+\
                 str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+'_acc_results.txt', 'w')
        
        
    few_shot_Test_acc_m = []
    few_shot_Test_acc_h = []
    transfer_best_acc   = []
           
    for exp_idx in range(num_experiments):
        print('=================================')
        cls_head = ClassificationHead(base_learner=args.DiffCO_form).to(device)
        ## ['SVM-CS', 'SVM-He', 'Ridge', 'R2D2']

        if args.FL_mode == 'few-shot':
            target_episodes_idx = Target_episodes_idx[exp_idx]
        else:
            args.num_rounds = 50

        Dagcn = Graph_Extractor(device).to(device)
        # print(Dagcn)
        Gin = GraphCNN(args, args.num_layers, args.num_mlp_layers, node_features_dim, args.hidden_dim,\
                       args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
        Graph_reps_gcn = Graph_reps_GCN(args).to(device)

        global_Dagcn_weights = Dagcn.state_dict()
        global_Gin_weights = Gin.state_dict()
        global_Class_gcn_weights = Graph_reps_gcn.state_dict()

        num_work_users = max(2, int(args.frac*args.num_users))
        best_val_acc = 0.0
        
        for global_round in range(args.num_rounds):
            local_Dagcn_weights = []
            local_Gin_weights = []
            local_Class_gcn_weights = []
            
            Dagcn.train()
            Gin.train()
            Graph_reps_gcn.train()

            print('******************************************')
            if args.FL_mode == 'few-shot':
                print('F Few-Shot L:', args.Local_train_data, '->:', args.target_data)
            else:
                print('meta-pretraining:', args.Local_train_data, '->:', args.target_data)

            users_index = [i for i in range(args.num_users)]
            random.shuffle(users_index)
            work_users = users_index[: num_work_users]

            print('work_users ID:', work_users)
            for user_idx in work_users: ## range(args.num_users):
                train_way, test_way = args.Train_way[user_idx], args.Test_way[user_idx+1]
                
                user_data = [Users_local_data[user_idx][0][exp_idx], \
                             Users_local_data[user_idx][1][exp_idx], Users_local_data[user_idx][2]]
                      

                Dagcn_cpoy = copy.deepcopy(Dagcn)
                Gin_cpoy = copy.deepcopy(Gin)
                Graph_reps_gcn_cpoy = copy.deepcopy(Graph_reps_gcn)
                

                local_w = update_local_weights(args, Dagcn_cpoy, Gin_cpoy, Graph_reps_gcn_cpoy, cls_head, \
                                               user_data, global_round, train_way, test_way, device)
                
                local_Dagcn_weights.append(local_w[0])
                local_Gin_weights.append(local_w[1])
                local_Class_gcn_weights.append(local_w[2])
                
            print('args.Agg_model:', args.Agg_model)
            if args.Agg_model == "ave":
                global_Dagcn_weights = average_weights(local_Dagcn_weights)
                global_Gin_weights = average_weights(local_Gin_weights)
                global_Class_gcn_weights = average_weights(local_Class_gcn_weights)
            elif args.Agg_model == "att":
                ### aggregate_att(w_clients, w_server, 1, 2, 0.001)
                ### aggregate_att(w_clients, w_server, args.epsilon, args.ord, args.dp)
                global_Dagcn_weights = aggregate_att(local_Dagcn_weights, global_Dagcn_weights,\
                                                     args.epsilon, args.ord, args.dp, device)
                global_Gin_weights = aggregate_att(local_Gin_weights, global_Gin_weights,\
                                                     args.epsilon, args.ord, args.dp, device)
                global_Class_gcn_weights = aggregate_att(local_Class_gcn_weights, global_Class_gcn_weights,\
                                                     args.epsilon, args.ord, args.dp, device)
            elif args.Agg_model == "ave_dp":
                global_Dagcn_weights = average_weights_dp(local_Dagcn_weights, args.dp, device)
                global_Gin_weights = average_weights_dp(local_Gin_weights, args.dp, device)
                global_Class_gcn_weights = average_weights_dp(local_Class_gcn_weights, args.dp, device)
            else:
                print('no exists agg model')
                sys.exit()
            
            Dagcn.load_state_dict(global_Dagcn_weights)
            Gin.load_state_dict(global_Gin_weights)
            Graph_reps_gcn.load_state_dict(global_Class_gcn_weights)

            Dagcn.eval()
            Gin.eval()
            Graph_reps_gcn.eval()
            
            val_acc = []
            user_m0 = []
            user_h0 = []
            for user_idx in range(args.num_users):
                train_way, test_way = args.Train_way[user_idx], args.Test_way[user_idx+1]
                
                user_data = [Users_local_data[user_idx][1][exp_idx], Users_local_data[user_idx][2]]
                test_episodes_idx, graphs_list = user_data[0], user_data[1]
                
                m, h = few_shot_test(args, Dagcn, Gin, Graph_reps_gcn, cls_head, test_episodes_idx , graphs_list, test_way , device, phase='test')

                user_m0.append(m)
                user_h0.append(h)

                val_acc.append(m)

            print('Average accuracy and h on test data of localed data:', np.mean(user_m0), np.mean(user_h0))

            if args.FL_mode == 'few-shot':
                if best_val_acc <= np.mean(val_acc):
                    best_val_acc = np.mean(val_acc)
                    
                    train_way, test_way = None, args.Test_way[0]
                    m, h = few_shot_test(args, Dagcn, Gin, Graph_reps_gcn, cls_head, target_episodes_idx, target_graphs_list, test_way, device, phase='target-test')
                    
                    test_acc_m = m
                    test_acc_h = h
                    
                    user_m = copy.deepcopy(user_m0)
                    user_h = copy.deepcopy(user_h0)

                    torch.save(Dagcn.state_dict(), pretrained_dir + \
                               args.target_data + '_Dagcn_' + str(exp_idx) + '_' + str(args.test_shot) + 'shot_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth')
                    torch.save(Gin.state_dict(), pretrained_dir + \
                               args.target_data + '_Gin_' + str(exp_idx) + '_' + str(args.test_shot) + 'shot_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth')
                    torch.save(Graph_reps_gcn.state_dict(), pretrained_dir + \
                               args.target_data + '_Graph_reps_gcn_' + str(exp_idx) + '_' + str(args.test_shot) + 'shot_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth')
                
                print('target data acc:', test_acc_m)

            elif best_val_acc <= np.mean(val_acc):
                best_val_acc = np.mean(val_acc)
                torch.save(Dagcn.state_dict(), pretrained_dir + \
                               args.target_data + '_Dagcn_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth')
                torch.save(Gin.state_dict(), pretrained_dir + \
                               args.target_data + '_Gin_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth')
                torch.save(Graph_reps_gcn.state_dict(), pretrained_dir + \
                               args.target_data + '_Graph_reps_gcn_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth')

            print('History best_val_acc:', best_val_acc)

        if args.FL_mode == 'transfer':
            Dagcn.zero_grad()
            Gin.zero_grad()
            Graph_reps_gcn.zero_grad()

            if args.num_rounds != 0: 
                Dagcn.load_state_dict(torch.load(pretrained_dir + \
                                      args.target_data + '_Dagcn_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth'))
                Gin.load_state_dict(torch.load(pretrained_dir + \
                                      args.target_data + '_Gin_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth'))
                Graph_reps_gcn.load_state_dict(torch.load(pretrained_dir + \
                                      args.target_data + '_Graph_reps_gcn_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '.pth'))

            target_ft_idx = Target_ft_idx[exp_idx]
            target_ft_test_idx = Target_ft_test_idx[exp_idx]
            FTL_Classifier = FTL_Classification(args, args.hidden_dim).to(device)

            Dagcn_optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, Dagcn.parameters()),              \
                                               lr=args.ft_learning_rate, weight_decay=args.weight_decay_value)
            Gin_optimizer   = torch.optim.Adam(filter(lambda p : p.requires_grad, Gin.parameters()),                \
                                               lr=args.ft_learning_rate, weight_decay=args.weight_decay_value)
            gcn_optimizer   = torch.optim.Adam(filter(lambda p : p.requires_grad, Graph_reps_gcn.parameters()), \
                                               lr=args.ft_learning_rate, weight_decay=args.weight_decay_value)
            FTL_optimizer   = torch.optim.Adam(filter(lambda p : p.requires_grad, FTL_Classifier.parameters()),              \
                                               lr=args.ft_learning_rate, weight_decay=args.weight_decay_value)

            if args.Fixed_weights:
                for param in Dagcn.parameters():
                    param.requires_grad = False
                    Dagcn_optimizer = None
                for param in Gin.parameters():
                    param.requires_grad = False
                    Gin_optimizer = None
                for param in Graph_reps_gcn.parameters():
                    param.requires_grad = False
                    gcn_optimizer = None

            num_target_g_train = [0 for _ in range(args.num_target_class)]
            for g_train_idx in target_ft_idx:
                for g_idx in g_train_idx:
                    num_target_g_train[target_graphs_list[g_idx].label] += 1.0

            # print('num_target_g_train:', num_target_g_train)
            loss_weights = torch.ones(args.num_target_class).to(device)
            for loss_weights_i in range(1, args.num_target_class):
                loss_weights[loss_weights_i] = \
                      num_target_g_train[0]/num_target_g_train[loss_weights_i]

            # print('loss_weights:', loss_weights)
            ft_criterion = nn.CrossEntropyLoss(weight=loss_weights)

            best_acc = 0.0
            for ft_epoch in range(args.num_ft_epochs):
                print('******************************************')
                print('F Transfer L:', args.Local_train_data, '->:', args.target_data, 'labeling rate:', args.labeling_rate)
                print('ft_epoch:', ft_epoch)

                Dagcn.train()
                Gin.train()
                Graph_reps_gcn.train()
                FTL_Classifier.train()
                
                acc = Fine_tuning(args, target_ft_idx, target_graphs_list, Dagcn, Gin, Graph_reps_gcn, FTL_Classifier, \
                            Dagcn_optimizer, Gin_optimizer, gcn_optimizer, FTL_optimizer, ft_epoch, ft_criterion, device, test_episodes_idx=target_ft_test_idx)    
                
                if best_acc <= acc:
                    best_acc = acc
                    torch.save(Dagcn.state_dict(), pretrained_dir +  args.target_data + \
                               '_Dagcn_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '_' + str(args.labeling_rate) + '.pth')
                    torch.save(Gin.state_dict(), pretrained_dir +  args.target_data + \
                               '_Gin_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '_' + str(args.labeling_rate) + '.pth')
                    torch.save(Graph_reps_gcn.state_dict(), pretrained_dir + args.target_data + \
                               '_Graph_reps_gcn_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '_' + str(args.labeling_rate) + '.pth')
                    torch.save(FTL_Classifier.state_dict(), pretrained_dir + args.target_data + \
                               '_FTL_Classifier_' + str(exp_idx) + '_' + str(args.num_GCN_layer)+'GNN_'+str(int(args.CE_lambda))+ '_' + str(args.labeling_rate) + '.pth')

        if args.FL_mode == 'few-shot':
            few_shot_Test_acc_m.append(test_acc_m)
            few_shot_Test_acc_h.append(test_acc_h)
            
            f.write(str(test_acc_m) + ',' + str(test_acc_h))
            for user_i in range(args.num_users):
                f.write(',' + str(user_m[user_i]) + ',' + str(user_h[user_i]))
            f.write('\n')
        else:
            transfer_best_acc.append(best_acc)
            f.write(str(best_acc) + '\n')

    if args.FL_mode == 'few-shot':
        print('acc mean, std:', np.mean(few_shot_Test_acc_m), np.std(few_shot_Test_acc_m))
        f.write('avg:\n')
        f.write(str(np.mean(few_shot_Test_acc_m)) + ',' + str(np.std(few_shot_Test_acc_m)))
    else:
        print('transfer acc mean, std:', np.mean(transfer_best_acc), np.std(transfer_best_acc))
        f.write('avg:\n')
        f.write(str(np.mean(transfer_best_acc)) + ',' + str(np.std(transfer_best_acc)))
            
    f.close()
