from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import networkx as nx
import pdb
import argparse
import sys
import copy
import torch
from sklearn.model_selection import StratifiedKFold

import pickle


cmd_opt = argparse.ArgumentParser(description='FFSL or FTL for graph classification')
cmd_opt.add_argument('--learn_eps', action="store_true",
                    help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
cmd_opt.add_argument('--mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('--train_bs', type=int, help='train minibatch size')
cmd_opt.add_argument('--test_bs', type=int, help='test minibatch size')
cmd_opt.add_argument('--seed', type=int, default=1, help='seed')
cmd_opt.add_argument('--feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('--edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('--num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('--latent_dim', type=str, default="32-32-32-1", help='dimension(s) of latent layers')
cmd_opt.add_argument('--sortpooling_k', type=float, default=0.6, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('--conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('--out_dim', type=int, default=0, help='graph embedding output size')
cmd_opt.add_argument('--weight_decay_value', type=float, default=5e-03, help='default: weight_decay_value=5e-03')

cmd_opt.add_argument('--DAGCN_latent_dim', type=int, default=100, help='dimension(s) of latent layers of DAGCN') ## default: 64
cmd_opt.add_argument('--multi_h_emb_weight', type=int, default=32, help='multi_h_emb_weight') ## default: 16
cmd_opt.add_argument('--max_k', type=int, default=5, help='k for capsules style')  ## default: 5
cmd_opt.add_argument('--max_block', type=int, default=1, help='num of block layer')
cmd_opt.add_argument('--dropout', type=float, default=0.5)
cmd_opt.add_argument('--reg', type=int, default=0, help='regular term')


cmd_opt.add_argument('--Num_experiments', type=int, default=10, help='default: 10')

# cmd_opt.add_argument('--Local_train_data', default=['COLORS_3_p1', 'COLORS_3_p2'],
#                                                            help='train_data   folder name')
# cmd_opt.add_argument('--target_data', default='COLORS_3_p0', help='target_data  folder name') 


cmd_opt.add_argument('--Local_train_data', default=['NCI109', 'NCI-H23'],
                                                           help='train_data   folder name')
cmd_opt.add_argument('--target_data', default='NCI1', help='target_data  folder name')


# cmd_opt.add_argument('--Local_train_data', default=['SW-620H', 'MCF-7H', 'SF-295H', 'MOLT-4H', 'OVCAR-8H'],
#                                                         help='train_data   folder name')
# cmd_opt.add_argument('--target_data', default='PC-3H', help='target_data  folder name')  ## 'P388H', 'SN12CH', 'YeastH', ('UACC257H')

# cmd_opt.add_argument('--Train_way', default=  [2,2,2,2,2], help='Be careful: min(train_way[i]) == 2, \
#                                                                       max(train_way[i]) == num_classes (i) ')
# cmd_opt.add_argument('--Test_way',  default=[2,2,2,2,2,2], help='According to the category of the dataset[NO.1(idx:0) is classes of target_data]')


cmd_opt.add_argument('--Train_way', default=  [2,2], help='Be careful: min(train_way[i]) == 2, \
                                                                       max(train_way[i]) == num_classes (i) ')
cmd_opt.add_argument('--Test_way',  default=[2,2,2], help='According to the category of the dataset[NO.1(idx:0) is classes of target_data]')


cmd_opt.add_argument('--num_target_class', type=int, default=2,  help='Be careful: #classes of target_data')


cmd_opt.add_argument('--train_shot',type=int, default=5, help='train_shot')
cmd_opt.add_argument('--test_shot', type=int, default=5, help='test_shot')
cmd_opt.add_argument('--num_query', type=int, default=50,help='num_query')

cmd_opt.add_argument('--num_epochs', type=int, default=1,  help='the number of local epochs: E')
cmd_opt.add_argument('--num_rounds', type=int, default=80, help='number of rounds of communication, default = 80')
cmd_opt.add_argument('--num_train_episodes', type=int, default=-1, help='num_train_episodes of Local_train_data')
cmd_opt.add_argument('--num_test_episodes',  type=int, default=100,  help='num_test_episodes of Local_train_data')
cmd_opt.add_argument('--num_target_episodes',type=int, default=200, help='num_target_episodes of target_data when FFSL: 200')
cmd_opt.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate when meat-train')  
### NCI_Datasets: 0.0005
### Chemical_DataSets: 0.0005
#### COLORS_3: 0.0001

cmd_opt.add_argument('--adjust_lr', type=str, default=True, help='True: adjust learning rate')
cmd_opt.add_argument('--adjust_epochs', type=int, default=40, help='adjust learning at adjust_epochs-th')
cmd_opt.add_argument('--adjust_lr_frac', type=float, default=0.2, help='new_lr =  learning_rate * adjust_lr_frac')


cmd_opt.add_argument('--labeling_rate', type=float, default=0.9, help='labeling rate for Fine_tuning_bs when FL_mode == transfer')
cmd_opt.add_argument('--Fine_tuning_bs', type=int, default=100, help='minibatch size for Fine_tuning when FL_mode == transfer')
cmd_opt.add_argument('--Fine_tuning_test_bs', type=int, default=8, help='minibatch size for Fine_tuning test when FL_mode == transfer')
cmd_opt.add_argument('--ft_learning_rate', type=float, default=0.0005, help='learning_rate when Fine_tuning')  
cmd_opt.add_argument('--num_ft_epochs', type=int, default=120, help='the number of epochs when Fine_tuning')
cmd_opt.add_argument('--Fixed_weights', type=str, default=False, help='Fixed pretrained weights when Fine_tuning')
cmd_opt.add_argument('--pre_trained', type=str, default=False, help='True: apply the pre_trained model')

cmd_opt.add_argument('--FL_mode', type=str, default='few-shot', choices=['few-shot', 'transfer'],\
                    help='few-shot: Federated Few-Shot Learning(FFSL),\
                          transfer: Federated Transfer Learning( FTL)')

cmd_opt.add_argument('--gm', default='DAGCN', help='gnn model to use[DAGCN, DGCNN]')
cmd_opt.add_argument('--num_layers', type=int, default=4,
                    help='number of layers INCLUDING the input one (default: 5 in original GIN)')
# cmd_opt.add_argument('--Input_layer', type=str, default=True,
#                     help='True: INCLUDING the Input layer in GIN')

cmd_opt.add_argument('--node_features_dim', type=int, default=10, \
                    help='If the dataset does not have node_labels or node_attributes')
cmd_opt.add_argument('--clip', dest='clip', type=float, default=4.0,
                    help='Gradient clipping.')
# cmd_opt.add_argument('--gcn_transfer_rep1', type=str, default=False,
#                     help='True: Transformation of graph representations_1 in Graph_reps_GCN')

cmd_opt.add_argument('--num_GCN_layer', type=int, default=1,
                    help='i(option: 1, 2): i Graph_reps_GCN layers')

cmd_opt.add_argument('--num_users', type=int, help='number of users: M ')
cmd_opt.add_argument('--frac', type=float, default=0.6, help="the fraction of clients: C")
cmd_opt.add_argument('--num_train_g', type=int, help='num_train_g of each Local dataset')
cmd_opt.add_argument('--num_test_g',  type=int,  default=100, help='num_test_g of each Local dataset')


cmd_opt.add_argument('--JS_div', default=True,
                    help='True: compute consistency of outputs from two views')

cmd_opt.add_argument("--warm_up", type = float, default=0.0)
cmd_opt.add_argument("--lambda_cot_max",type = float, default=200.0)
cmd_opt.add_argument('--CE_lambda', type=float, default=200.0, help='using for: CE_lambda * ce_loss + triplet_loss')

cmd_opt.add_argument("--LAMBDA_cot_max", default=[1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0], help='Parameter (lambda_cot_max) sensitivity analysis')
cmd_opt.add_argument('--CE_LAMBDA', default=     [1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0], help='Parameter (CE_lambda) sensitivity analysis')

cmd_opt.add_argument('--com_support_triplet_l', default=True,
                    help='compute the triplet loss of support samples')
cmd_opt.add_argument('--triplet_m', type = float, default=5.0,
                    help='compute the triplet loss of support samples')

cmd_opt.add_argument('--class_head', type=str, default='DiffCO',
                    help='DiffCO: Differentiable Convex Optimization; Proto: ProtoNet;')
cmd_opt.add_argument('--DiffCO_form', type=str, default='Ridge',
                    help='[SVM-CS, SVM-He, Ridge, R2D2] when class_head == DiffCO')

cmd_opt.add_argument('--Lambda_reg', type=float, default=100.0,
                    help='default: 100.0')


cmd_opt.add_argument('--Baseline', type=str, default='Baseline_2',
                    help='[Baseline_1, Baseline_2, Baseline_3, Baseline_4]')

cmd_opt.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
cmd_opt.add_argument('--num_mlp_layers', type=int, default=2,
                    help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
cmd_opt.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 64)')
cmd_opt.add_argument('--final_dropout', type=float, default=0.5,
                    help='final layer dropout (default: 0.5)')
cmd_opt.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                    help='Pooling for over nodes in a graph: sum or average')
cmd_opt.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                    help='Pooling for over neighboring nodes: sum, average or max')
cmd_opt.add_argument('--degree_as_tag', action="store_true",
                    help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')


cmd_opt.add_argument('--Agg_model', type=str, default="ave", choices=["ave", "ave_dp", "att"],
                    help='aggregate model')
cmd_opt.add_argument('--epsilon', type=float, default=1, help='stepsize')
cmd_opt.add_argument('--ord', type=int, default=2, help='similarity metric')
cmd_opt.add_argument('--dp', type=float, default=0.001, help='differential privacy')


cmd_args, _ = cmd_opt.parse_known_args()
cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]


Chemical_DataSets = ['MOLT-4H', 'MCF-7H', 'PC-3H', 'SF-295H', 'SW-620H', 'OVCAR-8H']

NCI_Datasets = ['NCI1', 'NCI109', 'NCI-H23']
COLORS_3_Datasets = ['COLORS_3', 'COLORS_3_p0', 'COLORS_3_p1', 'COLORS_3_p2']


def save_pkl(g_list0, data_type, dataset_name):
    print('saving pkl_file')
    data_pkl_file = './pkl_datasets/' + data_type + '/' + dataset_name + '.pkl'
    tmp_dict = dict()
    count = 0
    for g in g_list0:
        tmp_dict[count] = g
        count += 1

    G_data = open(data_pkl_file, 'wb')
    pickle.dump(tmp_dict, G_data)
    G_data.close()


def save_dataset_part(g_list, P_classes, data_type, dataset_name, name):
    data_p = [g for g in g_list if g.label in P_classes]
    Data_p = copy.deepcopy(data_p)

    g_data_p = []
    p_labels = []

    for g in Data_p:
        if g.label not in p_labels:
            p_labels.append(g.label)
        g.label = P_classes.index(g.label)
        g_data_p.append(g)
        
    print(name, ':', p_labels)
    print('num of g_data_p:', len(g_data_p))
    save_pkl(g_data_p, data_type, dataset_name + name)


def Split_dataset(g_list0, label_dict, data_type, dataset_name):
    print('label_dict:', label_dict)

    g_list = copy.deepcopy(g_list0)

    g_per_class = [[] for _ in range(len(label_dict))]
    num_nodes_per_calss = [[] for _ in range(len(label_dict))]
    for g in g_list:
        g_per_class[g.label].append(g)
        num_nodes_per_calss[g.label].append(g.num_nodes)

    count = 0
    for i in range(len(g_per_class)):
        count += len(g_per_class[i])
        print(len(g_per_class[i]), np.mean(num_nodes_per_calss[i]))
    print('Total number of Graphs:', count)

    g_Classes = [label for label in range(len(label_dict))]
    random.shuffle(g_Classes)
    np.savetxt('./pkl_datasets/'+data_type+'/'+dataset_name+'_classes.txt', np.array(g_Classes))

    P0_classes = g_Classes[:4]
    P1_classes = g_Classes[4:7]
    P2_classes = g_Classes[7:]

    # {2: 0, 8: 1, 9: 2, 4: 3, 6: 4, 0: 5, 7: 6, 3: 7, 10: 8, 1: 9, 5: 10}
    
    save_dataset_part(g_list, P0_classes, data_type, dataset_name, '_p0')
    save_dataset_part(g_list, P1_classes, data_type, dataset_name, '_p1')
    save_dataset_part(g_list, P2_classes, data_type, dataset_name, '_p2')
        

def create_meta_episodes(dataset_name, data_pkl_file, train_way, test_way, num_exps=10, obtain_training_data=False):
    print('&&&&& creating meta_episodes &&&&&')
    
    pkl_file = open(data_pkl_file, 'rb')
    G_data = pickle.load(pkl_file)
    pkl_file.close()

    g_list0 = []
    for key in G_data.keys():
        g_list0.append(G_data[key])
            
    g_label_list = [g.label for g in g_list0]
    if min(g_label_list) != 0 or max(g_label_list) != len(set(g_label_list)) - 1:
        print('some bug with labels of graphs')
        sys.exit()

    if dataset_name in NCI_Datasets:
        cmd_args.feat_dim = 65
    elif dataset_name in Chemical_DataSets:
        cmd_args.feat_dim = 66

    if dataset_name in COLORS_3_Datasets:
        cmd_args.feat_dim = 5

    print('# classes: %d' % len(set(g_label_list)))
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    print("# data: %d" % len(g_list0))
              

    # print('dataset_name:', dataset_name)
    # node_INF = [[] for i in range(len(set(g_label_list)))]
    # for g in g_list0:
    #     node_INF[g.label].append(g.num_nodes)

    # for i in range(len(set(g_label_list))):
    #     print('node_INF:', len(node_INF[i]), np.mean(node_INF[i]))

    num_nodes_per_g = [len(g.node_tags) for g in g_list0]
    print('Avg.nodes and Max.nodes', np.mean(num_nodes_per_g), max(num_nodes_per_g))

    g_list = copy.deepcopy(g_list0)
        
    if cmd_args.FL_mode == 'transfer' and dataset_name == cmd_args.target_data:
        graphs_idx_list = list(np.arange(len(g_list)))
        random.shuffle(graphs_idx_list)

        num_samples_exp_i = int(0.1 * len(g_list))
        multiple = int(cmd_args.labeling_rate/0.1)
        
        if num_exps > 10:
            print('must be: num_exps <= 10')
            sys.exit()
            
        Train_idx = []
        Test_idx = []
        count_exp = 0
        graphs_idx_list_double = copy.deepcopy(graphs_idx_list) + copy.deepcopy(graphs_idx_list)
        for num in range(0, len(graphs_idx_list), num_samples_exp_i):
            count_exp += 1
            if count_exp > num_exps:
                break
            
            tmp = copy.deepcopy(graphs_idx_list_double[num: num + num_samples_exp_i*multiple])
            Train_idx.append(tmp)
            Test_idx.append([g_idx for g_idx in graphs_idx_list if g_idx not in tmp])


        Target_ft_idx  = [[] for _ in range(num_exps)]
        Target_ft_test_idx   = [[] for _ in range(num_exps)]
                
        for exp_i in range(num_exps):
            for num in range(0, len(Train_idx[exp_i]), cmd_args.Fine_tuning_bs):
                if num + cmd_args.Fine_tuning_bs > len(Train_idx[exp_i]):
                    Target_ft_idx[exp_i].append(\
                                          Train_idx[exp_i][num:])
                else:
                    Target_ft_idx[exp_i].append(\
                                          Train_idx[exp_i][num:num+cmd_args.Fine_tuning_bs])
                    
            for num in range(0, len(Test_idx[exp_i]), cmd_args.Fine_tuning_test_bs):
                if num + cmd_args.Fine_tuning_test_bs > len(Test_idx[exp_i]):
                    Target_ft_test_idx[exp_i].append(\
                                          Test_idx[exp_i][num:])
                else:
                    Target_ft_test_idx[exp_i].append(\
                                          Test_idx[exp_i][num:num+cmd_args.Fine_tuning_test_bs])
        
        return Target_ft_idx, Target_ft_test_idx, g_list
        
       
    num_classes = len(set(g_label_list))
    if obtain_training_data:
        if (dataset_name in NCI_Datasets) or (dataset_name in COLORS_3_Datasets):
            cmd_args.num_train_g = 400 ## ## default: 400
            cmd_args.num_train_episodes = 200
        else:
            cmd_args.num_train_g = 200 ## ## default: 200
            cmd_args.num_train_episodes = 100
            cmd_args.num_rounds = 80
    else:
        cmd_args.num_train_g = 0

    Labels = list(np.arange(num_classes))
    train_labels = copy.deepcopy(Labels)
    test_labels = copy.deepcopy(Labels)
    
    print('train_labels:', train_labels)
    print('test_labels:', test_labels)
    
    train_shot = cmd_args.train_shot
    test_shot  = cmd_args.test_shot
    num_query  = cmd_args.num_query      

    if train_way is not None:
        if train_way > len(train_labels):
            print('error: The train_way should not exceed the total number of categories')
            sys.exit()

        cmd_args.train_bs = train_way * (train_shot + num_query)
        print('train way:', train_way)
        print('number of in a train episode:', cmd_args.train_bs)

    if test_way is not None:
        if test_way > len(test_labels):
            print('error: test_way should not exceed the total number of categories')
            sys.exit()

        cmd_args.test_bs =  test_way  * (test_shot  + num_query)
        print('test way:',  test_way)
        print('number of in a test  episode:', cmd_args.test_bs)
        
        
    if cmd_args.num_train_g != 0:
        if cmd_args.num_train_g < (train_shot + num_query):
            print('There is not enough training data. You should increase the training data')
            sys.exit()
            
    if cmd_args.num_test_g < (test_shot + num_query):
        print('There is not enough test data. You should increase the test data')
        sys.exit()
        

    graphs_idx_list = list(np.arange(len(g_list)))
    Graphs_idx_list = []
    for exp_i in range(num_exps):
        graphs_idx_list_copy = copy.deepcopy(graphs_idx_list)
        random.shuffle(graphs_idx_list_copy)
        Graphs_idx_list.append(graphs_idx_list_copy)
        
    Per_cls_idx_list  = [[[] for _ in range(num_classes)] for _ in range(num_exps)]
    Per_cls_train_idx = [[] for _ in range(num_exps)]
    Per_cls_test_idx  = [[] for _ in range(num_exps)]
    
    for exp_i in range(num_exps):
        for g_idx in Graphs_idx_list[exp_i]:
            Per_cls_idx_list[exp_i][Labels.index(g_list[g_idx].label)].append(g_idx)
        
        for cls_i in range(num_classes):
            cls_i_idx_list = copy.deepcopy(Per_cls_idx_list[exp_i][cls_i])
            Per_cls_train_idx[exp_i].append(cls_i_idx_list[:cmd_args.num_train_g])
            Per_cls_test_idx[exp_i].append( cls_i_idx_list[ cmd_args.num_train_g:(cmd_args.num_train_g + cmd_args.num_test_g)])
            
            print('num of per train or test cls:', len(Per_cls_train_idx[exp_i][cls_i]), len(Per_cls_test_idx[exp_i][cls_i]))

    if cmd_args.num_train_g != 0:
        ### obteain Train_episodes_idx ## for meta-train few-shot
        Train_episodes_idx  = [[] for _ in range(num_exps)]
        for exp_i in range(num_exps):
            for num in range(cmd_args.num_train_episodes):
                train_labels_copy = copy.deepcopy(train_labels)
                random.shuffle(train_labels_copy)
                select_classes = train_labels_copy[:train_way]

                tmp = []
                for select_class in select_classes:
                    select_class_labeled_samples_idx = copy.deepcopy(Per_cls_train_idx[exp_i][train_labels.index(select_class)])
                    random.shuffle(select_class_labeled_samples_idx)
                    tmp.append(select_class_labeled_samples_idx[: train_shot + num_query])

                train_episode = (np.array(tmp).transpose().reshape(1, -1)).tolist()[0]
                Train_episodes_idx[exp_i].append(train_episode)
        ### obteain Train_episodes_idx ## for meta-train few-shot
        
    
        ### obteain Test_episodes_idx ## for val in Federated Few-Shot Learning(FFSL) or Federated Transfer Learning(FTL)
        Test_episodes_idx = [[] for i in range(num_exps)]
        for exp_i in range(num_exps):
            for num in range(cmd_args.num_test_episodes):
                test_labels_copy = copy.deepcopy(test_labels) ##############
                random.shuffle(test_labels_copy) ##############
                select_classes = test_labels_copy[:test_way]
                #### select_classes = test_labels[:test_way]
                tmp = []
                for select_class in select_classes:
                    test_label_i_idx = copy.deepcopy(Per_cls_test_idx[exp_i][test_labels.index(select_class)])
                    random.shuffle(test_label_i_idx)
                    # test_support_idx.append(test_label_i_idx[:test_way * test_shot * 2])
                    tmp.append(test_label_i_idx[:(test_shot + num_query)])
                
                test_episode = (np.array(tmp).transpose().reshape(1, -1)).tolist()[0]
                Test_episodes_idx[exp_i].append(test_episode)
        ### obteain Test_episodes_idx ## for val in Federated Few-Shot Learning(FFSL) or Federated Transfer Learning(FTL)
        
        return Train_episodes_idx, Test_episodes_idx, g_list
    
    else:
        ### obteain Target_episodes_idx ## for test in Federated Few-Shot Learning(FFSL)
        Target_episodes_idx  = [[] for _ in range(num_exps)]
        for exp_i in range(num_exps):
            for num in range(cmd_args.num_target_episodes):
                test_labels_copy = copy.deepcopy(test_labels) ##############
                random.shuffle(test_labels_copy) ##############
                select_classes = test_labels_copy[:test_way]

                tmp = []
                for select_class in select_classes:
                    test_label_i_idx = copy.deepcopy(Per_cls_test_idx[exp_i][test_labels.index(select_class)])
                    random.shuffle(test_label_i_idx)
                    # test_support_idx.append(test_label_i_idx[:test_way * test_shot * 2])
                    tmp.append(test_label_i_idx[: (test_shot + num_query)])

                target_episode = (np.array(tmp).transpose().reshape(1, -1)).tolist()[0]
                Target_episodes_idx[exp_i].append(target_episode)
        ### obteain Target_episodes_idx ## for test in Federated Few-Shot Learning(FFSL)     
    
        return Target_episodes_idx, Target_episodes_idx, g_list


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0

        self.num_nodes = len(node_tags)

        self.max_neighbor = 0

        self.degs = list(dict(g.degree()).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])


def load_data(dataset_DIR, data_type, dataset_name, degree_as_tag=False):
    '''
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s/%s.txt' % (dataset_DIR, dataset_name, dataset_name), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
                
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # print('no node attributes')
                    row = [int(w) for w in row]
                    attr = None
                else:
                    # print('have node attributes')
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])

                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if attr is not None: # tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_features = torch.from_numpy(node_features).float()
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags, node_features=node_features))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        #####  added by me (9-16)
        if edges == []:
            # print('None edges:', edges)
            edges.append([0, 0])
        #####  added by me (9-16)
        
        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    len_tagset = len(tagset)
    tag2index = {tagset[i]:i for i in range(len_tagset)}

    print(dataset_name, 'maximum node tag:', len_tagset)


    ### node feature
    if dataset_name in NCI_Datasets:
        len_tagset = 65
    elif dataset_name in Chemical_DataSets:
        len_tagset = 66

    if not node_feature_flag:
        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len_tagset)
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

            if g.num_nodes != len(g.node_tags):
                print('feat_dim some bug')
                sys.exit()
        ### node feature
        ### g.node_features == node_feat ###  GIN DGCNN DAGCN


    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim  = len(feat_dict) # maximum node label (tag)

    if dataset_name in NCI_Datasets:
        cmd_args.feat_dim = 65
    elif dataset_name in Chemical_DataSets:
        cmd_args.feat_dim = 66

    if dataset_name in COLORS_3_Datasets:
        cmd_args.feat_dim = node_features.shape[1]

    print('# classes: %d' % cmd_args.num_class)
    print('# feat_dim of node: %d' % cmd_args.feat_dim)
    print('# maximum node tag: %d' % len_tagset)
    print("# data: %d" % len(g_list))

    if dataset_name not in COLORS_3_Datasets and cmd_args.feat_dim != len_tagset:
        print('feat_dim some bug')
        sys.exit()
        

    if dataset_name in Chemical_DataSets or dataset_name == 'NCI-H23':
        if not os.path.exists('./data_npy/' + data_type + '_npy/'):
            os.mkdir('./data_npy/' + data_type + '_npy/')

        G_list = copy.deepcopy(g_list)
        g_list = []
        g_list_1 = []
        for g in G_list:
            if g.label == 0:
                g_list.append(g)
            elif g.label == 1:
                g_list_1.append(g)
            else:
                print('error labels')
                sys.exit()
            
        if len(g_list) > len(g_list_1):
            tmp_g_list = copy.deepcopy(g_list_1)
            g_list_1 = copy.deepcopy(g_list)
            g_list = copy.deepcopy(tmp_g_list)
            
        npy_file = './data_npy/'+data_type+'_npy/' + dataset_name + '.npy'
        if not os.path.isfile(npy_file):
            graphs_1_idx_list = list(np.arange(len(g_list_1)))
            random.shuffle(graphs_1_idx_list)
            np.save(npy_file, np.array(graphs_1_idx_list))
            
        graphs_1_idx_list = list(np.load(npy_file))

        # if len(graphs_1_idx_list) // len(g_list) >= 2:
        #     choose_num_g = 2*len(g_list)
        # else:
        choose_num_g = len(g_list)

        g_list_1_ = [g_list_1[i] for i in graphs_1_idx_list[:choose_num_g]]
        g_list = g_list + g_list_1_

    save_pkl(g_list, data_type, dataset_name)
    if dataset_name in ['COLORS_3']:
        Split_dataset(g_list, label_dict, data_type, dataset_name)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def cos_metric(a, b):

    a = a.float()
    b = b.float()
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = cos(a,b)

    return logits