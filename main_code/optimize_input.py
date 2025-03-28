import FaultAnalysisDropout.pNN_FA as pNN
from configuration import *
from utils import *
import sys
import os
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import heapq

# Initialization
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
if not os.path.exists('../experiments/FaultAnalysis/evaluation/'):
    os.makedirs('../experiments/FaultAnalysis/evaluation/')

args = parser.parse_args([])
args = FormulateArgs(args)

args.DATASET = 0
args.SEED = 6
args.e_train = 0.0
args.N_fault = 1
args.e_fault = 0.0

args.dropout = 0.0
args.fault_ratio = 0.0


# Dataset Prepration
valid_loader, datainfo = GetDataLoader(args, 'valid', path='../dataset/')
test_loader, datainfo = GetDataLoader(args, 'test',  path='../dataset/')
pprint.pprint(datainfo)

for x, y in valid_loader:
    X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)
for x, y in test_loader:
    X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)

# # after clustering
num_fault_sub_type = [2, 6, 12, 0, 1, 3, 4]
cluster_dict_fault_sub_type, cluster_act_list_fault_sample = act_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=1, num_fault_sub_type=num_fault_sub_type)

# Load Model
topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
pnn = pNN.pNN(topology, args).to(args.DEVICE)

modelname = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{
    args.SEED:02d}_epsilon_{args.e_train}_dropout_{args.dropout}_fault_ratio_{args.fault_ratio}.model"
trained_model = torch.load(
    f'../trained_models/tanh_0_FaultAnalysisNormal/models/{modelname}')

for i, j in zip(trained_model.model, pnn.model):
    j.theta_.data = i.theta_.data.clone()

pnn.UpdateFault(N_fault=args.N_fault, e_fault=args.e_fault)
pnn.UpdateArgs(args)
