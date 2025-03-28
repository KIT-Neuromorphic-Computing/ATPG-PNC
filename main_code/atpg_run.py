
# # Import Libraries


import json
import torch.nn.functional as F
from matplotlib.colors import Normalize, TwoSlopeNorm
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
import time


# # Initialization


sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
if not os.path.exists('../experiments/FaultAnalysis/evaluation/'):
    os.makedirs('../experiments/FaultAnalysis/evaluation/')

args = parser.parse_args()
args = FormulateArgs(args)

# args.DATASET = 2
args.SEED = 6
args.e_train = 0.0
args.N_fault = 1
args.e_fault = 0.0

args.dropout = 0.0
args.fault_ratio = 0.0


# # Load Model and Dataset Prepration


# Update the args


valid_loader, datainfo = GetDataLoader(args, 'valid', path='../dataset/')
test_loader, datainfo = GetDataLoader(args, 'test',  path='../dataset/')
pprint.pprint(datainfo)

for x, y in valid_loader:
    X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)
for x, y in test_loader:
    X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)


# Load the Model
topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]

tmp_pnn = pNN.pNN(topology, args).to(args.DEVICE)

best_acc = 0.0
pnn = pNN.pNN(topology, args).to(args.DEVICE)
for seed in range(10):

    args.SEED = seed

    modelname = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{
        args.SEED:02d}_epsilon_{args.e_train}_dropout_{args.dropout}_fault_ratio_{args.fault_ratio}.model"
    trained_model = torch.load(
        f'../trained_models/tanh_0_FaultAnalysisNormal/models/{modelname}')

    for i, j in zip(trained_model.model, tmp_pnn.model):
        j.theta_.data = i.theta_.data.clone()

    tmp_pnn.UpdateFault(N_fault=args.N_fault, e_fault=args.e_fault)
    tmp_pnn.UpdateArgs(args)

    pred_valid = tmp_pnn(X_valid)[0, 0, :, :]
    base_acc_valid = (torch.argmax(pred_valid, dim=1) ==
                      y_valid).sum() / y_valid.numel()
    print(f"Seed: {seed}, Accuracy: {base_acc_valid}")
    if best_acc < base_acc_valid:
        best_acc = base_acc_valid
        best_seed = seed
        pnn = pNN.pNN(topology, args).to(args.DEVICE)
        for i, j in zip(trained_model.model, pnn.model):
            j.theta_.data = i.theta_.data.clone()
            pnn.UpdateFault(N_fault=args.N_fault, e_fault=args.e_fault)
            pnn.UpdateArgs(args)


pred_valid = pnn(X_valid)[0, 0, :, :]
base_acc_valid = (torch.argmax(pred_valid, dim=1) ==
                  y_valid).sum() / y_valid.numel()
base_acc_valid


# # Analysis


pred_valid = pnn(X_valid)[0, 0, :, :]
base_acc_valid = (torch.argmax(pred_valid, dim=1) ==
                  y_valid).sum() / y_valid.numel()
base_acc_valid


topology


# ## Define Methods


def act_get_drop_acc(pnn, X_valid, y_valid, base_acc_valid, type_fault=1, num_fault_sub_type=[]):
    # type of fault: 0 - theta, 1 - act, 2 - neg
    # dict_fault_sub_type = {i: [] for i in range(num_fault_sub_type)}
    dict_fault_sub_type = {i: [] for i in num_fault_sub_type}
    list_sample_fault = []
    for layer_i, num_neuron in enumerate(topology[1:]):
        # list of faulty layers starting from 0
        faulty_layer_list = [layer_i]
        for neuron_i in range(num_neuron):
            # index of element within the layer
            indice_to_modify = neuron_i
            # for sub_fault_i in range(num_fault_sub_type):
            for sub_fault_i in num_fault_sub_type:
                # type of fault within the non-linear circuit
                fault_type_non_linear = sub_fault_i

                sample_fault = (faulty_layer_list, type_fault,
                                indice_to_modify, fault_type_non_linear)
                list_sample_fault.append(sample_fault)

                pred_valid = pnn(X_valid, faulty_layer_list=faulty_layer_list, type_fault=type_fault,
                                 indice_to_modify=indice_to_modify, fault_type_non_linear=fault_type_non_linear)[0, 0, :, :]

                acc_valid = (torch.argmax(pred_valid, dim=1) ==
                             y_valid).sum() / y_valid.numel()
                drop_acc = base_acc_valid - acc_valid
                dict_fault_sub_type[sub_fault_i].append(drop_acc)
                # print(f'layer: {layer_i}, neuron: {neuron_i}, sub_fault: {sub_fault_i}, drop_acc: {drop_acc}')
    return dict_fault_sub_type, list_sample_fault


def theta_get_drop_acc(pnn, X_valid, y_valid, base_acc_valid, type_fault=0, num_fault_sub_type=2):
    # type of fault: 0 - theta, 1 - act, 2 - neg
    dict_fault_sub_type = {i: [] for i in range(num_fault_sub_type)}
    list_sample_fault = []
    for layer_i, _ in enumerate(topology[1:]):
        # list of faulty layers starting from 0
        faulty_layer_list = [layer_i]
        for connection_i in range(torch.prod(torch.tensor(pnn.model[layer_i].theta_.shape))):
            # index of element within the layer
            indice_to_modify = connection_i
            for sub_fault_i in range(num_fault_sub_type):
                # type of fault within the non-linear circuit
                fault_type_non_linear = sub_fault_i

                sample_fault = (faulty_layer_list, type_fault,
                                indice_to_modify, fault_type_non_linear)
                list_sample_fault.append(sample_fault)

                pred_valid = pnn(X_valid, faulty_layer_list=faulty_layer_list, type_fault=type_fault,
                                 indice_to_modify=indice_to_modify, fault_type_non_linear=fault_type_non_linear)[0, 0, :, :]

                acc_valid = (torch.argmax(pred_valid, dim=1) ==
                             y_valid).sum() / y_valid.numel()
                drop_acc = base_acc_valid - acc_valid
                dict_fault_sub_type[sub_fault_i].append(drop_acc)
                # print(f'layer: {layer_i}, connection: {connection_i}, sub_fault: {sub_fault_i}, drop_acc: {drop_acc}')
    return dict_fault_sub_type, list_sample_fault


def inv_get_drop_acc(pnn, X_valid, y_valid, base_acc_valid, type_fault=2, num_fault_sub_type=[]):
    # type of fault: 0 - theta, 1 - act, 2 - neg
    dict_fault_sub_type = {i: [] for i in num_fault_sub_type}
    list_sample_fault = []
    for layer_i, _ in enumerate(topology[1:]):
        # list of faulty layers starting from 0
        faulty_layer_list = [layer_i]
        for input_i in range(torch.tensor(pnn.model[layer_i].theta_.shape[0])):
            # index of element within the layer
            indice_to_modify = input_i
            for sub_fault_i in num_fault_sub_type:
                # type of fault within the non-linear circuit
                fault_type_non_linear = sub_fault_i

                sample_fault = (faulty_layer_list, type_fault,
                                indice_to_modify, fault_type_non_linear)
                list_sample_fault.append(sample_fault)

                pred_valid = pnn(X_valid, faulty_layer_list=faulty_layer_list, type_fault=type_fault,
                                 indice_to_modify=indice_to_modify, fault_type_non_linear=fault_type_non_linear)[0, 0, :, :]

                acc_valid = (torch.argmax(pred_valid, dim=1) ==
                             y_valid).sum() / y_valid.numel()
                drop_acc = base_acc_valid - acc_valid
                dict_fault_sub_type[sub_fault_i].append(drop_acc)
                # print(f'layer: {layer_i}, connection: {input_i}, sub_fault: {sub_fault_i}, drop_acc: {drop_acc}')
    return dict_fault_sub_type, list_sample_fault


# ## Plot Heatmap


dict_fault_sub_type, act_list_fault_sample = act_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=1, num_fault_sub_type=range(12))

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(
    np.array([dict_fault_sub_type[i] for i in dict_fault_sub_type.keys()]) * 100)

# Plot the heatmap using Seaborn for annotations
# plt.figure(figsize=(12, 8))
# Create a figure and axis
fig, axs = plt.subplots(3, 1, figsize=(18, 18), gridspec_kw={
                        'height_ratios': [1, 1, 1]})

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[0], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[0].set_title(
    "Impact of Tanh Faults on Accuracy Across Layers and Neurons", fontsize=14)
axs[0].set_xlabel("Layer-Neuron", fontsize=12)
axs[0].set_ylabel("Tanh Fault Types", fontsize=12)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

# -----------------------------------------

dict_fault_sub_type, inv_list_fault_sample = inv_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=2, num_fault_sub_type=range(18))

# Define layers and neurons (x-axis)
layers_neurons = [f"L{i}-I{j}" for i in range(len(topology[:-1]))
                  for j in range(pnn.model[i].theta_.shape[0])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"INV Fault {i}" for i in dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(
    np.array([dict_fault_sub_type[i] for i in dict_fault_sub_type.keys()]) * 100)

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[1], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[1].set_title(
    "Impact of Invertor Faults on Accuracy Across Layers and Inputs", fontsize=14)
axs[1].set_xlabel("Layer-Inputs", fontsize=12)
axs[1].set_ylabel("Invertor Fault Types", fontsize=12)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")

# -----------------------------------------

dict_fault_sub_type, theta_list_fault_sample = theta_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=0, num_fault_sub_type=2)

# Define layers and neurons (x-axis)
layers_neurons = [f"L{i}-C{j}" for i in range(len(topology[1:])) for j in range(
    torch.prod(torch.tensor(pnn.model[i].theta_.shape)))]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Theta Fault {i}" for i in dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(
    np.array([dict_fault_sub_type[i] for i in dict_fault_sub_type.keys()]) * 100)

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[2], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[2].set_title(
    "Impact of Theta Faults on Accuracy Across Layers and Connections", fontsize=14)
axs[2].set_xlabel("Layer-Connection", fontsize=12)
axs[2].set_ylabel("Theta Fault Types", fontsize=12)
axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha="right")


plt.tight_layout()
plt.show()


# ## Plot Tanh Heatmap


# after clustering
# num_fault_sub_type = [2, 6, 12, 0, 1, 3, 4]
num_fault_sub_type = np.array([3, 11, 5, 0, 2, 10, 1]) + 1
cluster_dict_fault_sub_type, cluster_act_list_fault_sample = act_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=1, num_fault_sub_type=num_fault_sub_type)

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in cluster_dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(np.array(
    [cluster_dict_fault_sub_type[i] for i in cluster_dict_fault_sub_type.keys()]) * 100)

# Plot the heatmap using Seaborn for annotations
# plt.figure(figsize=(12, 8))
# Create a figure and axis
fig, axs = plt.subplots(2, 1, figsize=(
    18, 18), gridspec_kw={'height_ratios': [1, 1]})

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[0], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[0].set_title(
    "(After Clustering) Impact of Tanh Faults on Accuracy Across Layers and Neurons", fontsize=14)
axs[0].set_xlabel("Layer-Neuron", fontsize=12)
axs[0].set_ylabel("Tanh Fault Types", fontsize=12)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

# -----------------------------------------

# before clustering
num_fault_sub_type = range(1, 13)
all_dict_fault_sub_type, all_act_list_fault_sample = act_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=1, num_fault_sub_type=num_fault_sub_type)

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in all_dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(np.array(
    [all_dict_fault_sub_type[i] for i in all_dict_fault_sub_type.keys()]) * 100)

# Plot the heatmap using Seaborn for annotations
# plt.figure(figsize=(12, 8))
# Create a figure and axis
# fig, axs = plt.subplots(3, 1, figsize=(18, 18), gridspec_kw={'height_ratios': [1, 1, 1]})

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[1], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[1].set_title(
    "(Before Clustering) Impact of Tanh Faults on Accuracy Across Layers and Neurons", fontsize=14)
axs[1].set_xlabel("Layer-Neuron", fontsize=12)
axs[1].set_ylabel("Tanh Fault Types", fontsize=12)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")


len(all_act_list_fault_sample), len(cluster_act_list_fault_sample)


def act_get_output_differences(pnn, test_inputs, topology, type_fault=1, num_fault_sub_type=[]):
    """
    Calculate the average output differences caused by faults for each layer and neuron.

    Args:
        pnn: The neural network model.
        test_inputs: A batch of test inputs (tensor).
        topology: List of layer sizes in the network.
        type_fault: Type of fault (0 - theta, 1 - act, 2 - neg).
        num_fault_sub_type: List of fault sub-types to analyze.

    Returns:
        dict_fault_sub_type: A dictionary mapping fault sub-types to their impact across neurons.
        list_sample_fault: A list of all fault configurations tested.
    """
    # Dictionary to store the impact for each fault sub-type
    dict_fault_sub_type = {i: [] for i in num_fault_sub_type}
    list_sample_fault = []
    active_faults = []
    inactive_faults = []

    # Loop over all layers and neurons
    for layer_i, num_neurons in enumerate(topology[1:]):  # Skip input layer
        # List of faulty layers starting from 0
        faulty_layer_list = [layer_i]

        for neuron_i in range(num_neurons):  # Loop through neurons in the layer
            indice_to_modify = neuron_i

            for sub_fault_i in num_fault_sub_type:  # Loop through fault sub-types
                fault_type_non_linear = sub_fault_i

                # Create a fault configuration
                sample_fault = (faulty_layer_list, type_fault,
                                indice_to_modify, fault_type_non_linear)
                list_sample_fault.append(sample_fault)

                # Compute fault-free outputs
                output_fault_free = pnn(test_inputs)[0, 0, :, :]

                # Compute faulty outputs
                output_faulty = pnn(test_inputs, faulty_layer_list=faulty_layer_list,
                                    type_fault=type_fault,
                                    indice_to_modify=indice_to_modify,
                                    fault_type_non_linear=fault_type_non_linear)[0, 0, :, :]

                # Calculate the average output difference for this fault
                # diff = torch.mean(torch.norm(output_faulty - output_fault_free, dim=1)).item()
                diff = torch.nn.functional.mse_loss(
                    output_faulty, output_fault_free).item()

                # Store the impact for this fault sub-type
                dict_fault_sub_type[sub_fault_i].append(diff)

                # Print for debugging
                # print(f"Layer: {layer_i}, Neuron: {neuron_i}, Sub-Fault: {sub_fault_i}, Avg Difference: {diff}")

                # Sum differences over all test inputs for the fault
                sum_diff = torch.sum(
                    torch.abs(output_faulty - output_fault_free)).item()

                # Classify faults as active or inactive
                if sum_diff > 0:
                    active_faults.append(sample_fault)
                else:
                    inactive_faults.append(sample_fault)

    return dict_fault_sub_type, list_sample_fault, active_faults, inactive_faults


# num_fault_sub_type = [2, 6, 12, 0, 1, 3, 4]
num_fault_sub_type = np.array([3, 11, 5, 0, 2, 10, 1]) + 1
test_input_dim = topology[0]
test_inputs = torch.rand(1000, test_input_dim)  # Adjust input dimension

dict_fault_sub_type, list_act_sample_fault, act_active_faults, act_inactive_faults = act_get_output_differences(
    pnn=pnn,
    test_inputs=test_inputs,
    topology=topology,
    type_fault=1,
    num_fault_sub_type=num_fault_sub_type)

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in dict_fault_sub_type.keys()]

# Simulate output difference data (rows: faults, columns: layer-neurons)
output_differences = np.array([dict_fault_sub_type[i]
                              for i in dict_fault_sub_type.keys()])

# Apply normalization to make zero values distinctly green
norm = TwoSlopeNorm(vmin=0, vcenter=0.01, vmax=np.max(output_differences))

# Plot the heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(
    output_differences,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn_r",  # Diverging colormap
    xticklabels=layers_neurons,
    yticklabels=tanh_faults,
    cbar_kws={'label': 'Output Difference (%)'},
    norm=norm  # Use TwoSlopeNorm for distinct green at zero
)

# Add titles and labels
plt.title("(After Clustering) Impact of Tanh Faults on Output Differences Across Layers and Neurons", fontsize=16)
plt.xlabel("Layer-Neuron", fontsize=14)
plt.ylabel("Tanh Fault Types", fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Adjust layout to fit labels properly
plt.tight_layout()
plt.show()


len(all_act_list_fault_sample), len(
    act_active_faults), len(act_inactive_faults)


act_active_faults


# ## Plot Invetor Heatmap


# after clustering
num_fault_sub_type = np.array([0, 4, 1, 16, 2, 15, 8, 11, 6]) + 1
cluster_dict_fault_sub_type, cluster_inv_list_fault_sample = inv_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=2, num_fault_sub_type=num_fault_sub_type)

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in cluster_dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(np.array(
    [cluster_dict_fault_sub_type[i] for i in cluster_dict_fault_sub_type.keys()]) * 100)

# Plot the heatmap using Seaborn for annotations
# plt.figure(figsize=(12, 8))
# Create a figure and axis
fig, axs = plt.subplots(2, 1, figsize=(
    18, 18), gridspec_kw={'height_ratios': [1, 1]})

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[0], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[0].set_title(
    "(After Clustering) Impact of Tanh Faults on Accuracy Across Layers and Neurons", fontsize=14)
axs[0].set_xlabel("Layer-Neuron", fontsize=12)
axs[0].set_ylabel("Tanh Fault Types", fontsize=12)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

# -----------------------------------------

# before clustering
num_fault_sub_type = range(1, 19)
all_dict_fault_sub_type, all_inv_list_fault_sample = inv_get_drop_acc(
    pnn, X_valid, y_valid, base_acc_valid, type_fault=2, num_fault_sub_type=num_fault_sub_type)

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in all_dict_fault_sub_type.keys()]

# Simulate accuracy drop data (rows: faults, columns: layer-neurons)
accuracy_drop = np.int16(np.array(
    [all_dict_fault_sub_type[i] for i in all_dict_fault_sub_type.keys()]) * 100)

# Plot the heatmap using Seaborn for annotations
# plt.figure(figsize=(12, 8))
# Create a figure and axis
# fig, axs = plt.subplots(3, 1, figsize=(18, 18), gridspec_kw={'height_ratios': [1, 1, 1]})

sns.heatmap(accuracy_drop, annot=True, fmt="d", cmap="RdYlGn_r", ax=axs[1], cbar_kws={
            'label': 'Accuracy Drop (%)'}, xticklabels=layers_neurons, yticklabels=tanh_faults)
axs[1].set_title(
    "(Before Clustering) Impact of Tanh Faults on Accuracy Across Layers and Neurons", fontsize=14)
axs[1].set_xlabel("Layer-Neuron", fontsize=12)
axs[1].set_ylabel("Tanh Fault Types", fontsize=12)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")


len(all_inv_list_fault_sample), len(cluster_inv_list_fault_sample)


def inv_get_output_differences(pnn, test_inputs, topology, type_fault=1, num_fault_sub_type=[]):
    """
    Calculate the average output differences caused by faults for each layer and neuron.

    Args:
        pnn: The neural network model.
        test_inputs: A batch of test inputs (tensor).
        topology: List of layer sizes in the network.
        type_fault: Type of fault (0 - theta, 1 - act, 2 - neg).
        num_fault_sub_type: List of fault sub-types to analyze.

    Returns:
        dict_fault_sub_type: A dictionary mapping fault sub-types to their impact across neurons.
        list_sample_fault: A list of all fault configurations tested.
    """
    # Dictionary to store the impact for each fault sub-type
    dict_fault_sub_type = {i: [] for i in num_fault_sub_type}
    list_sample_fault = []
    active_faults = []
    inactive_faults = []

    # Loop over all layers and neurons
    for layer_i, num_neurons in enumerate(topology[1:]):  # Skip input layer
        # List of faulty layers starting from 0
        faulty_layer_list = [layer_i]

        for input_i in range(torch.tensor(pnn.model[layer_i].theta_.shape[0])):
            # index of element within the layer
            indice_to_modify = input_i

            for sub_fault_i in num_fault_sub_type:
                # type of fault within the non-linear circuit
                fault_type_non_linear = sub_fault_i

                # Create a fault configuration
                sample_fault = (faulty_layer_list, type_fault,
                                indice_to_modify, fault_type_non_linear)
                list_sample_fault.append(sample_fault)

                # Compute fault-free outputs
                output_fault_free = pnn(test_inputs)[0, 0, :, :]

                # Compute faulty outputs
                output_faulty = pnn(test_inputs, faulty_layer_list=faulty_layer_list,
                                    type_fault=type_fault,
                                    indice_to_modify=indice_to_modify,
                                    fault_type_non_linear=fault_type_non_linear)[0, 0, :, :]

                # Calculate the average output difference for this fault
                # diff = torch.mean(torch.norm(output_faulty - output_fault_free, dim=1)).item()
                diff = torch.nn.functional.mse_loss(
                    output_faulty, output_fault_free).item()

                # Store the impact for this fault sub-type
                dict_fault_sub_type[sub_fault_i].append(diff)

                # Print for debugging
                # print(f"Layer: {layer_i}, Neuron: {neuron_i}, Sub-Fault: {sub_fault_i}, Avg Difference: {diff}")

                # Sum differences over all test inputs for the fault
                sum_diff = torch.sum(
                    torch.abs(output_faulty - output_fault_free)).item()

                # Classify faults as active or inactive
                if sum_diff > 0:
                    active_faults.append(sample_fault)
                else:
                    inactive_faults.append(sample_fault)

    return dict_fault_sub_type, list_sample_fault, active_faults, inactive_faults


num_fault_sub_type = np.array([0, 4, 1, 16, 2, 15, 8, 11, 6]) + 1
test_input_dim = topology[0]
test_inputs = torch.rand(1000, test_input_dim)  # Adjust input dimension

dict_fault_sub_type, inv_list_sample_fault, inv_active_faults, inv_inactive_faults = inv_get_output_differences(
    pnn=pnn,
    test_inputs=test_inputs,
    topology=topology,
    type_fault=2,
    num_fault_sub_type=num_fault_sub_type)

# Define layers and neurons (x-axis)
layers_neurons = [
    f"L{i}-N{j}" for i in range(len(topology[1:])) for j in range(topology[i+1])]

# Define Tanh fault types (y-axis)
tanh_faults = [f"Tanh Fault {i}" for i in dict_fault_sub_type.keys()]

# Simulate output difference data (rows: faults, columns: layer-neurons)
output_differences = np.array([dict_fault_sub_type[i]
                              for i in dict_fault_sub_type.keys()])

# Apply normalization to make zero values distinctly green
norm = TwoSlopeNorm(vmin=0, vcenter=0.01, vmax=np.max(output_differences))

# Plot the heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(
    output_differences,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn_r",  # Diverging colormap
    xticklabels=layers_neurons,
    yticklabels=tanh_faults,
    cbar_kws={'label': 'Output Difference (%)'},
    norm=norm  # Use TwoSlopeNorm for distinct green at zero
)

# Add titles and labels
plt.title("(After Clustering) Impact of Tanh Faults on Output Differences Across Layers and Neurons", fontsize=16)
plt.xlabel("Layer-Neuron", fontsize=14)
plt.ylabel("Tanh Fault Types", fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Adjust layout to fit labels properly
plt.tight_layout()
plt.show()


len(all_inv_list_fault_sample), len(
    inv_active_faults), len(inv_inactive_faults)


# # Optimize test input


# ## Initialization


# Freez the model
for param in pnn.parameters():
    param.requires_grad = False
test_input_dim = topology[0]


# Define the type of fault


'''
faulty_layer_list: list of faulty layers for initial stage it has only one element
indice_to_modify: An indice to modify, one of the theta, act or neg
fault_type_non_linear: type of fault within the non-linear circuit
type_fault: 0 - theta, 1 - act, 2 - neg
'''
# faulty_layer_list = [0]
# indice_to_modify = 2
# fault_type_non_linear = 4
# type_fault = 1


# Creat a pool of test inputs


topology, len(act_list_fault_sample), len(
    inv_list_fault_sample), len(theta_list_fault_sample)


fault_pool = act_list_fault_sample + \
    inv_list_fault_sample + theta_list_fault_sample
len(fault_pool)


# range of the input


torch.min(X_valid), torch.max(X_valid)


# ## Classes and Methods


class STEMultiClass(torch.nn.Module):

    def __init__(self, input_init, min_value=0.0, max_value=1.0,):
        '''
        input_init: Initial input to the model
        min_value: Minimum value for the input
        max_value: Maximum value for the input
        faulty_layer_list: list of faulty layers
        type_fault: 0 - theta, 1 - act, 2 - neg
        indice_to_modify: An indice to modify, one of the theta, act or neg
        fault_type_non_linear: type of fault within the non-linear circuit
        '''
        super().__init__()
        # Learnable parameter for the input
        self.test_input_ = torch.nn.Parameter(
            input_init.clone(), requires_grad=True)
        self.min_value = min_value
        self.max_value = max_value

    @property
    def test_input(self):
        # Clamp the test_input_ to the defined range and apply thresholding
        clamped_input = self.test_input_.clamp(self.min_value, self.max_value)
        # clamped_input[clamped_input.abs() < self.min_value] = 0.0
        return clamped_input

    def forward(self, pretrained_model, all_fault_list, selected_faults_indices):
        # Apply the STE logic to the clamped and thresholded input
        # test_input_with_ste = self.test_input.detach() + self.test_input_ - self.test_input_.detach()

        # Forward pass through the faulty model
        # faulty_output_list = [pretrained_model(test_input_with_ste, *all_fault_list[index_])[0, 0, :, :] for index_ in selected_faults_indices]
        faulty_output = pretrained_model(
            self.test_input, *all_fault_list[selected_faults_indices[0]])[0, 0, :, :]
        # faulty_output = pretrained_model(
        #         test_input_with_ste,
        #         faulty_layer_list=self.faulty_layer_list,
        #         type_fault=self.type_fault,
        #         indice_to_modify=self.indice_to_modify,
        #         fault_type_non_linear=self.fault_type_non_linear
        #     )[0, 0, :, :]  # Adjust slicing as necessary

        # Forward pass through the fault-free model
        fault_free_output = pretrained_model(
            self.test_input,
        )[0, 0, :, :]  # Adjust slicing as necessary

        return faulty_output, fault_free_output


# ### Loss Functions


def compute_mse_loss(faulty_output, fault_free_output):
    print(faulty_output, fault_free_output)
    faulty_probs = torch.softmax(faulty_output, dim=0)
    fault_free_probs = torch.softmax(fault_free_output, dim=0)

    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(faulty_probs, fault_free_probs)
    return -loss  # Negative to maximize difference


def compute_cl_lass(faulty_output, fault_free_output):
    faulty_probs = torch.softmax(faulty_output, dim=0)
    fault_free_probs = torch.softmax(fault_free_output, dim=0)
    loss = torch.nn.functional.cross_entropy(faulty_probs, fault_free_probs)
    return -loss


def compute_total_loss(fault_free_output, faulty_outputs):
    total_loss = 0.0
    for idx, faulty_output in enumerate(faulty_outputs):
        loss = compute_mse_loss(faulty_output, fault_free_output)
        total_loss += loss
    return total_loss


def compute_classification_loss(faulty_output, fault_free_output):
    """
    Computes classification loss based on class prediction differences.
    Loss is 0 if classes are the same; otherwise, 1 (to maximize difference).
    """
    faulty_probs = torch.softmax(faulty_output, dim=0)
    fault_free_probs = torch.softmax(fault_free_output, dim=0)

    # Compare predicted classes
    class_faulty = torch.argmax(faulty_probs)
    class_fault_free = torch.argmax(fault_free_probs)

    # Loss is 1 if classes differ, 0 otherwise
    loss = 1.0 if class_faulty != class_fault_free else 0.0
    return loss


def compute_combined_loss(fault_free_output, faulty_outputs, alpha=0.7, beta=0.3):
    """
    Computes the total loss combining MSE difference and classification coverage.
    Args:
        fault_free_output: Output of the fault-free model.
        faulty_outputs: List of outputs from the faulty models.
        alpha: Weight for MSE loss component.
        beta: Weight for classification coverage loss component.
    """
    total_loss = 0.0

    for faulty_output in faulty_outputs:
        # Difference Metric (MSE Loss)
        difference_loss = torch.nn.functional.mse_loss(
            faulty_output, fault_free_output)

        # Classification-Based Coverage Metric
        classification_loss = compute_classification_loss(
            faulty_output, fault_free_output)

        # Combined Loss (Weighted Sum)
        total_loss += alpha * difference_loss + beta * \
            classification_loss  # Subtract coverage loss to maximize it

    return - total_loss


def compute_kl_loss(faulty_output, fault_free_output):
    # Compute probabilities
    faulty_probs = torch.nn.functional.softmax(faulty_output, dim=1)
    fault_free_probs = torch.nn.functional.softmax(fault_free_output, dim=1)

    # Compute log probabilities
    fault_free_log_probs = torch.log(fault_free_probs + 1e-10)

    # Compute KL divergence
    kl_loss = torch.nn.functional.kl_div(
        input=fault_free_log_probs,
        target=faulty_probs,
        reduction='batchmean'
    )

    return kl_loss


def compute_symmetric_kl_loss(faulty_output, fault_free_output):
    # Compute probabilities
    faulty_probs = torch.nn.functional.softmax(faulty_output, dim=1)
    fault_free_probs = torch.nn.functional.softmax(fault_free_output, dim=1)

    # Compute log probabilities
    faulty_log_probs = torch.log(faulty_probs + 1e-10)
    fault_free_log_probs = torch.log(fault_free_probs + 1e-10)

    # Compute KL divergences
    kl_loss_1 = torch.nn.functional.kl_div(
        fault_free_log_probs, faulty_probs, reduction='batchmean')
    kl_loss_2 = torch.nn.functional.kl_div(
        faulty_log_probs, fault_free_probs, reduction='batchmean')

    # Symmetric KL divergence
    symmetric_kl_loss = kl_loss_1 + kl_loss_2

    return symmetric_kl_loss


# ### Metrics


def compute_difference_metric(faulty_output, fault_free_output):
    # Compute MSE difference
    mse_differece = torch.nn.functional.mse_loss(
        faulty_output, fault_free_output)

    faulty_class = torch.argmax(torch.nn.functional.softmax(
        faulty_output.detach(), dim=1)).numpy()  # Convert to numpy for readability
    fault_free_class = torch.argmax(torch.nn.functional.softmax(
        fault_free_output.detach(), dim=1)).numpy()  # Convert to numpy for readability
    equal = int(faulty_class != fault_free_class)

    return mse_differece, equal


# ## EA


# ### Initialize the Population as inital test inputs


# import torch
# import random

# # act_list_fault_temp = act_list_fault_sample[:20]
# act_list_fault_temp = active_faults

# # Evolutionary algorithm parameters
# population_size = len(act_list_fault_temp)+20
# num_generations = 50
# mutation_rate = 0.5
# crossover_rate = 0.7
# test_input_dim = topology[0]  # Dimensionality of the test input
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Weights for fitness function
# alpha = 0.5  # Weight for difference metric
# beta = 0.5   # Weight for coverage metric

# # Initialize population
# population = [torch.randn(test_input_dim, device=device, requires_grad=False) for _ in range(population_size)]

# # Fitness function (unchanged from previous implementation)
# def evaluate_fitness(test_input, pnn, act_list_fault_temp):
#     fault_free_output = pnn(test_input)[0, 0, :, :]
#     # fault_free_probs = torch.softmax(fault_free_output, dim=0)

#     faulty_outputs = [pnn(test_input, *fault)[0, 0, :, :] for fault in act_list_fault_temp]
#     # faulty_probs_list = [torch.softmax(output, dim=0) for output in faulty_outputs]

#     differences = [compute_difference_metric(faulty_output, fault_free_output)[0] for faulty_output in faulty_outputs]
#     difference_metric = sum(differences)

#     coverages = [compute_difference_metric(faulty_output, fault_free_output)[1] for faulty_output in faulty_outputs]
#     coverage_metric = sum(coverages)

#     # coverage_metric = sum(
#     #     1 for faulty_probs in faulty_probs_list
#     #     if torch.argmax(fault_free_probs) != torch.argmax(faulty_probs)
#     # )

#     fitness = alpha * difference_metric + beta * coverage_metric
#     return fitness, coverage_metric

# # Initialize population (constrained to [0, 1])
# population = [torch.rand(test_input_dim, device=device) for _ in range(population_size)]

# # Selection (Tournament Selection)
# def select_parents(population, fitnesses, tournament_size=3):
#     """Select parents using tournament selection."""
#     selected = []
#     for _ in range(2):  # Select two parents
#         tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
#         winner = max(tournament, key=lambda x: x[1])  # Select the one with the highest fitness
#         selected.append(winner[0])
#     return selected

# # Crossover with Range Constraint
# def crossover(parent1, parent2, crossover_rate=0.7):
#     """Perform crossover between two parents and clamp offspring to [0, 1]."""
#     if random.random() < crossover_rate:
#         point = random.randint(1, test_input_dim - 1)
#         offspring1 = torch.cat([parent1[:point], parent2[point:]])
#         offspring2 = torch.cat([parent2[:point], parent1[point:]])
#         return offspring1.clamp(0, 1), offspring2.clamp(0, 1)
#     return parent1.clone(), parent2.clone()

# # Mutation with Clamping
# def mutate(individual, mutation_rate=0.1):
#     """Apply mutations and clamp individual to [0, 1]."""
#     if random.random() < mutation_rate:
#         mutation = torch.randn_like(individual) * 0.1  # Scale mutation strength
#         individual += mutation
#         individual = individual.clamp(0, 1)
#     return individual

# # num_elites = 2  # Number of elite individuals to retain
# # Evolutionary Algorithm
# for generation in range(num_generations):
#     fitnesses = []
#     coverage_metrics = []
#     for individual in population:
#         fitness, coverage = evaluate_fitness(individual, pnn, act_list_fault_temp)
#         fitnesses.append(fitness)
#         coverage_metrics.append(coverage)

#     best_fitness = max(fitnesses)
#     best_individual = population[fitnesses.index(best_fitness)]
#     print(f"Generation {generation}, Best Fitness: {best_fitness}, Coverage: {max(coverage_metrics)}")

#     # # Retain elites
#     # elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:num_elites]
#     # elite_individuals = [elite[0] for elite in elites]

#     # Selection and Reproduction
#     new_population = []
#     for _ in range((population_size) // 2):  # Each loop creates two offspring
#         parent1, parent2 = select_parents(population, fitnesses)
#         offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
#         offspring1 = mutate(offspring1, mutation_rate)
#         offspring2 = mutate(offspring2, mutation_rate)
#         new_population.extend([offspring1, offspring2])

#     # Add elites back into the population
#     # population = new_population + elite_individuals

#     # Replace population with new individuals
#     population = new_population

# # Output the best test input pool
# optimized_test_input_pool = population
# print("Optimized Test Input Pool:", optimized_test_input_pool)


# ### Considering Test Compaction


# device = 'cpu'
# covered_faults = set()
# uncovered_faults = set()
# # act_list_fault_temp = act_list_fault_sample[:20]
# act_list_fault_temp = active_faults

# # act_test_input_pool = torch.randn((len(act_list_fault_temp), topology[0]), requires_grad=False).to(device)
# act_test_input_pool = population

# # Batch process all fault-free outputs
# # output_fault_free_list = pnn(act_test_input_pool)[0, 0, :, :]

# multi_fault = True
# while (len(covered_faults)+len(uncovered_faults)) != len(act_list_fault_temp):

#     print(f"Covered Faults: {covered_faults}")
#     test_input_metrics = []
#     for test_input_i, test_input in enumerate(act_test_input_pool):
#         # output_fault_free = output_fault_free_list[test_input_i: test_input_i+1]
#         output_fault_free = pnn(test_input)[0, 0, :, :]
#         output_faulty_list = [pnn(test_input,*fault)[0, 0, :, :] for fault in act_list_fault_temp]

#         differences = [compute_difference_metric(output, output_fault_free)[0] if (i_out not in covered_faults) else None for i_out, output in enumerate(output_faulty_list)]
#         equals = [compute_difference_metric(output, output_fault_free)[1] if (i_out not in covered_faults) else 0 for i_out, output in enumerate(output_faulty_list)]

#         # print('differences: ', differences)
#         # print('equals: ', equals)

#         total_difference = sum(diff for diff in differences if diff is not None)
#         total_equal = sum(equals)
#         indices = (torch.nonzero(torch.tensor(equals) == 1).squeeze()).tolist()
#         # print(f"indices: {indices}")
#             # coverage_count = sum(1 for diff in differences if diff > threshold)
#         test_input_metrics.append((test_input, total_difference, total_equal, indices))

#     # Select the best test input using heapq for efficiency
#     best_test_input_data = heapq.nlargest(1, test_input_metrics, key=lambda x: (x[2], x[1]))[0]
#     best_test_input, selected_faults = best_test_input_data[0], best_test_input_data[3]
#     multi_fault = True if len(selected_faults) > 1 else False

#     if len(selected_faults) == 0:
#         for i_fault, fault in enumerate(act_list_fault_temp):
#             if i_fault not in covered_faults:
#                 diff = -float('inf')
#                 best_equal = 0
#                 for test_input_i, test_input in enumerate(population):
#                     # output_fault_free = output_fault_free_list[test_input_i: test_input_i+1]
#                     output_fault_free = pnn(test_input)[0, 0, :, :]
#                     output_faulty = pnn(test_input,*fault)[0, 0, :, :]
#                     differences = compute_difference_metric(output_faulty, output_fault_free)[0]
#                     equals = compute_difference_metric(output_faulty, output_fault_free)[1]
#                     if best_equal < equals:
#                         best_equal = equals
#                         diff = differences
#                         best_test_input = test_input
#                     if differences > diff and best_equal == equals:
#                         diff = differences
#                         best_test_input = test_input
#                     print(f"fault: {i_fault}, differences: {differences}, equals: {equals}")
#                 if diff==0.0:
#                     uncovered_faults.add(i_fault)
#                 else:
#                     selected_faults = [i_fault]
#                     break
#     print(f"selected_faults", selected_faults)

#     if len(selected_faults) == 0:
#         break

#     # Optimize the best test input ----------------------------------------------
#     # Define fault-free and faulty models
#     ste_module = STEMultiClass(best_test_input, min_value=0.0, max_value=1.0)

#     # Define the optimizer for input optimization
#    # Initialize optimizer and scheduler
#     optimizer = optim.Adam([ste_module.test_input_], lr=0.001)
#     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#     num_epochs = 100
#     best_loss = float('inf')
#     patience = 10
#     # counter = 0
#     min_delta = 1e-6

#     # Regularization coefficient
#     regularization_coeff = 1e-4

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         faulty_output_list, fault_free_output = ste_module(pnn, act_list_fault_temp, selected_faults)
#         # Compute primary loss
#         # primary_loss = compute_total_loss(fault_free_output, faulty_output_list)

#         # Add regularization term to the loss
#         # reg_term = regularization_coeff * torch.norm(ste_module.test_input_)
#         # loss = primary_loss + reg_term
#         # loss = compute_total_loss(fault_free_output, faulty_output_list)
#         loss = compute_combined_loss(fault_free_output, faulty_output_list, alpha=0.2, beta=0.8)
#         if len(selected_faults) == 1:
#             print('fault free output: ', fault_free_output, faulty_output_list)
#         # assert ste_module.test_input_.requires_grad, "The gradient is not being tracked for the test input."
#         loss.backward()
#         optimizer.step()

#         # Adjust learning rate based on loss
#         # scheduler.step(loss.item())

#         if loss.item() < best_loss - min_delta:
#             best_loss = loss.item()
#             counter = 0
#         else:
#             counter += 1

#         if counter >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item()}")

#     # Checking fault-free and faulty outputs with the optimized test input

#     # Ensure the models are in evaluation mode
#     ste_module.eval()

#     optimized_test_input = ste_module.test_input
#     output_fault_free = pnn(optimized_test_input)[0, 0, :, :]
#     output_faulty_list = [pnn(optimized_test_input,*act_list_fault_temp[i])[0, 0, :, :] for i in selected_faults]

#     print("Fault-Free Output:", torch.argmax(nn.functional.softmax(output_fault_free.detach(), dim=1)).numpy())  # Convert to numpy for readability
#     for output in output_faulty_list:
#         print("Faulty Output:", torch.argmax(nn.functional.softmax(output.detach(), dim=1)).numpy())  # Convert to numpy for readability


#     differences = [compute_difference_metric(output, output_fault_free)[0] for output in output_faulty_list]
#     equals_i = [fault_i for fault_i, output in zip(selected_faults, output_faulty_list) if compute_difference_metric(output, output_fault_free)[1] == 1]
#     total_difference = sum(differences)
#     if len(equals_i) == 0 and multi_fault:
#         # save the initial test input
#         covered_faults.update(selected_faults)
#     elif len(equals_i) < len(selected_faults) and multi_fault:
#         # save the optimized test input
#         covered_faults.update(selected_faults)
#     else:
#         covered_faults.update(equals_i)


# ## Gradient Algorithm


# ### Remove Inactive Faults


def extract_active_faults(cluster_list_fault_sample, batch_size=100):
    threshold = 0  # Define a threshold for significant output differences
    active_cluster_list_fault_sample = []
    test_input_dim = topology[0]

    # Generate a batch of random test inputs (all at once for efficiency)
    # Shape: (batch_size, input_dim)
    test_inputs = torch.rand(batch_size, test_input_dim)

    # Process faults in batches to avoid Python loops
    for fault in cluster_list_fault_sample:
        # Compute fault-free and faulty outputs for all test inputs
        output_fault_free = pnn(test_inputs)[0, 0, :, :]  # Fault-free outputs
        output_faulty = pnn(test_inputs, *fault)[0, 0, :, :]  # Faulty outputs

        # Compute differences using MSE loss for all test inputs in the batch
        differences = torch.nn.functional.mse_loss(
            output_faulty, output_fault_free, reduction="none").mean(dim=1)

        # Check if the average difference exceeds the threshold
        if differences.mean().item() > threshold:
            active_cluster_list_fault_sample.append(fault)

    return active_cluster_list_fault_sample


active_cluster_act_list_fault_sample = extract_active_faults(
    cluster_act_list_fault_sample)
active_cluster_inv_list_fault_sample = extract_active_faults(
    cluster_inv_list_fault_sample)
active_cluster_theta_list_fault_sample = extract_active_faults(
    theta_list_fault_sample)


len(cluster_act_list_fault_sample), len(active_cluster_act_list_fault_sample), len(cluster_inv_list_fault_sample), len(
    active_cluster_inv_list_fault_sample), len(theta_list_fault_sample), len(active_cluster_theta_list_fault_sample)


active_list_fault_sample = active_cluster_act_list_fault_sample + \
    active_cluster_inv_list_fault_sample + active_cluster_theta_list_fault_sample
len(active_list_fault_sample)


# Save the fault counts to a text file
with open(f"../gradient_results/fault_counts_{args.DATASET}.txt", "w") as file:
    file.write(f"Active Faults: {len(active_list_fault_sample)}\n")
    file.write(f"Clustered Faults: {len(
        cluster_act_list_fault_sample + cluster_inv_list_fault_sample + theta_list_fault_sample)}\n")
    file.write(f"All Faults: {len(all_act_list_fault_sample +
               all_inv_list_fault_sample + theta_list_fault_sample)}\n")


# ### Random Initialization


# Finding the Best Test Input for Each Fault


random_search_covered_faults = set()
random_search_test_input = torch.zeros(
    (len(active_list_fault_sample), 1, test_input_dim), requires_grad=False)

# Iterate over all faults
start_time = time.time()
for fault_idx, fault in enumerate(active_list_fault_sample):
    # Generate a random pool of test inputs
    # Pool size set to 100 for efficiency
    test_input_pool = torch.rand((100, 1, test_input_dim), requires_grad=False)

    # Compute fault-free and faulty outputs for all test inputs
    output_fault_free = pnn(test_input_pool[:, 0, :])[
        0, 0, :, :]  # Fault-free output
    output_faulty = pnn(
        test_input_pool[:, 0, :], *fault)[0, 0, :, :]  # Faulty output

    # Apply softmax to convert logits to probabilities
    # Shape: (batch_size, num_classes)
    output_fault_free_probs = F.softmax(output_fault_free, dim=1)
    output_faulty_probs = F.softmax(output_faulty, dim=1)

    # Compute Cross-Entropy Loss
    # Comparing fault-free probabilities as the target to the faulty probabilities
    # 'none' computes loss for each input individually
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_per_test_input = criterion(
        # Shape: (batch_size,)
        output_faulty_probs, output_fault_free_probs.argmax(dim=1))

    # Identify the best test input (highest loss)
    best_test_input_index = torch.argmax(loss_per_test_input).item()
    best_test_input = test_input_pool[best_test_input_index]

    # Evaluate the class predictions for the fault-free and faulty outputs
    fault_free_class = output_fault_free_probs[best_test_input_index].argmax(
    ).item()
    faulty_class = output_faulty_probs[best_test_input_index].argmax().item()

    # Check if the fault causes a class mismatch
    if fault_free_class != faulty_class:
        random_search_covered_faults.add(fault_idx)
        # Save the test input that covers this fault
        random_search_test_input[fault_idx] = best_test_input

end_time = time.time()
print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")
print("Covered Faults:", len(random_search_covered_faults))
print("Test Coverage:", len(random_search_covered_faults) /
      len(active_list_fault_sample) * 100)

with open(f"../gradient_results/{args.DATASET}_rs.txt", "w") as file:
    file.write(f"Covered Faults: {len(random_search_covered_faults)}\n")
    file.write(f"Test Coverage: {len(
        random_search_covered_faults) / len(active_list_fault_sample) * 100:.2f}%\n")
    file.write(f"Time taken: {end_time - start_time:.2f} seconds\n")


# ### Optimize Test Input


list_fault_temp = active_list_fault_sample

coverage_faults = set()
test_input_list = torch.zeros(
    (len(list_fault_temp), 1, test_input_dim), requires_grad=False)
test_coverage = 0

patience_outer = 100
no_improve_counter_outer = 0

start_time = time.time()

while no_improve_counter_outer < patience_outer:

    # Iterate over all faults
    for i_fault in range(len(list_fault_temp)):

        if i_fault in coverage_faults:
            continue

        # Generate a random pool of test inputs
        test_input_pool = torch.rand(
            (100, 1, test_input_dim), requires_grad=False)  # Pool size set to 100

        # Define CrossEntropyLoss
        # Compute loss for each input in the pool
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # Compute fault-free and faulty outputs for all test inputs
        output_fault_free = pnn(test_input_pool[:, 0, :])[
            0, 0, :, :]  # Fault-free output
        output_faulty = pnn(
            # Faulty output
            test_input_pool[:, 0, :], *list_fault_temp[i_fault])[0, 0, :, :]

        # Apply softmax to convert logits to probabilities
        output_fault_free_probs = F.softmax(output_fault_free, dim=1)
        output_faulty_probs = F.softmax(output_faulty, dim=1)

        # Compute Cross-Entropy Loss
        loss_per_test_input = criterion(
            # Loss for each input
            output_faulty_probs, output_fault_free_probs.argmax(dim=1))

        # Identify the best test input (highest loss)
        best_test_input_index = torch.argmax(loss_per_test_input).item()

        # Initialize test input and optimizer
        test_input = test_input_pool[best_test_input_index].clone(
        ).detach().requires_grad_(True)
        optimizer = optim.Adam([test_input], lr=0.01)
        # KL divergence expects log probabilities
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

        num_epochs = 200
        early_stopping_patience = 50
        best_loss = float('inf')
        no_improvement_epochs = 0  # Counter for early stopping

        # Training loop with early stopping
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass through fault-free and faulty models
            fault_free_output = pnn(test_input)[0, 0, :, :]  # Fault-free model
            faulty_output = pnn(
                # Faulty model
                test_input, *list_fault_temp[i_fault])[0, 0, :, :]

            # Apply softmax and log for KL divergence
            faulty_probs = torch.log_softmax(
                faulty_output, dim=1)  # Log probabilities
            fault_free_probs = torch.softmax(
                fault_free_output, dim=1)  # Probabilities

            # Compute KL divergence loss
            # Negate to maximize difference
            loss_value = -loss_fn(faulty_probs, fault_free_probs)

            # Backward pass and optimization
            loss_value.backward()
            optimizer.step()

            # Clamp test input to valid range [0, 1]
            test_input.data.clamp_(0, 1)

            # Evaluate progress
            faulty_class = torch.argmax(torch.softmax(
                faulty_output, dim=1).detach(), dim=1).item()
            fault_free_class = torch.argmax(
                fault_free_probs.detach(), dim=1).item()

            if loss_value.item() < best_loss:  # Improvement threshold
                best_loss = loss_value.item()
                no_improvement_epochs = 0  # Reset early stopping counter
                if faulty_class != fault_free_class:
                    test_input_list[i_fault] = test_input.clone().detach()
                    coverage_faults.add(i_fault)
            else:
                no_improvement_epochs += 1

            # Early stopping check
            if no_improvement_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} for fault {i_fault}")
                break

            # # Print progress every 10 epochs
            # if epoch % 50 == 0:
            #     print(f"Fault: {i_fault}, Epoch: {epoch}")
            #     print("Fault-Free Output:", fault_free_class)
            #     print("Faulty Output:", faulty_class)
    temp_test_coverage = len(coverage_faults) / len(list_fault_temp)
    print(f"Test Coverage: {test_coverage * 100:.2f}%")
    if temp_test_coverage == test_coverage:
        no_improve_counter_outer += 1
    else:
        no_improve_counter_outer = 0
        test_coverage = temp_test_coverage
end_time = time.time()
print(f"Total Execution Time: {(end_time - start_time) / 60:.2f} minutes")

# Print final results
print("Covered Faults:", coverage_faults)


# check if all values are positive
all(v >= 0 for v in test_input_list.flatten())


len(coverage_faults)


len(coverage_faults) / len(list_fault_temp) * 100

with open(f"../gradient_results/{args.DATASET}_gd.txt", "w") as file:
    file.write(f"Covered Faults: {len(coverage_faults)}\n")
    file.write(f"Test Coverage: {
               len(coverage_faults) / len(list_fault_temp) * 100:.2f}%\n")
    file.write(f"Total Execution Time: {
               (end_time - start_time) / 60:.2f} minutes\n")


# Separate indices for each fault type
num_act_faults = len(active_cluster_act_list_fault_sample)
num_inv_faults = len(active_cluster_inv_list_fault_sample)
num_theta_faults = len(active_cluster_theta_list_fault_sample)

# Calculate start and end indices for each fault type in active_list_fault_sample
act_fault_indices = set(range(num_act_faults))
inv_fault_indices = set(range(num_act_faults, num_act_faults + num_inv_faults))
theta_fault_indices = set(
    range(num_act_faults + num_inv_faults, len(active_list_fault_sample)))

# Compute coverage for each fault type
covered_act_faults = act_fault_indices & coverage_faults
covered_inv_faults = inv_fault_indices & coverage_faults
covered_theta_faults = theta_fault_indices & coverage_faults

# Calculate coverage metrics
act_coverage = len(covered_act_faults) / \
    num_act_faults if num_act_faults > 0 else 0
inv_coverage = len(covered_inv_faults) / \
    num_inv_faults if num_inv_faults > 0 else 0
theta_coverage = len(covered_theta_faults) / \
    num_theta_faults if num_theta_faults > 0 else 0

# Print results
print("Test Coverage Metrics:")
print(f"Total ACT Faults: {num_act_faults}, Covered: {
      len(covered_act_faults)}, Coverage: {act_coverage:.2%}")
print(f"Total INV Faults: {num_inv_faults}, Covered: {
      len(covered_inv_faults)}, Coverage: {inv_coverage:.2%}")
print(f"Total THETA Faults: {num_theta_faults}, Covered: {
      len(covered_theta_faults)}, Coverage: {theta_coverage:.2%}")


covered_act_faults = len(covered_act_faults)
covered_inv_faults = len(covered_inv_faults)
covered_theta_faults = len(covered_theta_faults)

# Calculate uncovered faults for each type
uncovered_act_faults = num_act_faults - covered_act_faults
uncovered_inv_faults = num_inv_faults - covered_inv_faults
uncovered_theta_faults = num_theta_faults - covered_theta_faults

# Data for the pie chart
sizes = [
    covered_act_faults, uncovered_act_faults,
    covered_inv_faults, uncovered_inv_faults,
    covered_theta_faults, uncovered_theta_faults
]

labels = [
    "p-tanh   Covered", "p-tanh Uncovered",
    "Inverter Covered", "Inverter Uncovered",
    "crossbar Covered", "crossbar Uncovered"
]

# Define main colors and hover colors
colors = [
    '#1f77b4', '#aec7e8',  # ACT main and hover (blue shades)
    '#ff7f0e', '#ffbb78',  # INV main and hover (orange shades)
    '#2ca02c', '#98df8a'   # THETA main and hover (green shades)
]


# Create the pie chart
fig, ax = plt.subplots(figsize=(10, 10))
wedges, texts, autotexts = ax.pie(
    sizes, labels=None, autopct='%1.1f%%', startangle=140, colors=colors,
    # Adjust width to create a donut
    pctdistance=0.85, wedgeprops=dict(width=0.3)
)

# Add legend inside the ring
ax.legend(wedges, labels, loc="center", fontsize=16,
          title_fontsize=18, frameon=False,)

# Add title and customize font
ax.set_title("Test Coverage Distribution Across Fault Types",
             fontsize=16, fontweight='bold')

# Enhance label visibility
for text in texts:
    text.set_fontsize(16)
for autotext in autotexts:
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')

# Layout adjustments
plt.tight_layout()
plt.show()
# Save the pie chart as an image file
plt.savefig(f"../gradient_results/fault_per_type_{args.DATASET}.png")


# # ### Evalute on Optimized Test Input


# results = []  # List to store evaluation results
# covered_faults = set()  # Track covered faults during evaluation
# type_fault = "ALL"  # Replace with the actual fault type

# # Loop over all saved test inputs
# for i_fault, best_test_input in enumerate(test_input_list):
#     if i_fault not in coverage_faults:  # Skip faults not covered during training
#         continue

#     print(f"Re-evaluating Fault: {i_fault}")
#     # Ensure the input has the correct batch dimension
#     best_test_input = best_test_input.unsqueeze(0)

#     with torch.no_grad():  # Disable gradient computation for evaluation
#         # Forward pass for fault-free and faulty models
#         fault_free_output = pnn(best_test_input)[0, 0, :, :]
#         faulty_output = pnn(best_test_input, *
#                             list_fault_temp[i_fault])[0, 0, :, :]

#         # Compute probabilities
#         fault_free_probs = F.softmax(fault_free_output, dim=1)
#         faulty_probs = F.softmax(faulty_output, dim=1)

#         # Compute classification results
#         fault_free_class = torch.argmax(fault_free_probs, dim=1).item()
#         faulty_class = torch.argmax(faulty_probs, dim=1).item()

#         # Check if the test input distinguishes the fault-free and faulty models
#         if fault_free_class != faulty_class:
#             covered_faults.add(i_fault)  # Add to covered faults set

#         # Save result for this fault
#         results.append({
#             "fault_index": i_fault,
#             # Convert tensor to list
#             "test_input": best_test_input.squeeze(0).tolist(),
#             "fault_free_class": fault_free_class,
#             "faulty_class": faulty_class,
#             # True if covered, False otherwise
#             "is_covered": fault_free_class != faulty_class,
#             "num_epochs": num_epochs,
#         })

#         # Print progress
#         print(
#             f"Fault-Free Class: {fault_free_class}, Faulty Class: {faulty_class}")

# # Save results and covered faults to files
# dataset_name = args.DATASET  # Replace with the actual dataset name
# results_file = f"../gradient_results/{type_fault}_{
#     dataset_name}_evaluation_results.json"
# covered_faults_file = f"../gradient_results/{
#     type_fault}_{dataset_name}_covered_faults.json"

# # Save detailed results
# with open(results_file, "w") as f:
#     json.dump(results, f, indent=4)

# # Save covered faults separately
# with open(covered_faults_file, "w") as f:
#     json.dump(list(covered_faults), f, indent=4)

# print(f"Results saved to {results_file}")
# print(f"Covered faults saved to {covered_faults_file}")
# print(f"Total Covered Faults: {len(covered_faults)}/{len(list_fault_temp)}")


# # File paths
# dataset_name = args.DATASET  # Replace with your dataset name, e.g., "CIFAR10"
# results_file = f"../gradient_results/{type_fault}_{
#     dataset_name}_evaluation_results.json"
# covered_faults_file = f"../gradient_results/{
#     type_fault}_{dataset_name}_covered_faults.json"

# # Load evaluation results
# with open(results_file, "r") as f:
#     evaluation_results = json.load(f)

# # Display some results
# print(f"Loaded {len(evaluation_results)} evaluation results.")
# for result in evaluation_results[:5]:  # Display the first 5 results
#     print(f"Fault Index: {result['fault_index']}")
#     # Show only the first 10 values
#     print(f"Test Input: {result['test_input'][:10]}...")
#     print(f"Fault-Free Class: {result['fault_free_class']}")
#     print(f"Faulty Class: {result['faulty_class']}")
#     print(f"Is Covered: {result['is_covered']}")
#     print(f"Num Epochs: {result['num_epochs']}")
#     print()
