import torch


def get_act_list_fault(topology, type_fault=1, num_fault_sub_type=[]):
    # type of fault: 0 - theta, 1 - act, 2 - neg
    list_sample_fault = []
    for layer_i, num_neuron in enumerate(topology[1:]):
        # list of faulty layers starting from 0
        faulty_layer_list = [layer_i]
        for neuron_i in range(num_neuron):
            # index of element within the layer
            indice_to_modify = neuron_i
            for sub_fault_i in num_fault_sub_type:
                # type of fault within the non-linear circuit
                fault_type_non_linear = sub_fault_i
                sample_fault = (faulty_layer_list, type_fault,
                                indice_to_modify, fault_type_non_linear)
                list_sample_fault.append(sample_fault)

    return list_sample_fault
