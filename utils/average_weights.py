import copy
import torch

def average_weights(w, s_num, args):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            # w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i]/temp_sample_num)
            w_avg[k] = w_avg[k] % args.p + w[i][k] 
            w_avg[k] %= args.p
        # w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg

def average_weights_edge(w, s_num, client_learning_rate, edge_learning_rate):
    #copy the first client's weights
    client_learning_rate = [item / edge_learning_rate for item in client_learning_rate]
    # total_sample_num = sum(s_num)
    # temp_sample_num = s_num[0]
    w_avg = {key: val * client_learning_rate[0] for key, val in copy.deepcopy(w[0]).items()}
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], client_learning_rate[i])
    return w_avg


def average_weights_cloud(w, s_num, edge_learning_rate):
    #copy the first client's weights
    # total_sample_num = sum(s_num)
    # temp_sample_num = s_num[0]
    w_avg = {key: val * edge_learning_rate[0] for key, val in copy.deepcopy(w[0]).items()} 
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], edge_learning_rate[i])
    return w_avg
