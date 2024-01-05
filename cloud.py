# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
from utils.average_weights import average_weights
from utils.average_weights import average_weights_cloud
from utils.contra import contra
import torch
import logging

def cast_to_range(values, scale):
    return torch.round(values * scale).to(torch.long) 

def uncast_from_range(scaled_values, scale):
    return scaled_values / scale

class Cloud():

    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.model = shared_layers
        self.update_state_dict = shared_layers.state_dict()
        self.clock = []

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, message):
        self.receiver_buffer[edge_id] = message
        return None

    def aggregate(self, args):
        """
        I think the problem may lie in this...
        :param args:
        :return:
        """
        # logging.info('Average Old')
        # first make the state_dict and sample into num
        # The following code may cause some problem? I am not sure whether values keeps the values int the original order
        # But when the data sample number is the same,  this is not a problem
        # received_dict = [dict for dict in self.receiver_buffer.values()]
        if args.edge_average_uniform:
            sample_num = [1]*args.num_edges
        else:
            sample_num = [snum for snum in self.sample_registration.values()]
        edge_ids = [key for key in self.receiver_buffer.keys()]
        # self.update_state_dict = average_weights_cloud(w=received_dict,
        #                                          s_num=sample_num,
        #                                          edge_learning_rate = [args.edge_learning_rate[edge_id] for edge_id in edge_ids]
        #                                          )
        received_dict1 = [dict['update_state_dict1'] for dict in self.receiver_buffer.values()]
        update_state_dict1 = average_weights(w=received_dict1, s_num=sample_num, args = args)
        received_dict2 = [dict['update_state_dict2'] for dict in self.receiver_buffer.values()]
        update_state_dict2 = average_weights(w=received_dict2, s_num=sample_num, args= args)
        # gamma_sum = {}
        # for key in args.gamma[0].state_dict():
        #     gamma_sum[key] = torch.sum(torch.stack([args.gamma[i].state_dict()[key] for i in range(args.num_clients)]), dim = 0)
        # xi_sum = {}
        # for key in args.xi[0].state_dict():
        #     xi_sum[key] = torch.sum(torch.stack([args.xi[i].state_dict()[key] for i in range(args.num_clients)]), dim = 0)
        self.update_state_dict = {}
        for key in update_state_dict1:
            self.update_state_dict[key] = update_state_dict1[key] * args.c[0] % args.p + update_state_dict2[key] * args.c[1] % args.p 
            self.update_state_dict[key] %= args.p
            self.update_state_dict[key][self.update_state_dict[key] > args.g * args.w] -= args.p
            self.update_state_dict[key] = uncast_from_range(self.update_state_dict[key], args.g * args.w)
        sd = self.model.state_dict()
        for key in sd.keys():
            sd[key] = torch.add(self.model.state_dict()[key], self.update_state_dict[key])
        self.model.load_state_dict(sd)
        # logging.info('cloud after update')
        # logging.info(self.model.state_dict()['stem.0.conv.weight'])
        # exit()
        del received_dict1
        del received_dict2
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.model.state_dict()))
        return None
