# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from utils.average_weights import average_weights
from utils.average_weights import average_weights_edge
from fednn.intialize_model import initialize_model
import torch
from utils.quantization import quantization_nne
import logging

def cast_to_range(values, scale):
    return torch.round(values * scale).to(torch.long) 

def uncast_from_range(scaled_values, scale):
    return scaled_values / scale

class Edge():

    def __init__(self, id, cids, shared_layers, args):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.update_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.model = shared_layers
        # self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.args = args
        self.G = {}
        self.cos_client_ref = {}
        self.args = args

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, message):
        self.receiver_buffer[client_id] = message
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        client_ids = [key for key in self.receiver_buffer.keys()]
        # self.update_state_dict = average_weights_edge(w = received_dict,
        #                                          s_num= sample_num,
        #                                          client_learning_rate = [args.client_learning_rate[client_id] for client_id in client_ids],
        #                                          edge_learning_rate = args.edge_learning_rate[self.id]
        #                                          )
        received_dict1 = [dict['cshared_state_dict1'] for dict in self.receiver_buffer.values()]
        received_dict2 = [dict['cshared_state_dict2'] for dict in self.receiver_buffer.values()]
        self.update_state_dict1 = average_weights(w = received_dict1, s_num= sample_num, args = args)
        self.update_state_dict2 = average_weights(w = received_dict2, s_num= sample_num, args = args)
        del received_dict1
        del received_dict2
        # sd = self.model.state_dict()
        # for key in sd.keys():
        #     sd[key]= torch.add(self.model.state_dict()[key], self.update_state_dict[key])
        # self.model.load_state_dict(sd)
        # logging.info('edge after update')
        # logging.info(self.model.state_dict()['stem.0.conv.weight'])
        def flatten_list_of_tensors(tensors):
            flattened_tensors = []
            for tensor in tensors:
                flattened_tensor = tensor.view(-1)  # Flattening the tensor
                flattened_tensors.append(flattened_tensor)
            return flattened_tensors
        
        received_dict = {client_id: dict for client_id, dict in zip(client_ids, received_dict)}
        for client_id in received_dict:
            if args.model == 'lenet' or args.model == 'linear':
                last_layer = torch.flatten(received_dict[client_id]['cshared_state_dict1']['fc2.weight'])
                # last_gamma = torch.flatten(args.gamma[client_id].state_dict()['fc2.weight'])
            elif args.model == 'cnn_complex':
                last_layer = torch.flatten(received_dict[client_id]['cshared_state_dict1']['fc_layer.6.weight'])
                # last_gamma = torch.flatten(args.gamma[client_id].state_dict()['fc_layer.6.weight'])
            elif args.model == 'resnet18':
                last_layer = torch.flatten(received_dict[client_id]['cshared_state_dict1']['linear.weight'])
                # last_gamma = torch.flatten(args.gamma[client_id].state_dict()['linear.weight'])
            # last_layer = last_layer / args.a[client_id]
            # last_layer = uncast_from_range(last_layer, args.g)
            # if torch.linalg.norm(last_layer) > 1:
                # last_layer /= torch.linalg.norm(last_layer) 
            matmul = args.reference.to('cpu').matmul(last_layer.to('cpu'))
            matmul = matmul % args.p
            args.cos_client_ref[client_id] = matmul.to('cuda:0')
            # args.cos_client_ref[client_id] = args.reference.matmul(last_layer)
            # if client_id == 5:
                # logging.info(args.cos_client_ref[client_id])
           

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.model.state_dict()))
        return None

    def send_to_cloudserver(self, cloud, compression_ratio, q_method):
        # for key in self.update_state_dict.keys():
        #     self.update_state_dict[key] = torch.add(self.model.state_dict()[key],-cloud.model.state_dict()[key])
        # self.model.load_state_dict(self.update_state_dict)
        # Now we decomment the random sparsification first
        # quantization_nne(self.model, compression_ratio, q_method)
        message = {'update_state_dict1': self.update_state_dict1, 'update_state_dict2': self.update_state_dict2}
        cloud.receive_from_edge(edge_id=self.id,
                                message = message)
        del self.update_state_dict1
        del self.update_state_dict2
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.model.load_state_dict(shared_state_dict)
        return None
