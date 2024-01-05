import torch
import random
def get_reference(self):
        self.parameter_count = self.init_state.size()[0]
        self.parameter_count = int(self.parameter_count)
        if self.num_reference == 0:
            reference = torch.eye(self.parameter_count, device=self.device)
            return reference
        nonzero_per_reference =  self.parameter_count // self.num_reference
        reference = torch.zeros((self.num_reference,  self.parameter_count), device=self.device)
        parameter_index_random = list(range( self.parameter_count))
        random.shuffle(parameter_index_random)

        for reference_index in range(self.num_reference):
            index = parameter_index_random[reference_index * nonzero_per_reference: (reference_index + 1) * nonzero_per_reference]
            index = torch.tensor(index)
            reference[reference_index][index] = 1
        return reference
    