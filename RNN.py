import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNCell(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, bias:bool=1, nonlinearity:str='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weights_hidden = torch.zeros(hidden_size, input_size+hidden_size)
        self.bias_hidden = torch.zeros(hidden_size, 1)
        self.weights_output = torch.zeros(output_size, hidden_size)
        if nonlinearity != 'tanh':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Tanh()
        self.output_activation = torch.nn.Softmax()
        
    
    def forward(self, input_tensor, hidden_tensor):
        print(input_tensor.shape)
        input_tensor = input_tensor.view(input_tensor.shape[0], 1)
        print(input_tensor.shape)
        hidden_tensor = hidden_tensor.view(hidden_tensor.shape[0], 1)
        combined = torch.vstack((input_tensor, hidden_tensor))
        if self.bias==1:
            next_hidden_state = torch.add(torch.matmul(self.weights_hidden, combined), self.bias_hidden)
        else:
            next_hidden_state = torch.matmul(self.weights_hidden, combined)
        next_hidden_state = self.activation(next_hidden_state)
        output = torch.matmul(self.weights_output, next_hidden_state)
        output = self.activation(output)
        return next_hidden_state, output

    


    
    
class RNN(torch.nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, bias=False, dropout=False, bidirectional=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.parameter_dict = torch.nn.ParameterDict()
        self.parameter_dict_bias = torch.nn.ParameterDict()
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Dropout(0)
        self.D = 2 if bidirectional else 1 
        self.bias = torch.zeros(1, self.hidden_size) if bias else None
        weights_ih = torch.zeros(self.input_size+self.hidden_size, self.hidden_size)
        torch.nn.init.xavier_uniform_(weights_ih)
        weights_hh = torch.zeros(self.hidden_size+self.hidden_size*self.D, self.hidden_size)
        torch.nn.init.xavier_uniform_(weights_hh)
        self.tanh = torch.nn.Tanh()
        
        for layer_idx in range(num_layers): # NUM_LAYERS * NUM OF DIRECTIONS
            if self.D == 1:
                if layer_idx == 0:
                    self.parameter_dict["weights_ih_l_"+str(layer_idx)] = torch.nn.Parameter(weights_ih)
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
                else:
                    self.parameter_dict["weights_hh_l_"+str(layer_idx)] = torch.nn.Parameter(weights_hh) # CHANGE HERE
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
            elif self.D == 2:
                if layer_idx == 0:
                    self.parameter_dict["weights_ih_l_"+str(layer_idx)] = torch.nn.Parameter(weights_ih)
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
                    self.parameter_dict["weights_ih_l_"+str(layer_idx)+'_reverse'] = torch.nn.Parameter(weights_ih) # AND HERE
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)+'_reverse'] = torch.nn.Parameter(self.bias)
                else:
                    self.parameter_dict["weights_hh_l_"+str(layer_idx)] = torch.nn.Parameter(weights_hh)
                    self.parameter_dict_bias["bias_hh_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
                    self.parameter_dict["weights_hh_l_"+str(layer_idx)+'_reverse'] = torch.nn.Parameter(weights_hh) # AND HERE
                    self.parameter_dict_bias["bias_hh_l_"+str(layer_idx)+'reverse'] = torch.nn.Parameter(self.bias)
            
    
    def forward(self, input_tensor, hidden_tensor):
        assert self.D*self.num_layers == hidden_tensor.size(0)
        hidden_tensor_final = torch.empty(hidden_tensor.shape).to(device)
        if self.batch_first:
            batch_size, sequence_length, input_data = input_tensor.shape
            output = torch.zeros(batch_size, sequence_length, self.D, hidden_tensor.size(2)).to(device)
        else:
            sequence_length, batch_size, input_data = input_tensor.shape
            input_tensor = input_tensor.reshape(batch_size, sequence_length, input_data)
            output = torch.zeros(batch_size, sequence_length, self.D, hidden_tensor.size(2)).to(device)

        if self.D == 1:
            parameters_store = zip(self.parameter_dict.items(), self.parameter_dict_bias.items())
            for ht_idx, params in enumerate(parameters_store):
                weights, bias = params
                h_t = hidden_tensor[ht_idx, :, :].clone()
                for seq_idx in range(sequence_length):
                    curr_input_tensor = input_tensor[:, seq_idx, :].clone()
                    input_hidden_concat = torch.cat((curr_input_tensor, h_t), dim=1)
                    h_t = self.tanh(torch.addmm(bias[1], input_hidden_concat, weights[1]))
                    output[:, seq_idx, 0, :] = h_t
                hidden_tensor_final[ht_idx, :, :] = h_t
                input_tensor = output.reshape(batch_size, sequence_length, self.D*self.hidden_size)
                input_tensor = self.dropout(input_tensor)
        
        if self.D == 2:
            parameters_store = zip(self.parameter_dict.items(), self.parameter_dict_bias.items())
            for ht_idx, param in enumerate(parameters_store):
                h_t = hidden_tensor[ht_idx, :, :].clone()
                weights, bias = param
                if ht_idx % 2 == 0: # Only forward weights
                    for seq_idx in range(sequence_length):
                        curr_input_tensor_fwd = input_tensor[:, seq_idx, :].clone()
                        input_hidden_concat = torch.cat((curr_input_tensor_fwd, h_t), dim=1)
                        h_t = self.tanh(torch.addmm(bias[1], input_hidden_concat, weights[1]))
                        output[:, seq_idx, 0, :] = h_t
                    hidden_tensor_final[ht_idx, :, :] = h_t
                else: # Only reverse with is at an odd position
                    for rev_seq_idx in reversed(range(sequence_length)):
                        curr_input_tensor_bwd = input_tensor[:, rev_seq_idx, :].clone()
                        input_hidden_concat = torch.cat((curr_input_tensor_fwd, h_t), dim=1)
                        h_t = torch.tanh(torch.addmm(bias[1], input_hidden_concat, weights[1]))
                        output[:, rev_seq_idx, 1, :] = h_t
                    hidden_tensor_final[ht_idx, :, :] = h_t
                
                input_tensor = output.reshape(batch_size, sequence_length, self.D*self.hidden_size)
                input_tensor = self.dropout(input_tensor)

        if self.batch_first == False:
            output = output.reshape(sequence_length, batch_size, self.D*self.hidden_size)
        return input_tensor, hidden_tensor_final    
    




class CustomLSTMCell(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None) -> None:
        kwargs = {'device':device, 'dtype':dtype}
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(1, self.hidden_size*4, **kwargs))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = 0
        self.weights = torch.nn.Parameter(torch.empty(self.input_size+self.hidden_size, self.hidden_size*4, **kwargs))
        self.init_weights()
        
    
    def init_weights(self):
        k = 1. / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -k, k)
    
    
    def forward(self, input_tensor, hidden_tensor, initial_cell_state):
        assert hidden_tensor.size() == initial_cell_state.size()
        it_ht_combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        gates = torch.addmm(self.bias, it_ht_combined, self.weights)
        i_t, f_t, g_t, o_t =   (
            torch.sigmoid(gates[:, :self.hidden_size]), 
            torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2]),
            torch.tanh(gates[:, self.hidden_size*2:self.hidden_size*3]),
            torch.sigmoid(gates[:, self.hidden_size*3:])
        )
        initial_cell_state = f_t*initial_cell_state + i_t*g_t
        hidden_state = torch.tanh(initial_cell_state) * o_t
        
        return hidden_state, initial_cell_state





class CustomLSTM(torch.nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True, dropout=False, bidirectional=False):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.parameter_dict = torch.nn.ParameterDict()
        self.parameter_dict_bias = torch.nn.ParameterDict()
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Dropout(0)
        self.D = 2 if bidirectional else 1 
        self.bias = torch.zeros(1, self.hidden_size*4) if bias else None
        weights_ih = torch.zeros(self.input_size+self.hidden_size, self.hidden_size*4)
        torch.nn.init.xavier_uniform_(weights_ih)
        weights_hh = torch.zeros(self.hidden_size+self.hidden_size*self.D, self.hidden_size*4)
        torch.nn.init.xavier_uniform_(weights_hh)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        
        for layer_idx in range(num_layers): # NUM_LAYERS * NUM OF DIRECTIONS
            if self.D == 1:
                if layer_idx == 0:
                    self.parameter_dict["weights_ih_l_"+str(layer_idx)] = torch.nn.Parameter(weights_ih)
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
                else:
                    self.parameter_dict["weights_hh_l_"+str(layer_idx)] = torch.nn.Parameter(weights_hh) # CHANGE HERE
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
            elif self.D == 2:
                if layer_idx == 0:
                    self.parameter_dict["weights_ih_l_"+str(layer_idx)] = torch.nn.Parameter(weights_ih)
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
                    self.parameter_dict["weights_ih_l_"+str(layer_idx)+'_reverse'] = torch.nn.Parameter(weights_ih) # AND HERE
                    self.parameter_dict_bias["bias_ih_l_"+str(layer_idx)+'_reverse'] = torch.nn.Parameter(self.bias)
                else:
                    self.parameter_dict["weights_hh_l_"+str(layer_idx)] = torch.nn.Parameter(weights_hh)
                    self.parameter_dict_bias["bias_hh_l_"+str(layer_idx)] = torch.nn.Parameter(self.bias)
                    self.parameter_dict["weights_hh_l_"+str(layer_idx)+'_reverse'] = torch.nn.Parameter(weights_hh) # AND HERE
                    self.parameter_dict_bias["bias_hh_l_"+str(layer_idx)+'reverse'] = torch.nn.Parameter(self.bias)
        
    
    def forward(self, input_tensor, hidden_cell_tup):
        hidden_tensor, cell_state_tensor = hidden_cell_tup
        assert hidden_tensor.shape == cell_state_tensor.shape
        final_hidden_tensor = torch.zeros(hidden_tensor.shape).to(device)
        final_cell_state = torch.zeros(cell_state_tensor.shape).to(device)
        assert self.D*self.num_layers == hidden_tensor.size(0)
        hidden_tensor_final = torch.empty(hidden_tensor.shape).to(device)
        if self.batch_first:
            batch_size, sequence_length, input_data = input_tensor.shape
            output = torch.zeros(batch_size, sequence_length, self.D, hidden_tensor.size(2)).to(device)
        else:
            sequence_length, batch_size, input_data = input_tensor.shape
            input_tensor = input_tensor.reshape(batch_size, sequence_length, input_data)
            output = torch.zeros(batch_size, sequence_length, self.D, hidden_tensor.size(2)).to(device)

        if self.D == 1:
            parameters_store = zip(self.parameter_dict.items(), self.parameter_dict_bias.items())
            for ht_idx, params in enumerate(parameters_store):
                weights, bias = params
                h_t = hidden_tensor[ht_idx, :, :].clone()
                initial_cell_state = cell_state_tensor[ht_idx, :, :].clone()
                for seq_idx in range(sequence_length):
                    curr_input_tensor = input_tensor[:, seq_idx, :].clone()
                    input_hidden_concat = torch.cat((curr_input_tensor, h_t), dim=1)
                    gates = self.tanh(torch.addmm(bias[1], input_hidden_concat, weights[1]))
                    i_t, f_t, g_t, o_t =   (
                        torch.sigmoid(gates[:, :self.hidden_size]), 
                        torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2]),
                        torch.tanh(gates[:, self.hidden_size*2:self.hidden_size*3]),
                        torch.sigmoid(gates[:, self.hidden_size*3:])
                    )
                    initial_cell_state = f_t*initial_cell_state + i_t*g_t
                    h_t = torch.tanh(initial_cell_state) * o_t
                    output[:, seq_idx, 0, :] = h_t
                hidden_tensor_final[ht_idx, :, :] = h_t
                input_tensor = output.reshape(batch_size, sequence_length, self.D*self.hidden_size)  
                input_tensor = self.dropout(input_tensor)
                
        if self.D == 2:
            parameters_store = zip(self.parameter_dict.items(), self.parameter_dict_bias.items())
            for ht_idx, param in enumerate(parameters_store):
                h_t = hidden_tensor[ht_idx, :, :].clone()
                initial_cell_state = cell_state_tensor[ht_idx, :, :].clone()
                weights, bias = param
                if ht_idx % 2 == 0: # Only forward weights
                    for seq_idx in range(sequence_length):
                        curr_input_tensor_fwd = input_tensor[:, seq_idx, :].clone()
                        input_hidden_concat = torch.cat((curr_input_tensor_fwd, h_t), dim=1)
                        gates = self.tanh(torch.addmm(bias[1], input_hidden_concat, weights[1]))
                        i_t, f_t, g_t, o_t =   (
                            torch.sigmoid(gates[:, :self.hidden_size]), 
                            torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2]),
                            torch.tanh(gates[:, self.hidden_size*2:self.hidden_size*3]),
                            torch.sigmoid(gates[:, self.hidden_size*3:])
                        )
                        initial_cell_state = f_t*initial_cell_state + i_t*g_t
                        h_t = torch.tanh(initial_cell_state) * o_t
                        output[:, seq_idx, 0, :] = h_t
                    final_hidden_tensor[ht_idx, :, :] = h_t
                    final_cell_state[ht_idx, :, :] = initial_cell_state
                else: # Only reverse with is at an odd position
                    for rev_seq_idx in reversed(range(sequence_length)):
                        curr_input_tensor_bwd = input_tensor[:, rev_seq_idx, :].clone()
                        input_hidden_concat = torch.cat((curr_input_tensor_fwd, h_t), dim=1)
                        gates = torch.tanh(torch.addmm(bias[1], input_hidden_concat, weights[1]))
                        i_t, f_t, g_t, o_t =   (
                            torch.sigmoid(gates[:, :self.hidden_size]), 
                            torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2]),
                            torch.tanh(gates[:, self.hidden_size*2:self.hidden_size*3]),
                            torch.sigmoid(gates[:, self.hidden_size*3:])
                        )
                        initial_cell_state = f_t*initial_cell_state + i_t*g_t
                        h_t = torch.tanh(initial_cell_state) * o_t
                        output[:, rev_seq_idx, 1, :] = h_t
                    final_hidden_tensor[ht_idx, :, :] = h_t
                    final_cell_state[ht_idx, :, :] = initial_cell_state
                
                input_tensor = output.reshape(batch_size, sequence_length, self.D*self.hidden_size)
                input_tensor = self.dropout(input_tensor)
                
        if self.batch_first == False:
            output = output.reshape(sequence_length, batch_size, self.D*self.hidden_size)
        return input_tensor, (final_hidden_tensor, final_cell_state)