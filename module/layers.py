from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class SeqPoolingLayer(nn.Module):
    def __init__(self, pooling_type='mean', keepdim=False) -> None:
        super().__init__()
        if not pooling_type in ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']:
            raise ValueError("pooling_type can only be one of ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']"
                             f"but {pooling_type} is given.")
        self.pooling_type = pooling_type
        self.keepdim = keepdim

    def forward(self, batch_seq_embeddings, seq_len, weight=None, mask_token=None):
        # batch_seq_embeddings: [B, L, D] or [B, Neg, L, D]
        # seq_len: [B] or [B,Neg], weight: [B,L] or [B,Neg,L]
        B = batch_seq_embeddings.size(0)
        _need_reshape = False
        if batch_seq_embeddings.dim() == 4:
            _need_reshape = True
            batch_seq_embeddings = batch_seq_embeddings.view(
                -1, *batch_seq_embeddings.shape[2:])
            seq_len = seq_len.view(-1)
            if weight is not None:
                weight = weight.view(-1, weight.size(-1))

        N, L, D = batch_seq_embeddings.shape

        if weight is not None:
            batch_seq_embeddings = weight.unsqueeze(-1) * batch_seq_embeddings

        if self.pooling_type == 'mask':
            # Data type of mask_token should be bool and 
            # the shape of mask_token should be [B, L]
            assert mask_token != None, "mask_token can be None when pooling_type is 'mask'."
            result = batch_seq_embeddings[mask_token]
        elif self.pooling_type in ['origin', 'concat', 'mean', 'sum', 'max']:
            mask = torch.arange(L).unsqueeze(0).unsqueeze(2).to(batch_seq_embeddings.device)
            mask = mask.expand(N, -1,  D)
            seq_len = seq_len.unsqueeze(1).unsqueeze(2)
            seq_len_ = seq_len.expand(-1, mask.size(1), -1)
            mask = mask >= seq_len_
            batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)

            if self.pooling_type == 'origin':
                return batch_seq_embeddings
            elif self.pooling_type in ['concat', 'max']:
                if not self.keepdim:
                    if self.pooling_type == 'concat':
                        result = batch_seq_embeddings.reshape(N, -1)
                    else:
                        result = batch_seq_embeddings.max(dim=1)
                else:
                    if self.pooling_type == 'concat':
                        result = batch_seq_embeddings.reshape(N, -1).unsqueeze(1)
                    else:
                        result = batch_seq_embeddings.max(dim=1).unsqueeze(1)
            elif self.pooling_type in ['mean', 'sum']:
                batch_seq_embeddings_sum = batch_seq_embeddings.sum(dim=1, keepdim=self.keepdim)
                if self.pooling_type == 'sum':
                    result = batch_seq_embeddings_sum
                else:
                    result = batch_seq_embeddings_sum / (seq_len + torch.finfo(torch.float32).eps if self.keepdim else seq_len.squeeze(2))

        elif self.pooling_type == 'last':
            gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, D)  # B x 1 x D
            output = batch_seq_embeddings.gather(
                dim=1, index=gather_index).squeeze(1)  # B x D
            result = output if not self.keepdim else output.unsqueeze(1)

        if _need_reshape:
            return result.reshape(B, N//B, *result.shape[1:])
        else:
            return result
        

class HStackLayer(torch.nn.Sequential):

    def forward(self, *input):
        output = []
        assert (len(input) == 1) or (len(input) == len(list(self.children())))
        for i, module in enumerate(self):
            if len(input) == 1:
                output.append(module(input[0]))
            else:
                output.append(module(input[i]))
        return tuple(output)


class VStackLayer(torch.nn.Sequential):

    def forward(self, input):
        for module in self:
            if isinstance(input, Tuple):
                input = module(*input)
            else:
                input = module(input)
        return input
    
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambda_func) -> None:
        super().__init__()
        self.lambda_func = lambda_func

    def forward(self, *args):
        # attention: all data input into LambdaLayer will be tuple
        # even if there is only one input, the args will be the tuple of length 1
        if len(args) == 1:  # only one input
            return self.lambda_func(args[0])
        else:
            return self.lambda_func(args)
        
class GRULayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layer=1, bias=False, batch_first=True,
                 bidirectional=False, return_hidden=False) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=num_layer,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.return_hidden = return_hidden

    def forward(self, input):
        out, hidden = self.gru(input)
        if self.return_hidden:
            return out, hidden
        else:
            return out

def get_act(activation: str, dim=None):
    if activation == None or isinstance(activation, torch.nn.Module):
        return activation
    elif type(activation) == str:
        if activation.lower() == 'relu':
            return torch.nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return torch.nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        elif activation.lower() == 'identity':
            return lambda x: x
        elif activation.lower() == 'gelu':
            return torch.nn.GELU()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(
                f'activation function type "{activation}"  is not supported, check spelling or pass in a instance of torch.nn.Module.')
    else:
        raise ValueError(
            '"activation_func" must be a str or a instance of torch.nn.Module. ')

class MLPModule(torch.nn.Module):
    """
    MLPModule
    Gets a MLP easily and quickly.

    Args:
        mlp_layers(list): the dimensions of every layer in the MLP.
        activation_func(torch.nn.Module,str,None): the activation function in each layer.
        dropout(float): the probability to be set in dropout module. Default: ``0.0``.
        bias(bool): whether to add batch normalization between layers. Default: ``False``.
        last_activation(bool): whether to add activation in the last layer. Default: ``True``.
        last_bn(bool): whether to add batch normalization in the last layer. Default: ``True``.

    Examples:
    >>> MLP = MLPModule([64, 64, 64], 'ReLU', 0.2)
    >>> MLP.model
    Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=64, out_features=64, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=64, out_features=64, bias=True)
        (5): ReLU()
    )
    >>> MLP.add_modules(torch.nn.Linear(64, 10, True), torch.nn.ReLU())
    >>> MLP.model
    Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=64, out_features=64, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=64, out_features=64, bias=True)
        (5): ReLU()
        (6): Linear(in_features=64, out_features=10, bias=True)
        (7): ReLU()
    )
    """

    def __init__(self, mlp_layers, activation_func='ReLU', dropout=0.0, bias=True, batch_norm=False, last_activation=True, last_bn=True):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.bias = bias
        self.dropout = dropout
        self.activation_func = activation_func
        self.model = []
        last_bn = self.batch_norm and last_bn
        for idx, layer in enumerate((zip(self.mlp_layers[: -1], self.mlp_layers[1:]))):
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.Linear(*layer, bias=bias))
            if (idx == len(mlp_layers)-2 and last_bn) or (idx < len(mlp_layers)-2 and batch_norm):
                self.model.append(torch.nn.BatchNorm1d(layer[-1]))
            if ( (idx == len(mlp_layers)-2 and last_activation and activation_func is not None)
                or (idx < len(mlp_layers)-2 and activation_func is not None) ):
                activation = get_act(activation_func, dim=layer[-1])
                self.model.append(activation)
        self.model = torch.nn.Sequential(*self.model)

    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model.
        """
        for block in args:
            assert isinstance(block, torch.nn.Module)

        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, input):
        return self.model(input)