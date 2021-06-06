import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


NINE_WAY_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
FIVE_WAY_MAP = {0: 4, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 3, 8: 2}
FOUR_WAY_MAP = {0: 3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 1}


def get_channel_progression(in_channels, block_length, update_rules=[2], cycle=True, hid_max=2048):
    n = len(update_rules)
    last_val = in_channels
    channel_progression = [in_channels]
    for i in range(block_length):
        rule = update_rules[-1] if not cycle and i >= n else update_rules[i % n]
        last_val *= rule
        channel_progression.append(min(round(last_val), hid_max))

    return channel_progression[1:]


def get_model_size(model, input_size, batch_size=-1):
    def get_shape(output):
        shape = []
        if isinstance(output, (list, tuple)):
            shape = [get_shape(o) for o in output]
        else:
            shape = list(output.size())

            for i in range(len(shape)):
                if shape[i] == batch_size:
                    shape[i] = -1
                    return shape
        return shape

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = get_shape(output)

            params = 0

            for name, param in module.named_parameters():
                params += torch.prod(torch.LongTensor(list(param.size())))
                summary[m_key]['trainable'] = param.requires_grad
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
           not isinstance(module, nn.ModuleList) and
           not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(batch_size, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(batch_size, *input_size)).type(dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    key = [k for k in model.state_dict().keys() if 'head' in k and '0.0' in k and 'weight' in k]
    if not key:
        key = [k for k in model.state_dict().keys() if 'head' in k and 'weight_ih_l0' in k and 'reverse' not in k]
    prec = model.state_dict()[key[0]].shape[1]
    if prec >= 256:
        prec -= 256
        prec = Variable(torch.rand(batch_size, prec, x.shape[2] // 160)).type(dtype) if prec != 0 else None
        model(x, precomputed=prec)
    else:
        prec = Variable(torch.rand(batch_size, prec, x.shape[2] // 160)).type(dtype) if prec != 0 else None
        model(None, precomputed=prec)

    # remove these hooks
    for h in hooks:
        h.remove()

    total_params = 0
    frozen_output = 0
    train_output = 0
    for layer in summary:
        total_params += summary[layer]["nb_params"]

        los = summary[layer]["output_shape"]
        out = 0
        if isinstance(los[0], (list, tuple)):
            for l in los:
                out += np.prod(l)
        else:
            out = np.prod(los)

        if "trainable" in summary[layer] and summary[layer]["trainable"]:
            train_output += out
        else:
            frozen_output += out

    # assume 4 bytes/number (float on cuda).
    total_input_size  = abs(np.prod(input_size) * batch_size        * 4 / (1024 ** 2))
    total_output_size = abs((frozen_output + (1.5 * train_output))  * 4 / (1024 ** 2))  # x2 for gradients
    total_params_size = abs(total_params                            * 4 / (1024 ** 2))

    return (total_input_size, total_output_size, total_params_size)


def get_accumulation_iters(model, input_shape, memory_cap, batch_size):
    sizes = get_model_size(model, input_shape, 1)
    max_mb = min(memory_cap // sum(sizes), batch_size)
    acc_it = math.ceil(batch_size / max_mb)

    print(f'mb={max_mb}, acc_iter={acc_it}')
    return int(max_mb), int(acc_it)
