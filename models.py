import json
import warnings

from pase.models.frontend import wf_builder
import torch
import torch.nn as nn


warnings.filterwarnings('ignore')


class TransposeLayerNorm(nn.Module):
    def __init__(self, features, *args, **kwargs):
        super(TransposeLayerNorm, self).__init__()
        self.lnorm = nn.LayerNorm(features, *args, **kwargs)

    def forward(self, x):
        # MBS x EMB x SEQ
        x = x.transpose(1, 2)
        x = self.lnorm(x)
        x = x.transpose(1, 2)
        return x


def build_smooth_out(hid, cla, smooth=1):
    return nn.Sequential(
        nn.Conv1d(hid, hid, smooth, padding=smooth//2, padding_mode='replicate'),
        nn.LeakyReLU(),
        nn.Conv1d(hid, cla, 1)
    )


def build_norm(norm, hid, affine=True):
    if norm == 'bnorm':
        return nn.BatchNorm1d(hid, affine=affine)
    elif norm == 'lnorm':
        return TransposeLayerNorm(hid, elementwise_affine=affine)
    elif norm == 'inorm':
        return nn.InstanceNorm1d(hid, affine=False)
    elif norm == 'affinorm':
        return nn.InstanceNorm1d(hid, affine=True)
    elif norm is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm)


class PASEEncodedModel(nn.Module):
    def __init__(self, head, cfg, checkpoint, drop_inp=0.0, drop_emb=0.0, freeze_bn=False, tune=False):
        super(PASEEncodedModel, self).__init__()
        with open(cfg, 'r') as f:
            cfg = json.load(f)
        # Note: Modify cfg dict here if desired
        # This seems to overall make things worse though
        # cfg['norm_type'] = norm
        encoder = wf_builder(cfg)
        encoder.load_pretrained(checkpoint, load_last=True, verbose=False)

        if not tune:
            encoder.requires_grad_(False)

        self.tune = tune
        self.freeze_bn = freeze_bn
        self.drop_inp = nn.Dropout(p=drop_inp)
        self.encoder = encoder
        self.drop_emb = nn.Dropout(p=drop_emb)
        self.head = head

    def forward(self, signal=None, precomputed=None):
        if signal is not None:
            x = self.drop_inp(signal)
            x = self.encoder(x)
            if not self.tune:
                x = x.detach()
        if precomputed is not None:
            if signal is None:
                x = precomputed
            else:
                x = torch.cat([x, precomputed], dim=1)
        x = self.drop_emb(x)
        y = self.head(x)
        return y

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.freeze_bn:
            self.encoder.eval()
        return self


class CNNHead(nn.Module):
    def __init__(self, input_size, num_classes, dilation_base, hidden_channels,
                 smooth, context_size, kernel_size, norm, dropout):
        super(CNNHead, self).__init__()

        # Convs
        self.dilated_convs = nn.ModuleList()
        hidden_channels.insert(0, input_size)
        for i in range(1, len(hidden_channels)):
            ks = context_size if i == 1 else kernel_size
            dilation = int(round(dilation_base**(i-1)))
            block = [
                nn.Conv1d(hidden_channels[i-1], hidden_channels[i], ks, dilation=dilation),
                nn.LeakyReLU(),
            ]
            if norm is not None:
                block.append(build_norm(norm, hidden_channels[i], affine=True))
            self.dilated_convs.append(nn.Sequential(*block))

        # Hidden to output
        self.dropout = nn.Dropout(p=dropout)
        self.hid2out = build_smooth_out(hidden_channels[-1], num_classes, smooth=smooth)

    def forward(self, x):
        # Forward through CNN
        for dc in self.dilated_convs:
            x = dc(x)  # x: [mbs x DC[i] x seq]

        # Get output and smooth
        x = self.dropout(x)
        y = self.hid2out(x)  # x: [mbs x cla x seq]
        return y


class LSTMHead(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, hidden_layers,
                 smooth, bidirectional, dropout):
        super(LSTMHead, self).__init__()

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, hidden_layers, dropout=dropout,
                            batch_first=True, bidirectional=bidirectional)

        # Hidden to Output
        h_size = hidden_size*2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.hid2out = build_smooth_out(h_size, num_classes, smooth=smooth)

    def forward(self, x):
        # Forward through the LSTM
        x = x.transpose(1, 2)  # [mbs x emb x seq] --> [mbs x seq x emb]
        x, _ = self.lstm(x)    # [mbs x seq x emb] --> [mbs x seq x hid]
        x = x.transpose(1, 2)  # [mbs x seq x hid] --> [mbs x hid x seq]

        # Get output and smooth
        x = self.dropout(x)
        y = self.hid2out(x)   # [mbs x hid x seq] --> [mbs x cls x seq]
        return y


class GRUHead(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, hidden_layers,
                 smooth, bidirectional, dropout):
        super(GRUHead, self).__init__()

        # GRU
        self.gru = nn.GRU(input_size, hidden_size, hidden_layers, dropout=dropout,
                          batch_first=True, bidirectional=bidirectional)

        # Hidden to Output
        h_size = hidden_size*2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.hid2out = build_smooth_out(h_size, num_classes, smooth=smooth)

    def forward(self, x):
        # Forward through the GRU
        x = x.transpose(1, 2)  # [mbs x emb x seq] --> [mbs x seq x emb]
        x, _ = self.gru(x)     # [mbs x seq x emb] --> [mbs x seq x hid]
        x = x.transpose(1, 2)  # [mbs x seq x hid] --> [mbs x hid x seq]

        # Get output and smooth
        x = self.dropout(x)
        y = self.hid2out(x)   # [mbs x hid x seq] --> [mbs x cls x seq]
        return y
