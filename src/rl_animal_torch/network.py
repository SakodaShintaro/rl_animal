import torch
import torch.nn.functional as F
from torch import nn

from rl_animal_torch.config import EnvConfig

_ENV = EnvConfig()
VISUAL_SIZE = _ENV.resolution
# derived rather than written out so they cannot drift from what the wrappers produce
VISUAL_CHANNELS = 3 * _ENV.visual_frames
VELS_SIZE = 4 * _ENV.velocity_frames
ACTIONS_NUM = 9
DEPTHS = (16, 32, 64, 128)
HIDDEN_NODES = 1024
VELS_HIDDEN = 128
LSTM_UNITS = 512
LAYER_NORM_EPSILON = 1e-5


class ChannelAttention(nn.Module):
    """
    networks.channel_attention: the channel means through two bias-free 1x1 convolutions,
    squashed to a per-channel gate.
    """

    def __init__(self, depth):
        super().__init__()
        self.reduce = nn.Conv2d(depth, depth // 4, 1, bias=False)
        self.expand = nn.Conv2d(depth // 4, depth, 1, bias=False)

    def forward(self, x):
        out = x.mean(dim=(2, 3), keepdim=True)
        out = self.expand(F.elu(self.reduce(out)))
        return torch.sigmoid(out)


class FixupAttentionBlock(nn.Module):
    """
    networks.residual_block_fixup_attention: a residual block with no normalisation,
    four scalar biases and a scalar multiplier, and the channel gate applied between the
    two convolutions.
    """

    def __init__(self, depth):
        super().__init__()
        self.res1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.res2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.attention = ChannelAttention(depth)
        self.bias0 = nn.Parameter(torch.zeros(()))
        self.bias1 = nn.Parameter(torch.zeros(()))
        self.bias2 = nn.Parameter(torch.zeros(()))
        self.bias3 = nn.Parameter(torch.zeros(()))
        self.multiplier = nn.Parameter(torch.ones(()))

    def forward(self, x):
        out = F.elu(x) + self.bias0
        out = self.res1(out) + self.bias1
        out = out * self.attention(out)
        out = F.elu(out) + self.bias2
        out = self.res2(out) * self.multiplier + self.bias3
        return out + x


class LayerNormLSTMCell(nn.Module):
    """
    networks.lnlstm. The gates are ordered i, f, o, u, the input and recurrent
    contributions are normalized separately over the whole 4 * units axis, and the cell
    state is normalized again before the output gate.
    """

    def __init__(self, input_size, units):
        super().__init__()
        self.units = units
        self.wx = nn.Parameter(torch.zeros(input_size, 4 * units))
        self.gx = nn.Parameter(torch.ones(4 * units))
        self.bx = nn.Parameter(torch.zeros(4 * units))
        self.wh = nn.Parameter(torch.zeros(units, 4 * units))
        self.gh = nn.Parameter(torch.ones(4 * units))
        self.bh = nn.Parameter(torch.zeros(4 * units))
        self.b = nn.Parameter(torch.zeros(4 * units))
        self.gc = nn.Parameter(torch.ones(units))
        self.bc = nn.Parameter(torch.zeros(units))

    @staticmethod
    def normalize(x, gain, bias):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + LAYER_NORM_EPSILON) * gain + bias

    def forward(self, inputs, state, masks):
        """
        inputs and masks are sequences of (env_num, ...) tensors, state is the
        (env_num, 2 * units) pair the training loop carries between batches.
        """
        cell, hidden = torch.split(state, self.units, dim=1)
        outputs = []
        for x, mask in zip(inputs, masks):
            keep = (1.0 - mask).unsqueeze(1)
            cell = cell * keep
            hidden = hidden * keep
            z = (
                self.normalize(x.matmul(self.wx), self.gx, self.bx)
                + self.normalize(hidden.matmul(self.wh), self.gh, self.bh)
                + self.b
            )
            i, f, o, u = torch.split(z, self.units, dim=1)
            cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(u)
            hidden = torch.sigmoid(o) * torch.tanh(self.normalize(cell, self.gc, self.bc))
            outputs.append(hidden)

        return outputs, torch.cat([cell, hidden], dim=1)


class AnimalAgent(nn.Module):
    """
    The whole network. forward takes the observations the v1 wrappers produced: visual
    uint8 in NHWC and the stacked velocity vector.
    """

    def __init__(self):
        super().__init__()
        self.tower = nn.ModuleList()
        in_channels = VISUAL_CHANNELS
        for depth in DEPTHS:
            self.tower.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels, depth, 3, padding=1),
                        "block1": FixupAttentionBlock(depth),
                        "block2": FixupAttentionBlock(depth),
                    }
                )
            )
            in_channels = depth

        spatial = VISUAL_SIZE
        for _ in DEPTHS:
            spatial = -(-spatial // 2)
        self.flat_size = spatial * spatial * DEPTHS[-1]

        self.vels_hidden = nn.Linear(VELS_SIZE, VELS_HIDDEN)
        self.visual_hidden = nn.Linear(self.flat_size, HIDDEN_NODES)
        self.joint_hidden = nn.Linear(VELS_HIDDEN + HIDDEN_NODES, HIDDEN_NODES)
        self.lstm = LayerNormLSTMCell(HIDDEN_NODES, LSTM_UNITS)
        self.value_head = nn.Linear(LSTM_UNITS, 1)
        self.logits_head = nn.Linear(LSTM_UNITS, ACTIONS_NUM)

    def initial_state(self, env_num, dtype=torch.float32, device=None):
        return torch.zeros(env_num, 2 * LSTM_UNITS, dtype=dtype, device=device)

    def features(self, visual):
        """
        visual is uint8 NHWC, as the environment wrappers produce it.
        """
        out = visual.permute(0, 3, 1, 2).to(self.value_head.weight.dtype) / 255.0
        for stage in self.tower:
            out = stage["conv"](out)
            out = F.max_pool2d(out, 3, 2, padding=1)
            out = stage["block1"](out)
            out = stage["block2"](out)
        out = F.elu(out)
        # channels last before flattening: the order the dense layer's weights expect
        return out.permute(0, 2, 3, 1).reshape(out.shape[0], -1)

    def forward(self, visual, vels, state, dones, env_num):
        """
        The batch is laid out environment-major: the entry for environment e at step t is
        at e * steps_num + t, which is what batch_to_seq assumed and what the rollout
        buffer reproduces when it flattens.
        """
        steps_num = visual.shape[0] // env_num
        flat = self.features(visual)
        hidden = torch.cat([F.elu(self.vels_hidden(vels)), F.elu(self.visual_hidden(flat))], dim=-1)
        hidden = F.elu(self.joint_hidden(hidden))

        # the batch is environment-major: environment e at step t sits at e * steps_num + t
        sequence = hidden.reshape(env_num, steps_num, -1).unbind(dim=1)
        mask_sequence = dones.to(hidden.dtype).reshape(env_num, steps_num).unbind(dim=1)
        outputs, lstm_state = self.lstm(sequence, state, mask_sequence)
        lstm_out = torch.stack(outputs, dim=1).reshape(-1, LSTM_UNITS)

        return self.logits_head(lstm_out), self.value_head(lstm_out), lstm_state
