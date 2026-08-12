"""The trained policy network in PyTorch, loading the TensorFlow 1.15 checkpoint.

networks.animal_a2c_network_lstm6 is the architecture the released checkpoints were
trained with, and TensorFlow 1.15 only runs on this GPU generation through NVIDIA's
container image, which is not distributed as a wheel. This is the same network built with
PyTorch so that inference no longer depends on that image.

verify_torch_parity.py checks this against a reference forward pass dumped by
export_tf_reference.py. The things that have to be right, and are easy to get wrong:

- TensorFlow flattens NHWC, so the feature map has to be permuted back to NHWC before
  being flattened, or the 4608-wide dense layer sees its inputs in the wrong order.
- 'SAME' padding is asymmetric when the padding is odd, which it is for the first two
  max pools (84 -> 42 and 42 -> 21 both need one row and column, and TensorFlow puts it
  at the bottom and right). nn.MaxPool2d's symmetric padding gives the same output size
  but shifts every pixel.
- The layer-normalised LSTM normalises the input and recurrent contributions separately,
  over the whole 4 * units axis rather than per gate, adds a third bias afterwards, and
  normalises the cell state again inside the output gate. Its gate order is i, f, o, u.
- A batch of batch_num observations is laid out environment-major: the entry for
  environment e at step t is at e * steps_num + t.

The action is not reproduced here. models.LSTMModelA2C draws it with a Gumbel-max over a
TensorFlow random tensor, so sampling can only agree in distribution; callers that need
the same decisions should sample from these logits themselves.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

VISUAL_SIZE = 84
VISUAL_CHANNELS = 6
VELS_SIZE = 8
ACTIONS_NUM = 9
DEPTHS = (16, 32, 64, 128)
HIDDEN_NODES = 1024
VELS_HIDDEN = 128
LSTM_UNITS = 512
'''
networks._ln adds this to the variance before the square root.
'''
LAYER_NORM_EPSILON = 1e-5


def same_padding(size, kernel, stride):
    '''
    What TensorFlow's 'SAME' adds for one dimension, as (before, after). The total is
    split with the larger half at the end, so odd totals are asymmetric.
    '''
    out = -(-size // stride)
    total = max((out - 1) * stride + kernel - size, 0)
    before = total // 2
    return before, total - before


def max_pool_same(x, kernel, stride):
    '''
    tf.layers.max_pooling2d(padding='same') excludes the padded region from the maximum,
    which is -inf padding, and pads asymmetrically when the total is odd.
    '''
    top, bottom = same_padding(x.shape[2], kernel, stride)
    left, right = same_padding(x.shape[3], kernel, stride)
    padded = F.pad(x, (left, right, top, bottom), value=float('-inf'))
    return F.max_pool2d(padded, kernel, stride)


class ChannelAttention(nn.Module):
    '''
    networks.channel_attention: the channel means through two bias-free 1x1 convolutions,
    squashed to a per-channel gate.
    '''
    def __init__(self, depth):
        super(ChannelAttention, self).__init__()
        self.reduce = nn.Conv2d(depth, depth // 4, 1, bias=False)
        self.expand = nn.Conv2d(depth // 4, depth, 1, bias=False)

    def forward(self, x):
        out = x.mean(dim=(2, 3), keepdim=True)
        out = self.expand(F.elu(self.reduce(out)))
        return torch.sigmoid(out)


class FixupAttentionBlock(nn.Module):
    '''
    networks.residual_block_fixup_attention: a residual block with no normalisation,
    four scalar biases and a scalar multiplier, and the channel gate applied between the
    two convolutions.
    '''
    def __init__(self, depth):
        super(FixupAttentionBlock, self).__init__()
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
    '''
    networks.lnlstm. The gates are ordered i, f, o, u, the input and recurrent
    contributions are normalised separately over the whole 4 * units axis, and the cell
    state is normalised again before the output gate.
    '''
    def __init__(self, input_size, units):
        super(LayerNormLSTMCell, self).__init__()
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
    def normalise(x, gain, bias):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + LAYER_NORM_EPSILON) * gain + bias

    def forward(self, inputs, state, masks):
        '''
        inputs and masks are sequences of (env_num, ...) tensors, state is the
        (env_num, 2 * units) pair the training loop carries between batches.
        '''
        cell, hidden = torch.split(state, self.units, dim=1)
        outputs = []
        for x, mask in zip(inputs, masks):
            keep = (1.0 - mask).unsqueeze(1)
            cell = cell * keep
            hidden = hidden * keep
            z = (self.normalise(x.matmul(self.wx), self.gx, self.bx)
                 + self.normalise(hidden.matmul(self.wh), self.gh, self.bh)
                 + self.b)
            i, f, o, u = torch.split(z, self.units, dim=1)
            cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(u)
            hidden = torch.sigmoid(o) * torch.tanh(self.normalise(cell, self.gc, self.bc))
            outputs.append(hidden)

        return outputs, torch.cat([cell, hidden], dim=1)


class AnimalAgent(nn.Module):
    '''
    The whole network. forward takes the observations the v1 wrappers produced: visual
    uint8 in NHWC and the stacked velocity vector.
    '''
    def __init__(self, env_num, steps_num):
        super(AnimalAgent, self).__init__()
        self.env_num = env_num
        self.steps_num = steps_num

        self.tower = nn.ModuleList()
        in_channels = VISUAL_CHANNELS
        for depth in DEPTHS:
            self.tower.append(nn.ModuleDict({
                'conv': nn.Conv2d(in_channels, depth, 3, padding=1),
                'block1': FixupAttentionBlock(depth),
                'block2': FixupAttentionBlock(depth),
            }))
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

    def initial_state(self, dtype=torch.float32):
        return torch.zeros(self.env_num, 2 * LSTM_UNITS, dtype=dtype)

    def features(self, visual):
        '''
        visual is uint8 NHWC, as the environment wrappers produce it.
        '''
        out = visual.permute(0, 3, 1, 2).to(self.value_head.weight.dtype) / 255.0
        for stage in self.tower:
            out = stage['conv'](out)
            out = max_pool_same(out, 3, 2)
            out = stage['block1'](out)
            out = stage['block2'](out)
        out = F.elu(out)
        '''
        Back to NHWC before flattening, because that is the order the dense layer's
        weights were trained in.
        '''
        return out.permute(0, 2, 3, 1).reshape(out.shape[0], -1)

    def forward(self, visual, vels, state, dones):
        flat = self.features(visual)
        hidden = torch.cat([F.elu(self.vels_hidden(vels)),
                            F.elu(self.visual_hidden(flat))], dim=-1)
        hidden = F.elu(self.joint_hidden(hidden))

        '''
        batch_to_seq reshapes the environment-major batch into (env_num, steps_num) and
        hands the steps over one at a time.
        '''
        sequence = hidden.reshape(self.env_num, self.steps_num, -1).unbind(dim=1)
        mask_sequence = dones.to(hidden.dtype).reshape(self.env_num, self.steps_num).unbind(dim=1)
        outputs, lstm_state = self.lstm(sequence, state, mask_sequence)
        lstm_out = torch.stack(outputs, dim=1).reshape(-1, LSTM_UNITS)

        return self.logits_head(lstm_out), self.value_head(lstm_out), lstm_state


def load_tf_weights(agent, weights):
    '''
    Copies the TensorFlow variables in `weights`, keyed by variable name without the
    ':0', into the module. Convolution kernels are stored as [H, W, in, out] and dense
    kernels as [in, out], both of which PyTorch wants transposed.

    The channel attention convolutions are the anonymous `conv2d`, `conv2d_1`, ... in the
    checkpoint, numbered in creation order: the tower visits the depths in order and each
    of its two blocks builds its own pair, so the pairs run rb11, rb21, rb12, rb22, rb13,
    rb23, rb14, rb24. The residual convolutions instead carry the block name, as
    res1/rb1<layer> and res2/rb1<layer>.
    '''
    def conv_kernel(name):
        return torch.as_tensor(np.transpose(weights[name], (3, 2, 0, 1)).copy())

    def dense_kernel(name):
        return torch.as_tensor(np.transpose(weights[name], (1, 0)).copy())

    def vector(name):
        return torch.as_tensor(weights[name].copy())

    with torch.no_grad():
        attention_index = 0
        for layer, (stage, depth) in enumerate(zip(agent.tower, DEPTHS), start=1):
            stage['conv'].weight.copy_(conv_kernel('agent/layer_%d/kernel' % layer))
            stage['conv'].bias.copy_(vector('agent/layer_%d/bias' % layer))
            for block_name, block in (('rb1%d' % layer, stage['block1']),
                                      ('rb2%d' % layer, stage['block2'])):
                block.res1.weight.copy_(conv_kernel('agent/res1/%s/kernel' % block_name))
                block.res2.weight.copy_(conv_kernel('agent/res2/%s/kernel' % block_name))
                for scalar in ('bias0', 'bias1', 'bias2', 'bias3', 'multiplier'):
                    getattr(block, scalar).copy_(
                        vector('agent/%s/%s' % (block_name, scalar)))

                reduce_name = 'agent/conv2d' if attention_index == 0 \
                    else 'agent/conv2d_%d' % attention_index
                expand_name = 'agent/conv2d_%d' % (attention_index + 1)
                block.attention.reduce.weight.copy_(conv_kernel(reduce_name + '/kernel'))
                block.attention.expand.weight.copy_(conv_kernel(expand_name + '/kernel'))
                attention_index += 2

        for module, name in ((agent.vels_hidden, 'agent/dense'),
                             (agent.visual_hidden, 'agent/dense_1'),
                             (agent.joint_hidden, 'agent/dense_2'),
                             (agent.value_head, 'agent/dense_3'),
                             (agent.logits_head, 'agent/dense_4')):
            module.weight.copy_(dense_kernel(name + '/kernel'))
            module.bias.copy_(vector(name + '/bias'))

        for parameter in ('wx', 'gx', 'bx', 'wh', 'gh', 'bh', 'b', 'gc', 'bc'):
            getattr(agent.lstm, parameter).copy_(
                vector('agent/lstm_ac/lnlstm/%s' % parameter))

    return agent


def load_from_reference(path, env_num, steps_num, dtype=torch.float32):
    '''
    Builds the agent from the weights export_tf_reference.py wrote.
    '''
    dump = np.load(path)
    weights = {key[len('weight/'):]: dump[key] for key in dump.files
               if key.startswith('weight/')}
    agent = AnimalAgent(env_num, steps_num).to(dtype)
    load_tf_weights(agent, weights)
    agent.eval()
    return agent
