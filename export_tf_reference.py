"""Dump the checkpoint's weights and a reference forward pass, for the PyTorch port.

Runs inside the TensorFlow 1.15 container, which is the only place this network can be
executed. verify_torch_parity.py reads what this writes and checks the port against it.

    python export_tf_reference.py --checkpoint nn/last84_10_5 \
        --output local/tf_reference.npz --env-num 2 --steps-num 1

The graph is built by calling networks.animal_a2c_network_lstm6 directly rather than
through models.LSTMModelA2C, because the wrapper only exposes the sampled action and the
port has to be compared on the logits: the action is drawn with a Gumbel-max over a
TensorFlow random tensor, which no other framework will reproduce.
"""
import argparse
import re

import numpy as np
import tensorflow as tf

import networks

'''
Every intermediate worth comparing when a mismatch has to be localised. Anything
matching is dumped alongside the outputs; the names are TensorFlow operation names.
'''
INTERMEDIATE_PATTERNS = (
    r'^agent/layer_\d+/BiasAdd$',
    r'^agent/max_pooling2d(_\d+)?/MaxPool$',
    r'^agent/add(_\d+)?$',
    r'^agent/Elu$',
    r'^agent/[Ff]latten/[Rr]eshape$',
    r'^agent/dense(_\d+)?/Elu$',
    r'^agent/dense(_\d+)?/BiasAdd$',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='e.g. nn/last84_10_5')
    parser.add_argument('--output', required=True, help='npz to write')
    parser.add_argument('--env-num', required=True, type=int)
    parser.add_argument('--steps-num', required=True, type=int,
                        help='decisions per environment in the batch; 1 for inference')
    parser.add_argument('--seed', required=True, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    env_num = args.env_num
    batch_num = env_num * args.steps_num

    obs_ph = tf.placeholder('uint8', (None, 84, 84, 6), name='obs')
    vels_ph = tf.placeholder(tf.float32, [batch_num, 8], name='vels')
    logits, value, states_ph, _, dones_ph, lstm_state, initial_state = \
        networks.animal_a2c_network_lstm6('agent', tf.to_float(obs_ph) / 255.0, 9,
                                         env_num, batch_num, vels_ph)

    session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    '''
    Only the variables this graph declares are restored; the checkpoint also carries the
    Adam slots and the epoch counter, which are not needed to reproduce a forward pass.
    '''
    tf.train.Saver(tf.trainable_variables()).restore(session, args.checkpoint)

    random = np.random.RandomState(args.seed)
    obs = random.randint(0, 256, size=(batch_num, 84, 84, 6)).astype(np.uint8)
    vels = random.normal(size=(batch_num, 8)).astype(np.float32)
    states = random.normal(size=np.shape(initial_state)).astype(np.float32)
    '''
    The mask restarts the recurrent state, which is what the training loop feeds from the
    previous step's done flags; exercise both values so the port cannot pass by ignoring
    it.
    '''
    dones = (np.arange(batch_num) % 2 == 0)

    feed = {obs_ph: obs, vels_ph: vels, states_ph: states, dones_ph: dones}

    graph = tf.get_default_graph()
    wanted = [operation.name for operation in graph.get_operations()
              if any(re.match(pattern, operation.name) for pattern in INTERMEDIATE_PATTERNS)]
    intermediates = {name: graph.get_tensor_by_name(name + ':0') for name in wanted}

    outputs = session.run(
        dict(logits=logits, value=value, lstm_state=lstm_state, **intermediates), feed)

    dump = {'input/obs': obs, 'input/vels': vels, 'input/states': states,
            'input/dones': dones}
    for name, array in outputs.items():
        dump['output/' + name] = array
    for variable in tf.trainable_variables():
        dump['weight/' + variable.name.split(':')[0]] = session.run(variable)

    np.savez_compressed(args.output, **dump)
    print('wrote %s' % args.output)
    print('  weights: %d tensors' % sum(1 for k in dump if k.startswith('weight/')))
    print('  intermediates: %d tensors' % len(wanted))
    print('  logits[0]: %s' % np.array2string(outputs['logits'][0], precision=6))
    print('  value[0]: %s' % np.array2string(np.reshape(outputs['value'], (-1,))[0],
                                             precision=6))


if __name__ == '__main__':
    main()
