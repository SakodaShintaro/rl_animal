"""Check torch_agent against the TensorFlow forward pass export_tf_reference.py dumped.

    uv run --project pyproject.toml python verify_torch_parity.py \
        --reference local/tf_reference.npz --env-num 2 --steps-num 1

Bit-identical output is not attainable: cuBLAS and cuDNN pick their own reduction orders,
so even the same TensorFlow build disagrees with itself across GPUs at the last few bits
of a float32. What is attainable is agreement to the precision of the arithmetic, and the
way to tell a port bug from arithmetic noise is to run the port in float64: a structurally
identical network lands within about 1e-6 of the float32 reference, while a wrong padding
alignment or a transposed flatten is off by orders of magnitude.
"""
import argparse

import numpy as np
import torch

from rl_animal_torch import network as torch_agent

'''
The float32 reference carries roughly 1e-7 of relative error per operation and this
network is about twenty deep, so a correct port sits near 1e-6 on the logits. A failure
from a structural mistake is never this small.
'''
TOLERANCE = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='npz from export_tf_reference')
    parser.add_argument('--env-num', required=True, type=int)
    parser.add_argument('--steps-num', required=True, type=int)
    return parser.parse_args()


def compare(label, reference, ported):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    ported = np.asarray(ported, dtype=np.float64).reshape(-1)
    assert reference.shape == ported.shape, (label, reference.shape, ported.shape)
    absolute = np.max(np.abs(reference - ported))
    scale = np.maximum(np.max(np.abs(reference)), 1e-12)
    print('%-28s max abs %.3e   max abs / scale %.3e   %s' % (
        label, absolute, absolute / scale, 'ok' if absolute / scale < TOLERANCE else 'MISMATCH'))
    return absolute / scale


def main():
    args = parse_args()
    dump = np.load(args.reference)

    visual = torch.as_tensor(dump['input/obs'].astype(np.int64))
    dones = torch.as_tensor(dump['input/dones'].astype(np.float64))

    worst = 0.0
    for dtype, name in ((torch.float64, 'float64'), (torch.float32, 'float32')):
        agent = torch_agent.load_from_reference(args.reference, dtype=dtype)
        vels = torch.as_tensor(dump['input/vels']).to(dtype)
        state = torch.as_tensor(dump['input/states']).to(dtype)
        with torch.no_grad():
            logits, value, lstm_state = agent(visual, vels, state, dones.to(dtype),
                                              args.env_num)

        print('--- port in %s against the float32 reference' % name)
        worst = max(worst, compare('logits', dump['output/logits'], logits.numpy()))
        worst = max(worst, compare('value', dump['output/value'], value.numpy()))
        worst = max(worst, compare('lstm_state', dump['output/lstm_state'],
                                   lstm_state.numpy()))

    print()
    print('worst relative difference: %.3e (tolerance %.0e)' % (worst, TOLERANCE))
    if worst >= TOLERANCE:
        raise SystemExit('the port does not match')
    print('the port matches')


if __name__ == '__main__':
    main()
