"""The Animal-AI Olympics winning agent, in PyTorch, against the Animal-AI v4 build.

The network is a port of networks.animal_a2c_network_lstm6 and is numerically the same:
verify_torch_parity.py checks it against a forward pass exported from the TensorFlow 1.15
checkpoint and agrees to a relative 7e-07, which is the precision of a float32 reference.
"""
