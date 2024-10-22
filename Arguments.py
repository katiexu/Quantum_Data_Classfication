import numpy as np

class Arguments:
    def __init__(self):
        self.task = 'Ansatz Depth Classification'     # 'Entangled State Classification' or 'Ansatz Depth Classification'
        self.seed = 42
        self.steps = 200

        self.n_qubits = 4
        self.m = 2      # input m-copies of each state

        self.num_data0 = 400
        self.num_data1 = 400

        self.label0 = 1
        self.label1 = 6
        self.params0 = np.load('Hardware_Efficient/4_Qubits/Depth_1/hwe_4q_ps_25_1_weights.npy')
        self.params1 = np.load('Hardware_Efficient/4_Qubits/Depth_6/hwe_4q_ps_25_6_weights.npy')
        self.depth = 6

        self.einsum = 'ijklmnop->ikmo'  # 'ijklmnop->ikmo' for 4 qubits; 'abcdefghijklmnop->acegikmo' for 8 qubits
        self.reduced_rho_reshape_size = 4          # 4 for 4 qubits; 16 for 8 qubits