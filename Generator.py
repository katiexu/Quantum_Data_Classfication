import pennylane as qml
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

from Arguments import Arguments
args = Arguments()


np.random.seed(args.seed)
torch.manual_seed(args.seed)


dev = qml.device("default.qubit", wires=args.n_qubits)

@qml.qnode(dev)
def HWE_qnn(params, depth):
    # Parameters of the first column of U3 gates are sampled from a uniform distribution [-1, 1]
    params_initialize = 2 * np.random.rand(args.n_qubits, 3) - 1
    for i in range(args.n_qubits):
        qml.U3(params_initialize[i][0], params_initialize[i][1], params_initialize[i][2], wires=i)
    for d in range(depth):
        for i1 in range(args.n_qubits):
            qml.U3(params[d][i1][0], params[d][i1][1], params[d][i1][2], wires=i1)
        for i2 in range(math.floor(args.n_qubits / 2)):
            qml.CZ(wires=[2 * i2, 2 * i2 + 1])
        for i3 in range(args.n_qubits):
            qml.U3(params[d][i3][0], params[d][i3][1], params[d][i3][2], wires=i3)
        for i4 in range(math.floor((args.n_qubits - 1) / 2)):
            qml.CZ(wires=[2 * i4 + 1, 2 * i4 + 2])
    return qml.state()

@qml.qnode(dev)
def DL_HWE_qnn(params, depth):
    # Parameters of the first column of U3 gates are sampled from a uniform distribution [-1, 1]
    params_initialize = 2 * np.random.rand(args.n_qubits, 3) - 1
    for i in range(args.n_qubits):
        qml.U3(params_initialize[i][0], params_initialize[i][1], params_initialize[i][2], wires=i)
    for d in range(depth):
        for i1 in range(math.floor(args.n_qubits / 2)):
            qml.CNOT(wires=[2 * i1, 2 * i1 + 1])
        for i2 in range(math.floor((args.n_qubits - 1) / 2)):
            qml.CNOT(wires=[2 * i2 + 1, 2 * i2 + 2])
        for i3 in range(args.n_qubits):
            qml.U3(params[d][i3][0], params[d][i3][1], params[d][i3][2], wires=i3)
    return qml.state()


# Function to calculate Concentratable Entanglement (CE)
def concentratable_entanglement(state, n_qubits):
    # Density matrix
    rho = qml.math.outer(state, qml.math.conj(state))

    # Reshape density matrix to n qubits
    dim = 2 ** n_qubits
    rho = rho.reshape((dim, dim))

    # Initialize the sum
    trace_sum = 0

    # Iterate over all possible subsystems (pairs of qubits)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Compute the partial trace over the selected qubits
            reduced_rho = np.einsum(args.einsum, rho.reshape([2] * 2 * n_qubits))
            reduced_rho = reduced_rho.reshape(args.reduced_rho_reshape_size, args.reduced_rho_reshape_size)

            # Compute the trace of the square of the reduced density matrix
            trace_sum += np.trace(reduced_rho @ reduced_rho)

    # Calculate CE
    ce = 1 - (1 / (2 ** n_qubits)) * trace_sum
    return ce


# Generate data
def generate_data(num_samples, label):
    data = []
    labels = []
    # params = 2 * math.pi * np.random.rand(depth, args.n_qubits, 3)
    if label == args.label0:
        params = args.params0
    else:
        params = args.params1
    for _ in range(num_samples):
        if args.task == 'Entangled State Classification':
            depth = args.depth
            state = HWE_qnn(params, depth)
        else:
            depth = label
            state = DL_HWE_qnn(params, depth)
        # Duplicate the state to create m copies
        if args.m == 2:
            state = np.kron(state, state)
        data.append(state)
        labels.append(label)
    return data, labels


data0, labels0 = generate_data(args.num_data0, args.label0)
data1, labels1 = generate_data(args.num_data1, args.label1)

# Combine data and split into training and testing sets
data = np.array(data0 + data1)
labels = (np.array(labels0 + labels1) == args.label1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).unsqueeze(1)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).unsqueeze(1)
torch.save(X_train, 'data/X_train.pt')
torch.save(y_train, 'data/y_train.pt')
torch.save(X_test, 'data/X_test.pt')
torch.save(y_test, 'data/y_test.pt')


# # Parameters for training
# params = args.params6
#
# # Get the output state of the QNN
# output_state = qnn(params)
# print(f"Output State: {output_state}")
#
# # Calculate CE value
# ce = concentratable_entanglement(output_state, args.n_qubits)
# print(f"Concentratable Entanglement (CE): {ce:.4f}")

# # Draw the circuit
# depth = 1
# params = np.random.random((depth, args.n_qubits, 3))
# fig = qml.draw_mpl(HWE_qnn)(params, depth)
# # fig = qml.draw_mpl(DL_HWE_qnn)(params, depth)
# plt.show()
