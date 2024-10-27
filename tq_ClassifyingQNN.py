import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np

from Arguments import Arguments

args = Arguments()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

X_train = torch.load('data/X_train.pt').to(dtype=torch.complex64)
y_train = torch.load('data/y_train.pt')
X_test = torch.load('data/X_test.pt').to(dtype=torch.complex64)
y_test = torch.load('data/y_test.pt')


class QuantumCircuit(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_qubits = args.n_qubits * args.m
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.rxs = [tq.RX(has_params=True, trainable=True) for i in range(self.n_qubits)]
        self.rys = [tq.RY(has_params=True, trainable=True) for i in range(self.n_qubits)]
        self.cnots = [tq.CNOT() for _ in range(self.n_qubits // 2)]
        self.hadamards = [tq.Hadamard() for _ in range(self.n_qubits // 2)]
        self.measure = tq.MeasureAll(tq.PauliZ)


    def forward(self, x):
        self.q_device.reset_states(bsz=x.shape[0])
        self.q_device.set_states(x)

        # Apply RX and RY gates
        for i in range(self.n_qubits):
            self.rxs[i](self.q_device, wires=i)
            self.rys[i](self.q_device, wires=i)

        # Apply CNOT and Hadamard as convolution and pooling
        for i in range(self.n_qubits // 2):
            self.cnots[i](self.q_device, wires=[2 * i, 2 * i + 1])
            self.hadamards[i](self.q_device, wires=[2 * i])

        return self.measure(self.q_device)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(args.n_qubits * args.m, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        self.qcircuit = QuantumCircuit()
        self.classifier = Classifier()

    def forward(self, x):
        quantum_out = self.qcircuit(x)
        return self.classifier(quantum_out)


model = HybridNet()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(args.steps):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train.float())
    loss.backward()
    optimizer.step()

    test_output = model(X_test)
    pred = (test_output >= 0.5).float()
    accuracy = accuracy_score(y_test, pred.long())
    print(f'Epoch: {epoch}/{args.steps}\tLoss: {loss.item():.4f}\tTest Accuracy: {accuracy:.4f}')

# Final test accuracy
test_output = model(X_test)
pred = (test_output >= 0.5).float()
accuracy = accuracy_score(y_test, pred.long())
print(f'Test Accuracy: {accuracy:.4f}')