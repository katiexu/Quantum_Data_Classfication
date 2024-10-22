import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

from Arguments import Arguments
args = Arguments()


np.random.seed(args.seed)
torch.manual_seed(args.seed)

X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_test = torch.load('data/X_test.pt')
y_test = torch.load('data/y_test.pt')


# Define device with wires equal to args.n_qubits * m
m = args.m
dev = qml.device("default.qubit", wires=args.n_qubits * m)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def qcnn(params, inputs):
    # Initialize the input state with m copies
    qml.QubitStateVector(inputs, wires=range(args.n_qubits * m))
    # Apply RX and RY gates
    for i in range(args.n_qubits * m):
        qml.RX(params[i], wires=i)
        qml.RY(params[i + args.n_qubits * m], wires=i)
    # Apply convolution and pooling layers
    for i in range(args.n_qubits * m // 2):
        qml.CNOT(wires=[2 * i, 2 * i + 1])
        qml.Hadamard(wires=2 * i)

    return [qml.expval(qml.PauliZ(i)) for i in range(args.n_qubits * m)]


class Classifier(nn.Module):
    def __init__(self, n_qubits):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_qubits * m, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        weight_shapes = {"params": (2 * args.n_qubits * m,)}
        self.qlayer = qml.qnn.TorchLayer(qcnn, weight_shapes)
        self.classifier = Classifier(args.n_qubits)

    def forward(self, x):
        x = self.qlayer(x)
        x = self.classifier(x.real.float())
        return x


model = HybridNet()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Define loss function and training function

qcnn_params = pnp.array(np.random.rand(2 * args.n_qubits * m), requires_grad=True)
for epoch in range(args.steps):
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train.float())

    loss.backward()
    optimizer.step()
    # print(f'Epoch: {epoch}/{args.steps}\tLoss: {loss.item():.4f}')
    test_output = model(X_test)
    pred = (test_output >= 0.5).float()
    accuracy = accuracy_score(y_test, pred.long())
    print(f'Epoch: {epoch}/{args.steps}\tLoss: {loss.item():.4f}\ttest Accuracy: {accuracy:.4f}')

test_output = model(X_test)
pred = (test_output >= 0.5).float()
accuracy = accuracy_score(y_test, pred.long())
print(f'Test Accuracy: {accuracy:.4f}')
