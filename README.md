# Ansatz_Depth_Classfication

## Requirements
python==3.9
pennylane==0.34.0
torch==2.1.2
qiskit==0.38.0
torchquantum==0.1.7
scikit-learn
opt_einsum

## Run Code
1) Set configs in **Arguments.py**.
2) Run **Generator.py** to generate entangled data.
3) Run **ClassifyingQNN.py** to train the model and to obtain test accuracy using PennyLane framework.
4) Run **tq_ClassifyingQNN.py** to train the model and to obtain test accuracy using TorchQuantum framework.
