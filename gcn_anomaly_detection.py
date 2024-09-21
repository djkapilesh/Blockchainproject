import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Change the current working directory to the 'BLOCKCHAIN' directory
os.chdir('C:/Users/ADMIN/Desktop/BLOCKCHAIN')
print("Current working directory:", os.getcwd())

# Convert the heterogeneous graph to PyTorch Geometric data
def graph_to_pyg_data(graph):
    # Convert the NetworkX graph to a PyTorch Geometric Data object
    node_features = torch.tensor([list(node.values()) for node in graph.nodes.values()], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

# Compute anomaly scores using degree centrality
def compute_anomaly_scores(G, node_embeddings):
    anomaly_scores = {node: G.degree(node) for node in G.nodes()}
    return anomaly_scores

# Detect anomalies based on anomaly scores
def detect_anomalies(anomaly_scores, threshold):
    anomalies = [node for node, score in anomaly_scores.items() if score > threshold]
    return anomalies

# Load transaction data from CSV (replace 'your_transaction_data.csv' with actual file path)
file_path = 'C:/Users/ADMIN/Desktop/BLOCKCHAIN/your_transaction_data.csv'
transaction_data = pd.read_csv(file_path)

# Create the transaction graph
G = nx.DiGraph()
# Populate G with your transaction data...

# Convert the graph to PyTorch Geometric data
pyg_data = graph_to_pyg_data(G)

# Compute anomaly scores using degree centrality
anomaly_scores = compute_anomaly_scores(G, None)  # Placeholder for node embeddings

# Set threshold for anomaly detection
threshold = 2  # Adjust threshold as needed

# Detect anomalies based on anomaly scores
anomalies = detect_anomalies(anomaly_scores, threshold)
print("Detected Anomalies:", anomalies)

# Define a simple Graph Convolutional Network (GCN) model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Ensure that node features have correct dimensionality
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Check if edge_index is empty
        if edge_index.numel() == 0:
            return x  # Return node features without convolution
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the GCN model
model = GCN(input_dim=pyg_data.num_node_features, hidden_dim=16, output_dim=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
optimizer.zero_grad()
out = model(pyg_data).view(-1)
labels = torch.zeros_like(out, requires_grad=True)  # Set requires_grad=True for labels
loss = F.binary_cross_entropy_with_logits(out, labels.float())
loss.backward()
optimizer.step()

print("GCN model training completed.")
