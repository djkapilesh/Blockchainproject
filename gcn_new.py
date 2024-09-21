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

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Load transaction data from CSV
file_path = 'C:/Users/ADMIN/Desktop/BLOCKCHAIN/your_transaction_data.csv'  # Updated file path
transaction_data = pd.read_csv(file_path)

# Perform data splitting
train_data, test_data = train_test_split(transaction_data, test_size=0.2, random_state=42)

# Continue with your existing code for model training and evaluation
# You can use train_data for training the model and test_data for evaluating its performance
# Make sure to incorporate the splitting and evaluation steps into your existing code


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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming 'predictions' are the predicted labels from the GCN model on the test data
predictions = model(test_data).view(-1).detach().numpy()

# Assuming 'true_labels' are the true labels from the test data
true_labels = torch.zeros_like(predictions, requires_grad=False).numpy()  # Assuming all labels are 0 for anomaly detection

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.model_selection import GridSearchCV
from torch.optim import Adam

# Define hyperparameters grid
param_grid = {
    'hidden_dim': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1],
}

# Define the GCN model
model = GCN(input_dim=pyg_data.num_node_features, output_dim=1).to(device)

# Define optimizer
optimizer = Adam(model.parameters())

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(train_data, train_labels)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

from sklearn.model_selection import cross_val_score

# Define the GCN model
model = GCN(input_dim=pyg_data.num_node_features, output_dim=1)

# Perform cross-validation
scores = cross_val_score(model, train_data, train_labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)

from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

# Compute precision-recall curve and area under the curve
precision, recall, _ = precision_recall_curve(true_labels, predictions)
pr_auc = auc(recall, precision)

print("ROC AUC:", roc_auc)
print("Precision-Recall AUC:", pr_auc)
