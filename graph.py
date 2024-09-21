import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

print("Current working directory:", os.getcwd())

# Change the current working directory
os.chdir('C:/Users/ADMIN/Desktop/BLOCKCHAIN')

# Load transaction data from CSV (replace 'your_transaction_data.csv' with actual file path)
transaction_data = pd.read_csv('your_transaction_data.csv')

# Create a directed graph
G = nx.DiGraph()

# Add nodes for entities with default colors
G.add_node("Transaction", color='blue')
G.add_node("Smart Contract", color='green')
G.add_node("Address", color='red')

# Define attributes for each entity
transaction_attributes = [
    "Txhash", "Method", "Blockno", "DateTime (UTC)", 
    "From", "From_Nametag", "To", "To_Nametag", 
    "Value", "Txn Fee"
]
smart_contract_attributes = [
    "Txhash", "Method", "Blockno", "DateTime (UTC)", 
    "From", "From_Nametag", "To", "To_Nametag", 
    "Value", "Txn Fee"
]
address_attributes = ["Unique Identifier"]

# Add attributes to nodes
for attr in transaction_attributes:
    G.nodes["Transaction"][attr] = None

for attr in smart_contract_attributes:
    G.nodes["Smart Contract"][attr] = None

for attr in address_attributes:
    G.nodes["Address"][attr] = None

# Define relationships
relationships = [
    ("Transaction", "Address", {"label": "Sent/Received", "property": "Amount Transferred"}),
    ("Transaction", "Smart Contract", {"label": "Interaction", "property": "Type of Interaction"}),
    ("Smart Contract", "Address", {"label": "Sent/Received", "property": "Amount Transferred"}),
    ("Transaction", "Transaction", {"label": "Related", "property": "Type of Relation"})
]

# Add relationships to the graph
for source, target, data in relationships:
    G.add_edge(source, target, **data)

# Add nodes and edges based on transaction data
for _, row in transaction_data.iterrows():
    # Add transaction node
    G.add_node(row["Txhash"], type="Transaction")
    # Add sender address node and edge
    G.add_node(row["From"], type="Address", color='red', **{"Unique Identifier": row["From"]})
    G.add_edge(row["From"], row["Txhash"], label="Sent", property="Amount Transferred")
    # Add receiver address node and edge
    G.add_node(row["To"], type="Address", color='red', **{"Unique Identifier": row["To"]})
    G.add_edge(row["Txhash"], row["To"], label="Received", property="Amount Transferred")

# Calculate node degree
node_degree = dict(G.degree())

# Compute centrality measures
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Extract transaction amounts and frequencies
transaction_amounts = {}
transaction_frequencies = {}
for node in G.nodes():
    if 'type' in G.nodes[node] and G.nodes[node]['type'] == 'Transaction':
        transaction_amounts[node] = transaction_data.loc[transaction_data['Txhash'] == node, 'Value'].values[0]
        transaction_frequencies[node] = len(list(G.neighbors(node)))

# Add extracted features as node attributes
for node in G.nodes():
    G.nodes[node]['degree'] = node_degree.get(node, 0)
    G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
    G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
    G.nodes[node]['eigenvector_centrality'] = eigenvector_centrality.get(node, 0)
    if 'type' in G.nodes[node] and G.nodes[node]['type'] == 'Transaction':
        G.nodes[node]['transaction_amount'] = transaction_amounts.get(node, 0)
        G.nodes[node]['transaction_frequency'] = transaction_frequencies.get(node, 0)

# Draw the graph
pos = nx.spring_layout(G, seed=42)

# Get node colors with default to 'blue' if 'color' attribute is not present
node_colors = [G.nodes[n].get('color', 'blue') for n in G.nodes]

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=10)

edge_labels = {(source, target): data['label'] for source, target, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

# Show the graph
plt.show()
