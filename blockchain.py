import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load transaction data from CSV (replace 'your_transaction_data.csv' with actual file path)
transaction_data = pd.read_csv('your_transaction_data.csv')

# Create a directed graph
G = nx.DiGraph()

# Add nodes for entities
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
    G.add_node(row["From"], type="Address", **{"Unique Identifier": row["From"]})
    G.add_edge(row["From"], row["Txhash"], label="Sent", property="Amount Transferred")
    # Add receiver address node and edge
    G.add_node(row["To"], type="Address", **{"Unique Identifier": row["To"]})
    G.add_edge(row["Txhash"], row["To"], label="Received", property="Amount Transferred")

# Draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=[G.nodes[n]['color'] for n in G.nodes], node_size=2000, font_size=10)
edge_labels = {(source, target): data['label'] for source, target, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

# Show the graph
plt.show()
