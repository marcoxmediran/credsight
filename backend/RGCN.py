import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc, classification_report
)
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# FILE PATHS
MODEL_PATH = 'RGCN_files/rgcn_fraud_model_unsorted.pt'
PREPROCESSING_ARTIFACTS_PATH = 'RGCN_files/preprocessing_artifacts.pkl'
GRAPH_METADATA_PATH = 'RGCN_files/graph_metadata.pkl'
CSV_PATH = 'RGCN_files/demo.csv'

# LOAD ARTIFACTS

# Model
model_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
print(f"Model state dictionary loaded from '{MODEL_PATH}'")

# Graph Metadata
with open(GRAPH_METADATA_PATH, 'rb') as f:
    graph_metadata = pickle.load(f)
card_to_idx = graph_metadata['card_to_idx']
email_to_idx = graph_metadata['email_to_idx']
transaction_feature_cols = graph_metadata['transaction_feature_cols']
print(f"Graph metadata loaded from '{GRAPH_METADATA_PATH}'")

# Preprocessing Artifacts
with open(PREPROCESSING_ARTIFACTS_PATH, 'rb') as f:
    artifacts = pickle.load(f)
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
train_medians = artifacts['train_medians']
train_modes = artifacts['train_modes']
numerical_cols = artifacts['numerical_cols']
categorical_cols = artifacts['categorical_cols']
print(f"Preprocessing artifacts loaded from '{PREPROCESSING_ARTIFACTS_PATH}'")

def analyze_transactions(csv_data: str):
    """
    Analyze transactions using RGCN model.
    
    Args:
        csv_data: CSV data as string containing transaction data
        
    Returns:
        Dictionary containing predictions and metrics
    """
    # Load CSV data from string
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    
    # EXTRA DATA PROCESSING
    # Define required features 
    REQUIRED_FEATURES = [
        'isFraud', 'C14', 'C1', 'V317', 'V308', 'card6', 'V306', 'V126',
        'C11', 'C6', 'V282', 'TransactionAmt', 'V53', 'P_emaildomain',
        'card5', 'C13', 'V35', 'V280', 'V279', 'card2', 'V258', 'M6',
        'V90', 'V82', 'TransactionDT', 'C2', 'V87', 'V294', 'C12', 'V313', 'id_06'
    ]

    # Ensure TransactionID is included
    if 'TransactionID' not in df.columns:
        raise ValueError("CSV must contain TransactionID column")
    
    # Select required columns in the correct order
    available_features = [f for f in REQUIRED_FEATURES if f in df.columns]
    df = df[['TransactionID'] + available_features]

    # Identify column types
    categorical_cols = ['card6', 'P_emaildomain', 'card5', 'M6', 'card2', 'id_06']
    numerical_cols = [col for col in REQUIRED_FEATURES
                      if col not in categorical_cols + ['isFraud', 'TransactionDT']]

    # Fill Missing Values (using training medians/modes)
    for col in numerical_cols:
        df[col].fillna(train_medians[col], inplace=True)
    for col in categorical_cols:
        df[col].fillna(train_modes[col], inplace=True)
    print("Missing values filled using pre-calculated statistics.")

    # Scale Numerical Features (using training scaler's transform method)
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    print("Numerical features scaled using pre-fitted scaler.")

    # GRAPH CONSTRUCTION
    # Transaction features (numerical features only, exclude identifiers and target)
    transaction_feature_cols = numerical_cols
    print(f"Transaction node features ({len(transaction_feature_cols)} features)")

    # Create composite card key (card2_card5_card6)
    df['card_key'] = (
        df['card2'].astype(str) + '_' +
        df['card5'].astype(str) + '_' +
        df['card6'].astype(str)
    )

    # Create unique node mappings
    unique_cards = df['card_key'].unique()
    unique_emails = df['P_emaildomain'].unique()

    card_to_idx = {card: idx for idx, card in enumerate(unique_cards)}
    email_to_idx = {email: idx for idx, email in enumerate(unique_emails)}

    print(f"\nGraph statistics:")
    print(f"  Transaction nodes: {len(df)}")
    print(f"  Card nodes: {len(unique_cards)}")
    print(f"  Email nodes: {len(unique_emails)}")

    # Map to indices
    df['card_idx'] = df['card_key'].map(card_to_idx)
    df['email_idx'] = df['P_emaildomain'].map(email_to_idx)

    # Create HeteroData object
    data = HeteroData()

    # Transaction node features
    transaction_features = torch.tensor(
        df[transaction_feature_cols].values,
        dtype=torch.float
    )
    data['transaction'].x = transaction_features
    
    # Handle ground truth labels - use zeros if isFraud contains NaN
    if 'isFraud' in df.columns and not df['isFraud'].isna().all():
        labels = df['isFraud'].fillna(0).values  # Fill NaN with 0 for safety
    else:
        labels = np.zeros(len(df))  # No ground truth available
    
    data['transaction'].y = torch.tensor(labels, dtype=torch.float)
    data['transaction'].transaction_id = torch.tensor(df['TransactionID'].values, dtype=torch.long)

    print(f"Transaction features shape: {data['transaction'].x.shape}")

    # Card node features (initialize with mean of connected transactions, or zeros)
    # For simplicity, we'll use learnable embeddings or zero initialization
    data['card'].num_nodes = len(unique_cards)

    # Email node features
    data['email'].num_nodes = len(unique_emails)

    # Create edges: (transaction, uses_card, card)
    transaction_indices = torch.arange(len(df))
    card_indices = torch.tensor(df['card_idx'].values, dtype=torch.long)

    data['transaction', 'uses_card', 'card'].edge_index = torch.stack([
        transaction_indices,
        card_indices
    ], dim=0)

    # Create edges: (transaction, has_email, email)
    email_indices = torch.tensor(df['email_idx'].values, dtype=torch.long)

    data['transaction', 'has_email', 'email'].edge_index = torch.stack([
        transaction_indices,
        email_indices
    ], dim=0)

    # Reverse edges for message passing
    data['card', 'used_by', 'transaction'].edge_index = torch.stack([
        card_indices,
        transaction_indices
    ], dim=0)

    data['email', 'belongs_to', 'transaction'].edge_index = torch.stack([
        email_indices,
        transaction_indices
    ], dim=0)

    print(f"\nEdge statistics:")
    print(f"  (transaction, uses_card, card): {data['transaction', 'uses_card', 'card'].edge_index.shape[1]} edges")
    print(f"  (transaction, has_email, email): {data['transaction', 'has_email', 'email'].edge_index.shape[1]} edges")
    print(f"  (card, used_by, transaction): {data['card', 'used_by', 'transaction'].edge_index.shape[1]} edges")
    print(f"  (email, belongs_to, transaction): {data['email', 'belongs_to', 'transaction'].edge_index.shape[1]} edges")

    # Move data to DEVICE
    data = data.to(DEVICE)

    # GET PREDICTIONS
    # Create a mask to evaluate all transactions in the input data
    n_total = data['transaction'].num_nodes
    all_nodes_mask = torch.ones(n_total, dtype=torch.bool)
    data['transaction'].test_mask = all_nodes_mask

    # Run evaluation
    test_loss, test_f1, test_recall, test_auc, test_probs, test_preds, test_labels = evaluate(
        data, data['transaction'].test_mask
    )
    print("Predictions generated for all nodes in the input data.")

    # Create results DataFrame
    predictions_df = pd.DataFrame({
        'TransactionID': data['transaction'].transaction_id.cpu().numpy(),
        'Actual_isFraud': test_labels.astype(int),
        'Predicted_isFraud': test_preds,
        'Fraud_Probability': test_probs
    })

    # Return results in the format expected by the API
    results = {
        'transactions': predictions_df.to_dict('records'),
        'metrics': {
            'loss': test_loss,
            'f1': test_f1,
            'recall': test_recall,
            'auc': test_auc
        },
        'summary': {
            'total_transactions': len(predictions_df),
            'fraud_detected_by_RGCN': sum(test_preds),
            'legitimate_detected_by_RGCN': len(test_preds) - sum(test_preds),
            'fraud_rate': (sum(test_labels) / len(test_labels)) * 100 if len(test_labels) > 0 else 0
        }
    }
    
    return results

# DEFINE MODEL
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_transaction_features):
        super().__init__()

        # Transaction node encoder
        self.transaction_lin = Linear(num_transaction_features, hidden_channels)

        # Card and Email node embeddings (learnable)
        # We'll initialize these and update them through message passing

        # First HeteroConv layer
        self.conv1 = HeteroConv({
            ('transaction', 'uses_card', 'card'): SAGEConv(hidden_channels, hidden_channels),
            ('card', 'used_by', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
            ('transaction', 'has_email', 'email'): SAGEConv(hidden_channels, hidden_channels),
            ('email', 'belongs_to', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # Second HeteroConv layer
        self.conv2 = HeteroConv({
            ('transaction', 'uses_card', 'card'): SAGEConv(hidden_channels, hidden_channels),
            ('card', 'used_by', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
            ('transaction', 'has_email', 'email'): SAGEConv(hidden_channels, hidden_channels),
            ('email', 'belongs_to', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # Classifier head for transaction nodes
        self.classifier = Linear(hidden_channels, out_channels)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        # Encode transaction features
        x_dict['transaction'] = self.transaction_lin(x_dict['transaction'])

        # Initialize card and email features with zeros
        # Get unique node counts from edge indices
        card_nodes = edge_index_dict[('transaction', 'uses_card', 'card')][1].max().item() + 1
        email_nodes = edge_index_dict[('transaction', 'has_email', 'email')][1].max().item() + 1
        
        x_dict['card'] = torch.zeros(
            (card_nodes, x_dict['transaction'].size(1)),
            device=x_dict['transaction'].device
        )
        x_dict['email'] = torch.zeros(
            (email_nodes, x_dict['transaction'].size(1)),
            device=x_dict['transaction'].device
        )

        # First conv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Second conv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Classify transaction nodes only
        out = self.classifier(x_dict['transaction'])

        return out.squeeze(-1)

# Initialize model
model = HeteroGNN(
    hidden_channels=128,
    out_channels=1,
    num_transaction_features=len(transaction_feature_cols)
).to(DEVICE)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# EVAL METHOD
# Define loss function, required by the evaluate function
criterion = nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(data, mask):
    model.eval()

    out = model(data.x_dict, data.edge_index_dict)

    # Get predictions for the specified mask
    logits = out[mask]
    labels = data['transaction'].y[mask]

    # Loss
    loss = criterion(logits, labels).item()

    # Convert to probabilities
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Predictions (threshold 0.5 as specified)
    preds = (probs >= 0.5).astype(int)

    # Metrics
    f1 = f1_score(labels_np, preds)
    recall = recall_score(labels_np, preds)

    # AUC (only if both classes present)
    if len(np.unique(labels_np)) > 1:
        auc_score = roc_auc_score(labels_np, probs)
    else:
        auc_score = 0.0

    return loss, f1, recall, auc_score, probs, preds, labels_np

# LOAD MODEL
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Pre-trained model weights loaded successfully.")

# Example usage (commented out for production)
if __name__ == "__main__":
    # Load demo data for testing
    with open('RGCN_files/demo.csv', 'r') as f:
        demo_csv = f.read()
    results = analyze_transactions(demo_csv)
    
    metrics = results.get("metrics", {})
    f1 = metrics.get("f1")
    recall = metrics.get("recall")
    auc = metrics.get("auc")

    print("Analysis complete!")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"AUC: {auc}")

    transactions = results.get("transactions", [])
    num_pred_fraud = sum(1 for tx in transactions if tx["Predicted_isFraud"] == 1)
    num_pred_legit = sum(1 for tx in transactions if tx["Predicted_isFraud"] == 0)
    print(f"Predicted Fraudulent transactions: {num_pred_fraud}")
    print(f"Predicted Legitimate transactions: {num_pred_legit}")