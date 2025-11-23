import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default model path
DEFAULT_MODEL_PATH = 'RGCN_files/rgcn_fraud_model.pt'

## Define Model
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_transaction_features, num_card_nodes, num_email_nodes):
        super().__init__()
        self.num_card_nodes = num_card_nodes
        self.num_email_nodes = num_email_nodes

        # Transaction node encoder
        self.transaction_lin = Linear(num_transaction_features, hidden_channels)

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
        x_dict['card'] = torch.zeros(
            (self.num_card_nodes, x_dict['transaction'].size(1)),
            device=x_dict['transaction'].device
        )
        x_dict['email'] = torch.zeros(
            (self.num_email_nodes, x_dict['transaction'].size(1)),
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

# Define loss function
criterion = nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model on given data and mask"""
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

def analyze_transactions(csv_data: str, model_path: str = DEFAULT_MODEL_PATH):
    """
    Analyze transactions using RGCN model.
    
    Args:
        csv_data: CSV data as string containing transaction data
        model_path: Path to the trained model file
        
    Returns:
        Dictionary containing predictions and metrics
    """
    print(f"Using device: {DEVICE}")
    
    # Load CSV data from string
    df = pd.read_csv(StringIO(csv_data))
    print(f"Data Shape: {df.shape}")
    
    # Identify column types
    feature_cols = [col for col in df.columns if col not in ['TransactionID', 'isFraud', 'TrueLabel']]
    categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']
    numerical_cols = [col for col in feature_cols if df[col].dtype != 'object']
    
    # Fill Missing Values (using training medians/modes)
    col_medians = df[numerical_cols].median()
    for col in numerical_cols:
        df[col].fillna(col_medians[col], inplace=True)
    for col in categorical_cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna('missing', inplace=True)
    print("Missing values filled using pre-calculated statistics.")
    
    # Label Encode Categorical Features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale Numerical Features
    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    ## Graph construction
    print("\n" + "="*80)
    print("GRAPH CONSTRUCTION")
    print("="*80)
    
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
    
    # Card and Email nodes
    num_card_nodes = len(unique_cards)
    num_email_nodes = len(unique_emails)
    data['card'].num_nodes = num_card_nodes
    data['email'].num_nodes = num_email_nodes
    
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
    
    # Initialize model
    model = HeteroGNN(
        hidden_channels=128,
        out_channels=1,
        num_transaction_features=len(transaction_feature_cols),
        num_card_nodes=num_card_nodes,
        num_email_nodes=num_email_nodes
    ).to(DEVICE)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load Model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Pre-trained model weights loaded successfully.")
    
    # Move data to device
    data = data.to(DEVICE)
    
    # Get Predictions
    # Create a mask to evaluate all transactions in the input data
    n_total = data['transaction'].num_nodes
    all_nodes_mask = torch.ones(n_total, dtype=torch.bool, device=DEVICE)
    data['transaction'].test_mask = all_nodes_mask
    
    # Run evaluation
    test_loss, test_f1, test_recall, test_auc, test_probs, test_preds, test_labels = evaluate(
        model, data, data['transaction'].test_mask
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

if __name__ == "__main__":
    # Standalone execution for testing
    CSV_PATH = 'ERGCN_files/merged_demo.csv'
    MODEL_PATH = 'RGCN_files/rgcn_fraud_model.pt'
    
    print("\n" + "="*80)
    print("STANDALONE RGCN MODEL EXECUTION")
    print("="*80)
    
    # Load demo data
    with open(CSV_PATH, 'r') as f:
        demo_csv = f.read()
    
    # Run analysis
    results = analyze_transactions(demo_csv, MODEL_PATH)
    
    # Print results
    metrics = results.get("metrics", {})
    f1 = metrics.get("f1")
    recall = metrics.get("recall")
    auc = metrics.get("auc")
    loss = metrics.get("loss")
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Performance Metrics:")
    print(f"  Loss:   {loss:.5f}")
    print(f"  F1:     {f1:.5f}")
    print(f"  Recall: {recall:.5f}")
    print(f"  AUC:    {auc:.5f}")
    
    summary = results.get("summary", {})
    print(f"\nSummary:")
    print(f"  Total transactions: {summary.get('total_transactions', 0)}")
    print(f"  Fraud detected: {summary.get('fraud_detected_by_RGCN', 0)}")
    print(f"  Legitimate detected: {summary.get('legitimate_detected_by_RGCN', 0)}")
    print(f"  Fraud rate: {summary.get('fraud_rate', 0):.2f}%")
    
    # transactions = results.get("transactions", [])
    # if transactions:
    #     print(f"\nFirst 10 predictions:")
    #     for i, tx in enumerate(transactions[:10]):
    #         print(f"  TransactionID {tx['TransactionID']}: Predicted={tx['Predicted_isFraud']}, "
    #               f"Probability={tx['Fraud_Probability']:.4f}, Actual={tx['Actual_isFraud']}")
