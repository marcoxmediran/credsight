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

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Default model path
DEFAULT_MODEL_PATH = 'ERGCN_files/ergcn_fraud_model_v1.pt'

"""# ERGCN Model Definition"""

class ERGCN(nn.Module):
    """
    Enhanced Relational Graph Convolutional Network with GRU
    Architecture based on the proposed system with Global and Local Learning Units
    """
    def __init__(self, hidden_channels, out_channels, num_transaction_features,
                 num_card_nodes, num_email_nodes, gru_hidden=64):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.gru_hidden = gru_hidden
        self.num_card_nodes = num_card_nodes
        self.num_email_nodes = num_email_nodes

        # Transaction node encoder
        self.transaction_lin = Linear(num_transaction_features, hidden_channels)

        # ===== GLOBAL LEARNING UNIT =====
        # Double RGCN layers for global learning
        self.global_conv1 = HeteroConv({
            ('transaction', 'uses_card', 'card'): SAGEConv(hidden_channels, hidden_channels),
            ('card', 'used_by', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
            ('transaction', 'has_email', 'email'): SAGEConv(hidden_channels, hidden_channels),
            ('email', 'belongs_to', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        self.global_conv2 = HeteroConv({
            ('transaction', 'uses_card', 'card'): SAGEConv(hidden_channels, hidden_channels),
            ('card', 'used_by', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
            ('transaction', 'has_email', 'email'): SAGEConv(hidden_channels, hidden_channels),
            ('email', 'belongs_to', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # Max pooling for global features
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # GRU for global temporal patterns
        self.global_gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Global feature projection
        self.global_fc = Linear(gru_hidden, hidden_channels // 2)

        # ===== LOCAL LEARNING UNIT =====
        # Single RGCN layer for local learning
        self.local_conv = HeteroConv({
            ('transaction', 'uses_card', 'card'): SAGEConv(hidden_channels, hidden_channels),
            ('card', 'used_by', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
            ('transaction', 'has_email', 'email'): SAGEConv(hidden_channels, hidden_channels),
            ('email', 'belongs_to', 'transaction'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # GRU for local temporal patterns
        self.local_gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )

        # Local feature projection
        self.local_fc = Linear(gru_hidden, hidden_channels // 2)

        # ===== OUTPUT LAYER =====
        # Combine global and local features
        self.combine_fc1 = Linear(hidden_channels, hidden_channels // 2)
        self.combine_fc2 = Linear(hidden_channels // 2, out_channels)

        # Dropout and batch norm
        self.dropout = nn.Dropout(0.5)
        self.bn_global = nn.BatchNorm1d(hidden_channels)
        self.bn_local = nn.BatchNorm1d(hidden_channels)
        self.bn_combine = nn.BatchNorm1d(hidden_channels // 2)

    def forward(self, x_dict, edge_index_dict):
        # Encode transaction features
        x_dict['transaction'] = self.transaction_lin(x_dict['transaction'])

        # Initialize card and email features with proper dimensions
        device = x_dict['transaction'].device
        x_dict['card'] = torch.zeros(
            (self.num_card_nodes, self.hidden_channels),
            device=device
        )
        x_dict['email'] = torch.zeros(
            (self.num_email_nodes, self.hidden_channels),
            device=device
        )

        # ===== GLOBAL LEARNING PATH =====
        # First global conv
        global_x_dict = {k: v.clone() for k, v in x_dict.items()}
        global_x_dict = self.global_conv1(global_x_dict, edge_index_dict)
        global_x_dict = {key: F.relu(x) for key, x in global_x_dict.items()}
        global_x_dict = {key: self.dropout(x) for key, x in global_x_dict.items()}

        # Second global conv
        global_x_dict = self.global_conv2(global_x_dict, edge_index_dict)
        global_x_dict = {key: F.relu(x) for key, x in global_x_dict.items()}

        # Get transaction embeddings and apply batch norm
        global_trans = global_x_dict['transaction']
        global_trans = self.bn_global(global_trans)

        # Reshape for GRU: (batch, seq_len=1, features)
        global_trans_seq = global_trans.unsqueeze(1)

        # Apply GRU
        global_gru_out, _ = self.global_gru(global_trans_seq)
        global_gru_out = global_gru_out[:, -1, :]  # Take last output

        # Project global features
        global_features = self.global_fc(global_gru_out)
        global_features = F.relu(global_features)
        global_features = self.dropout(global_features)

        # ===== LOCAL LEARNING PATH =====
        # Local conv
        local_x_dict = {k: v.clone() for k, v in x_dict.items()}
        local_x_dict = self.local_conv(local_x_dict, edge_index_dict)
        local_x_dict = {key: F.relu(x) for key, x in local_x_dict.items()}

        # Get transaction embeddings and apply batch norm
        local_trans = local_x_dict['transaction']
        local_trans = self.bn_local(local_trans)

        # Reshape for GRU
        local_trans_seq = local_trans.unsqueeze(1)

        # Apply GRU
        local_gru_out, _ = self.local_gru(local_trans_seq)
        local_gru_out = local_gru_out[:, -1, :]

        # Project local features
        local_features = self.local_fc(local_gru_out)
        local_features = F.relu(local_features)
        local_features = self.dropout(local_features)

        # ===== COMBINE AND CLASSIFY =====
        # Concatenate global and local features
        combined = torch.cat([global_features, local_features], dim=1)

        # Final classification layers
        combined = self.combine_fc1(combined)
        combined = self.bn_combine(combined)
        combined = F.relu(combined)
        combined = self.dropout(combined)

        out = self.combine_fc2(combined)

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
    Analyze transactions using ERGCN model.
    
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
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
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
    
    print("Categorical encoding complete")
    
    # Scale Numerical Features
    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    print("Numerical scaling complete")
    
    """# Graph Construction"""
    
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
    
    """# Model Initialization"""
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    
    # Initialize model
    model = ERGCN(
        hidden_channels=128,
        out_channels=1,
        num_transaction_features=len(transaction_feature_cols),
        num_card_nodes=num_card_nodes,
        num_email_nodes=num_email_nodes,
        gru_hidden=64
    ).to(DEVICE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    """# Load Pre-trained Model"""
    
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED MODEL")
    print("="*80)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Pre-trained model weights loaded successfully.")
    
    """# Inference"""
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    
    # Move data to device
    data = data.to(DEVICE)
    
    # Create a mask to evaluate all transactions in the input data
    n_total = data['transaction'].num_nodes
    all_nodes_mask = torch.ones(n_total, dtype=torch.bool, device=DEVICE)
    data['transaction'].test_mask = all_nodes_mask
    
    # Run evaluation
    test_loss, test_f1, test_recall, test_auc, test_probs, test_preds, test_labels = evaluate(
        model, data, data['transaction'].test_mask
    )
    
    print('Threshold at 0.5')
    print('Test Probabilities (first 10):')
    print(test_probs[:10])
    print('Test Predictions (first 10):')
    print(test_preds[:10])
    
    print("\nPredictions generated for all nodes in the input data.")
    
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
            'fraud_detected_by_ERGCN': sum(test_preds),
            'legitimate_detected_by_ERGCN': len(test_preds) - sum(test_preds),
            'fraud_rate': (sum(test_labels) / len(test_labels)) * 100 if len(test_labels) > 0 else 0
        }
    }
    
    return results

if __name__ == "__main__":
    # Standalone execution for testing
    CSV_PATH = 'ERGCN_files/merged_demo.csv'
    MODEL_PATH = 'ERGCN_files/ergcn_fraud_model_v1.pt'
    
    print("\n" + "="*80)
    print("STANDALONE ERGCN MODEL EXECUTION")
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
    print(f"  Fraud detected: {summary.get('fraud_detected_by_ERGCN', 0)}")
    print(f"  Legitimate detected: {summary.get('legitimate_detected_by_ERGCN', 0)}")
    print(f"  Fraud rate: {summary.get('fraud_rate', 0):.2f}%")
    
    # transactions = results.get("transactions", [])
    # if transactions:
    #     print(f"\nFirst 10 predictions:")
    #     for i, tx in enumerate(transactions[:10]):
    #         print(f"  TransactionID {tx['TransactionID']}: Predicted={tx['Predicted_isFraud']}, "
    #               f"Probability={tx['Fraud_Probability']:.4f}, Actual={tx['Actual_isFraud']}")
