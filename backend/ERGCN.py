import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc, classification_report
)
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, global_max_pool

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# FILE PATHS
ORIGINAL_TRANSACTIONS_CSV = 'ERGCN_files/min_30_train_transaction.csv'
ORIGINAL_IDENTITY_CSV = 'ERGCN_files/min_30_train_identity.csv'
MODEL_PATH = 'ERGCN_files/training_v12_model.pth'

# GLOBAL PREPROCESSOR INSTANCE
preprocessor = None
model = None

# =============================================================================
# PART 1: THE PREPROCESSOR CLASS
# =============================================================================
class InferencePreprocessor:
    def __init__(self):
        # Match RGCN's required features including isFraud
        self.required_features = [
            'isFraud', 'C14', 'C1', 'V317', 'V308', 'card6', 'V306', 'V126',
            'C11', 'C6', 'V282', 'TransactionAmt', 'V53', 'P_emaildomain',
            'card5', 'C13', 'V35', 'V280', 'V279', 'card2', 'V258', 'M6',
            'V90', 'V82', 'TransactionDT', 'C2', 'V87', 'V294', 'C12', 'V313', 'id_06'
        ]
        self.imputation_values_ = {}
        self.label_encoders_ = {}
        self.scaler_ = None
        self.card_to_idx_ = {}
        self.email_to_idx_ = {}
        self.is_fitted = False

    def fit(self, transactions_file_path, identity_file_path):
        print("Fitting preprocessor in-memory from original training data...")
        transactions_df = pd.read_csv(transactions_file_path)
        identity_df = pd.read_csv(identity_file_path)
        df = pd.merge(transactions_df, identity_df, on='TransactionID', how='left')
        
        # Select available required features (handle missing isFraud gracefully)
        available_features = [f for f in self.required_features if f in df.columns]
        df = df[available_features + ['TransactionID']]

        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype == 'object':
                    self.imputation_values_[column] = df[column].mode()[0]
                else:
                    skewness = stats.skew(df[column].dropna())
                    self.imputation_values_[column] = df[column].median() if abs(skewness) > 1 else df[column].mean()

        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            le = LabelEncoder()
            le.fit(df[column].astype(str).unique())
            self.label_encoders_[column] = le

        temp_df = self._apply_imputation_and_encoding(df.copy())
        
        feature_cols_to_scale = [col for col in self.required_features if col not in ['TransactionID', 'TransactionDT', 'P_emaildomain', 'M6', 'card6']]
        self.scaler_ = StandardScaler().fit(temp_df[feature_cols_to_scale])

        # Create card key mapping - handle different data formats
        df['card_key'] = df['card2'].astype(str) + '_' + df['card5'].astype(str) + '_' + df['card6'].astype(str)
        self.card_to_idx_ = {card: idx for idx, card in enumerate(df['card_key'].unique())}
        
        # Handle email domain mapping - ensure it exists
        if 'P_emaildomain' in df.columns:
            self.email_to_idx_ = {email: idx for idx, email in enumerate(df['P_emaildomain'].astype(str).unique())}
        else:
            self.email_to_idx_ = {'unknown': 0}  # Fallback for missing email domain
        
        self.is_fitted = True
        print(f"Preprocessor is ready. Card mappings: {len(self.card_to_idx_)}, Email mappings: {len(self.email_to_idx_)}")

    def _apply_imputation_and_encoding(self, df):
        df.fillna(self.imputation_values_, inplace=True)
        for col, encoder in self.label_encoders_.items():
            known_classes = set(encoder.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else 'unknown')
            if 'unknown' not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, 'unknown')
            df[col] = encoder.transform(df[col])
        return df

    def transform(self, new_df):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        print("Transforming demo data...")
        # Select available required features (handle missing isFraud gracefully)
        available_features = [f for f in self.required_features if f in new_df.columns]
        df = new_df.copy()[available_features + ['TransactionID']]
        df_processed = self._apply_imputation_and_encoding(df)
        
        feature_cols_to_scale = [col for col in self.required_features if col not in ['TransactionID', 'TransactionDT', 'P_emaildomain', 'M6', 'card6']]
        df_processed[feature_cols_to_scale] = self.scaler_.transform(df_processed[feature_cols_to_scale])
        
        print("Transformation complete.")
        return df_processed

# =============================================================================
# PART 2: THE MODEL ARCHITECTURE
# (Copied exactly from the training notebook)
# =============================================================================
class GlobalLearningUnit(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_relations, dropout_rate=0.5):
        super(GlobalLearningUnit, self).__init__()
        self.rgcn1, self.rgcn2 = RGCNConv(in_channels, hidden_dim, num_relations), RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.dropout1, self.dropout2 = nn.Dropout(dropout_rate), nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        self.fc, self.dropout_fc = nn.Linear(gru_hidden_dim, hidden_dim), nn.Dropout(dropout_rate)
    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = self.dropout1(x)
        x = F.relu(self.rgcn2(x, edge_index, edge_type))
        x_pooled = global_max_pool(self.dropout2(x), batch)
        return x, x_pooled
    def temporal_forward(self, seq, h=None): return self.dropout_fc(self.fc(self.gru(seq, h)[0])), self.gru(seq, h)[1]

class LocalLearningUnit(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_relations, dropout_rate=0.5):
        super(LocalLearningUnit, self).__init__()
        self.rgcn, self.dropout = RGCNConv(in_channels, hidden_dim, num_relations), nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        self.fc, self.dropout_fc = nn.Linear(gru_hidden_dim, hidden_dim), nn.Dropout(dropout_rate)
    def forward(self, x, edge_index, edge_type): return self.dropout(F.relu(self.rgcn(x, edge_index, edge_type)))
    def temporal_forward(self, seq, h=None): return self.dropout_fc(self.fc(self.gru(seq, h)[0])), self.gru(seq, h)[1]

class TemporalFraudDetector(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_cards, num_emails, dropout_rate=0.5, actual_in_channels=None):
        super(TemporalFraudDetector, self).__init__()
        self.num_cards, self.num_emails = num_cards, num_emails
        self.in_channels = in_channels
        self.actual_in_channels = actual_in_channels or in_channels
        
        # Feature adapter if dimensions don't match
        if self.actual_in_channels != self.in_channels:
            self.feature_adapter = nn.Linear(self.actual_in_channels, self.in_channels)
        else:
            self.feature_adapter = None
        
        self.card_embedding, self.email_embedding = nn.Embedding(num_cards, in_channels), nn.Embedding(num_emails, in_channels)
        self.glu = GlobalLearningUnit(in_channels, hidden_dim, gru_hidden_dim, 4, dropout_rate)
        self.llu = LocalLearningUnit(in_channels, hidden_dim, gru_hidden_dim, 4, dropout_rate)
        self.fc_final, self.dropout_final = nn.Linear(hidden_dim * 2, 1), nn.Dropout(dropout_rate)
    def forward(self, graphs):
        preds = [self._forward_batch([g])[0] for g in graphs]
        return torch.cat(preds), None
    def _forward_batch(self, graphs):
        device = next(self.parameters()).device
        pooled_feats, node_feats_list, counts = [], [], []
        for g in graphs:
            g = g.to(device)
            trans_x = g['transaction'].x
            
            # Apply feature adapter if needed
            if self.feature_adapter is not None:
                trans_x = self.feature_adapter(trans_x)
            
            num_trans = trans_x.shape[0]
            counts.append(num_trans)
            card_x, email_x = self.card_embedding(torch.arange(self.num_cards, device=device)), self.email_embedding(torch.arange(self.num_emails, device=device))
            x_all = torch.cat([trans_x, card_x, email_x])
            edges, types = [], []
            edge_map = {('transaction', 'uses_card', 'card'):0, ('transaction', 'has_email', 'email'):1, ('card', 'used_by', 'transaction'):2, ('email', 'belongs_to', 'transaction'):3}
            offsets = {'card': num_trans, 'email': num_trans + self.num_cards}
            for (src, _, dst), type_val in edge_map.items():
                edge = g[src, _, dst].edge_index
                edge[0] += offsets.get(src, 0)
                edge[1] += offsets.get(dst, 0)
                edges.append(edge)
                types.append(torch.full((edge.shape[1],), type_val, device=device))
            _, pooled = self.glu(x_all, torch.cat(edges, 1), torch.cat(types), torch.zeros(x_all.shape[0], dtype=torch.long, device=device))
            pooled_feats.append(pooled.unsqueeze(0))
            node_feats_list.append(self.llu(x_all, torch.cat(edges, 1), torch.cat(types))[:num_trans])
        global_feats, _ = self.glu.temporal_forward(torch.cat([f.squeeze(0) for f in pooled_feats]).unsqueeze(0))
        expanded_global = torch.cat([global_feats.squeeze(0)[t].expand(c, -1) for t, c in enumerate(counts)])
        local_feats_list = [self.llu.temporal_forward(f.unsqueeze(1))[0].squeeze(1) for f in node_feats_list]
        combined = self.dropout_final(torch.cat([expanded_global, torch.cat(local_feats_list)], 1))
        return torch.sigmoid(self.fc_final(combined)).squeeze(1), None

# =============================================================================
# PART 3: GRAPH CREATION AND PREDICTION LOGIC
# =============================================================================
def create_graphs_for_demo(df, card_to_idx, email_to_idx):
    print("Creating temporal graphs for demo...")
    df = df.copy()
    
    # Create card key - handle missing columns gracefully
    if all(col in df.columns for col in ['card2', 'card5', 'card6']):
        df['card_key'] = df['card2'].astype(str) + '_' + df['card5'].astype(str) + '_' + df['card6'].astype(str)
    else:
        # Fallback: create a simple card key from available data
        available_card_cols = [col for col in ['card2', 'card5', 'card6'] if col in df.columns]
        if available_card_cols:
            df['card_key'] = df[available_card_cols[0]].astype(str)
        else:
            df['card_key'] = 'unknown_card'
    
    df['day'] = df['TransactionDT'] // (24 * 3600)
    
    # Map card indices with fallback for unknown cards
    df['card_idx'] = df['card_key'].map(card_to_idx)
    # Replace unknown cards with a default index (0)
    df['card_idx'] = df['card_idx'].fillna(0).astype(int)
    
    # Map email indices with fallback
    if 'P_emaildomain' in df.columns:
        df['email_idx'] = df['P_emaildomain'].astype(str).map(email_to_idx)
    else:
        df['email_idx'] = 0  # Default email index
    # Replace unknown emails with a default index (0)
    df['email_idx'] = df['email_idx'].fillna(0).astype(int)
    
    # Clamp indices to valid ranges to avoid embedding lookup errors
    max_card_idx = len(card_to_idx) - 1
    max_email_idx = len(email_to_idx) - 1
    df['card_idx'] = df['card_idx'].clip(0, max_card_idx)
    df['email_idx'] = df['email_idx'].clip(0, max_email_idx)
    
    original_rows = len(df)
    
    # Don't drop transactions anymore - use fallback mappings instead
    print(f"Processing {len(df)} transactions (kept all with fallback mappings)")
    
    if df.empty: return [], None

    # Handle ground truth labels
    labels = None
    if 'isFraud' in df.columns and not df['isFraud'].isna().all():
        labels = df['isFraud'].fillna(0).values
    else:
        labels = np.zeros(len(df))

    graphs = []
    # Remove non-feature columns for numerical features
    exclude_cols = ['TransactionID','TransactionDT','card_key','day','card_idx','email_idx','card2','card5','card6','M6','isFraud', 'P_emaildomain']
    numerical_cols = [c for c in df.columns if c not in exclude_cols]
    
    for day in sorted(df['day'].unique()):
        day_df = df[df['day'] == day]
        g = HeteroData()
        g['transaction'].x = torch.tensor(day_df[numerical_cols].values, dtype=torch.float)
        g['transaction'].transaction_id = torch.tensor(day_df['TransactionID'].values, dtype=torch.long)
        
        # Add labels to the graph
        day_labels = labels[df['day'] == day] if labels is not None else np.zeros(len(day_df))
        g['transaction'].y = torch.tensor(day_labels, dtype=torch.float)
        
        g['card'].num_nodes, g['email'].num_nodes = len(card_to_idx), len(email_to_idx)
        trans_idx, card_idx, email_idx = torch.arange(len(day_df)), torch.tensor(day_df['card_idx'].values), torch.tensor(day_df['email_idx'].values)
        g['transaction','uses_card','card'].edge_index = torch.stack([trans_idx, card_idx])
        g['transaction','has_email','email'].edge_index = torch.stack([trans_idx, email_idx])
        g['card','used_by','transaction'].edge_index = torch.stack([card_idx, trans_idx])
        g['email','belongs_to','transaction'].edge_index = torch.stack([email_idx, trans_idx])
        graphs.append(g)
    return graphs, labels

def analyze_transactions(csv_data: str):
    """
    Analyze transactions using ERGCN model.
    
    Args:
        csv_data: CSV data as string containing transaction data
        
    Returns:
        Dictionary containing predictions and metrics
    """
    global preprocessor, model
    
    # Initialize preprocessor and model if not already done
    if preprocessor is None:
        print("Initializing ERGCN preprocessor...")
        preprocessor = InferencePreprocessor()
        preprocessor.fit(ORIGINAL_TRANSACTIONS_CSV, ORIGINAL_IDENTITY_CSV)
        
    if model is None:
        print("Loading ERGCN model...")
        # We'll initialize model after getting the first batch of data
        
    # Load CSV data from string
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    
    # Ensure TransactionID is included
    if 'TransactionID' not in df.columns:
        raise ValueError("CSV must contain TransactionID column")
    
    original_ids = df['TransactionID'].copy()
    
    # Transform data using preprocessor
    processed_df = preprocessor.transform(df)
    
    # Create graphs
    graphs, all_labels = create_graphs_for_demo(processed_df, preprocessor.card_to_idx_, preprocessor.email_to_idx_)
    if not graphs:
        print("No valid transactions to process.")
        # Return empty results in RGCN format
        return {
            'transactions': [],
            'metrics': {'loss': 0.0, 'f1': 0.0, 'recall': 0.0, 'auc': 0.0},
            'summary': {
                'total_transactions': len(df),
                'fraud_detected_by_ERGCN': 0,
                'legitimate_detected_by_ERGCN': len(df),
                'fraud_rate': 0.0
            }
        }
    
    # Initialize model if not done yet
    if model is None:
        # Load the original model state to get the correct dimensions
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Extract original dimensions from the checkpoint
        original_in_channels = checkpoint['card_embedding.weight'].shape[1]  # 28
        original_num_cards = checkpoint['card_embedding.weight'].shape[0]    # 340487
        original_num_emails = checkpoint['email_embedding.weight'].shape[0]  # 181462
        
        print(f"Original model dimensions:")
        print(f"  Input channels: {original_in_channels}")
        print(f"  Cards: {original_num_cards}")
        print(f"  Emails: {original_num_emails}")
        
        # Get actual input channels from our data
        actual_in_channels = graphs[0]['transaction'].x.shape[1]
        print(f"Actual input channels from data: {actual_in_channels}")
        
        # Create model with original dimensions but actual input channels
        model = TemporalFraudDetector(
            original_in_channels, 64, 32, 
            original_num_cards, 
            original_num_emails, 0.5,
            actual_in_channels=actual_in_channels
        ).to(DEVICE)
        
        # Load the original state dict (excluding feature_adapter if it exists)
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            if "feature_adapter" in str(e):
                # Load state dict excluding feature_adapter
                model_state = model.state_dict()
                for key in checkpoint:
                    if key not in model_state or key.startswith('feature_adapter'):
                        continue
                    model_state[key] = checkpoint[key]
                model.load_state_dict(model_state)
                print("Loaded model state dict (excluding feature_adapter)")
            else:
                raise e
        
        model.eval()
        print("ERGCN model loaded successfully with original dimensions.")
        
        # Store the original dimensions for later use
        model.original_in_channels = original_in_channels
        model.original_num_cards = original_num_cards
        model.original_num_emails = original_num_emails
    
    # Make predictions
    print("Making ERGCN predictions...")
    with torch.no_grad():
        probs, _ = model(graphs)
    
    # Get predictions and transaction IDs from graphs
    all_transaction_ids = []
    all_predictions = []
    all_probabilities = []
    
    for graph in graphs:
        all_transaction_ids.extend(graph['transaction'].transaction_id.cpu().numpy())
    
    all_probabilities = probs.cpu().numpy()
    all_predictions = (all_probabilities >= 0.5).astype(int)
    
    # Create results DataFrame matching RGCN format
    predictions_df = pd.DataFrame({
        'TransactionID': all_transaction_ids,
        'Actual_isFraud': all_labels if all_labels is not None else [None] * len(all_transaction_ids),
        'Predicted_isFraud': all_predictions,
        'Fraud_Probability': all_probabilities
    })
    
    # Calculate metrics if ground truth is available
    if all_labels is not None and len(np.unique(all_labels)) > 1:
        f1 = f1_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        auc_score = roc_auc_score(all_labels, all_probabilities)
        fraud_rate = (sum(all_labels) / len(all_labels)) * 100
    else:
        f1 = recall = auc_score = 0.0
        fraud_rate = 0.0
    
    # Return results in the format expected by the API (matching RGCN)
    results = {
        'transactions': predictions_df.to_dict('records'),
        'metrics': {
            'loss': 0.0,  # ERGCN doesn't calculate loss during inference
            'f1': f1,
            'recall': recall,
            'auc': auc_score
        },
        'summary': {
            'total_transactions': len(predictions_df),
            'fraud_detected_by_ERGCN': sum(all_predictions),
            'legitimate_detected_by_ERGCN': len(all_predictions) - sum(all_predictions),
            'fraud_rate': fraud_rate
        }
    }
    
    print("ERGCN analysis complete.")
    return results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Load demo data for testing
    with open('ERGCN_files/demo.csv', 'r') as f:
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