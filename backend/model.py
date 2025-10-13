import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, global_max_pool
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class FraudDetectionModel:
    def __init__(self, model_path="training_v12_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.threshold = 0.5
        
        # Required features (already preprocessed in input CSV)
        self.required_features = [
            'C14', 'C1', 'V317', 'V308', 'card6', 'V306', 'V126',
            'C11', 'C6', 'V282', 'TransactionAmt', 'V53', 'card5',
            'C13', 'V35', 'V280', 'V279', 'card2', 'V258', 'M6',
            'V90', 'V82', 'C2', 'V87', 'V294', 'C12', 'V313', 'id_06'
        ]
        
        self.numerical_cols = self.required_features
        self.model = None
        

    
    def create_temporal_graphs(self, df):
        """Create temporal knowledge graphs"""
        df = df.copy()
        
        # Create composite card key
        df['card_key'] = (
            df['card2'].astype(str) + '_' +
            df['card5'].astype(str) + '_' +
            df['card6'].astype(str)
        )
        
        # Convert TransactionDT to days
        df['day'] = df['TransactionDT'] // (24 * 3600)
        
        # Get unique entities
        unique_cards = df['card_key'].unique()
        unique_emails = df['P_emaildomain'].unique()
        
        card_to_idx = {card: idx for idx, card in enumerate(unique_cards)}
        email_to_idx = {email: idx for idx, email in enumerate(unique_emails)}
        
        # Map indices
        df['card_idx'] = df['card_key'].map(card_to_idx)
        df['email_idx'] = df['P_emaildomain'].map(email_to_idx)
        
        # Split by day
        days = sorted(df['day'].unique())
        temporal_graphs = []
        
        for day in days:
            day_df = df[df['day'] == day].copy()
            
            # Create HeteroData object
            data = HeteroData()
            
            # Transaction features
            transaction_features = torch.tensor(
                day_df[self.numerical_cols].fillna(0).values,
                dtype=torch.float
            )
            data['transaction'].x = transaction_features
            data['transaction'].y = torch.tensor(day_df['isFraud'].values, dtype=torch.float)
            data['transaction'].transaction_id = torch.tensor(
                day_df['TransactionID'].values, dtype=torch.long
            )
            
            # Card and email nodes
            data['card'].num_nodes = len(unique_cards)
            data['email'].num_nodes = len(unique_emails)
            
            # Create edges
            num_transactions = len(day_df)
            transaction_indices = torch.arange(num_transactions)
            card_indices = torch.tensor(day_df['card_idx'].values, dtype=torch.long)
            email_indices = torch.tensor(day_df['email_idx'].values, dtype=torch.long)
            
            # Edge relationships
            data['transaction', 'uses_card', 'card'].edge_index = torch.stack([
                transaction_indices, card_indices
            ], dim=0)
            
            data['transaction', 'has_email', 'email'].edge_index = torch.stack([
                transaction_indices, email_indices
            ], dim=0)
            
            data['card', 'used_by', 'transaction'].edge_index = torch.stack([
                card_indices, transaction_indices
            ], dim=0)
            
            data['email', 'belongs_to', 'transaction'].edge_index = torch.stack([
                email_indices, transaction_indices
            ], dim=0)
            
            data.day = day
            data.num_transactions = num_transactions
            
            temporal_graphs.append(data)
        
        return temporal_graphs, len(unique_cards), len(unique_emails)
    
    def load_model(self, num_cards, num_emails):
        """Load the trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract original dimensions from checkpoint
        original_num_cards = checkpoint['card_embedding.weight'].shape[0]
        original_num_emails = checkpoint['email_embedding.weight'].shape[0]
        
        self.model = TemporalFraudDetector(
            in_channels=len(self.numerical_cols),
            hidden_dim=64,
            gru_hidden_dim=32,
            num_cards=original_num_cards,
            num_emails=original_num_emails,
            dropout_rate=0.5
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint)
        self.model.eval()
    
    def predict(self, csv_file_path):
        """Complete prediction pipeline"""
        print(f"Loading data from {csv_file_path}...")
        
        # Load CSV (already preprocessed)
        df = pd.read_csv(csv_file_path)
        
        # Create temporal graphs
        print("Creating temporal knowledge graphs...")
        temporal_graphs, num_cards, num_emails = self.create_temporal_graphs(df)
        
        # Load model
        print("Loading trained model...")
        self.load_model(num_cards, num_emails)
        
        # Make predictions
        print("Making predictions...")
        with torch.no_grad():
            predictions, labels = self.model(temporal_graphs)
            
            # Convert to binary predictions
            preds_binary = (predictions > self.threshold).float()
            raw_scores = predictions.cpu().numpy()
            binary_preds = preds_binary.cpu().numpy()
            true_labels = labels.cpu().numpy()
        
        # Create results table
        results_df = pd.DataFrame({
            'TransactionID': df['TransactionID'].values,
            'True_Label': true_labels,
            'Predicted_Label': binary_preds,
            'Raw_Score': raw_scores,
            'Is_Fraud_Predicted': binary_preds.astype(bool),
            'Prediction_Correct': (binary_preds == true_labels).astype(bool)
        })
        
        # Print summary
        accuracy = (binary_preds == true_labels).mean()
        print(f"\nPrediction Summary:")
        print(f"Total transactions: {len(results_df)}")
        print(f"Predicted fraud: {binary_preds.sum()}")
        print(f"Actual fraud: {true_labels.sum()}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report if we have both classes
        if len(np.unique(true_labels)) > 1:
            print("\nClassification Report:")
            print(classification_report(true_labels, binary_preds, 
                                      target_names=['Legitimate', 'Fraud']))
        
        return results_df


# Model architecture classes (from Training_v12)
class GlobalLearningUnit(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_relations, dropout_rate=0.5):
        super(GlobalLearningUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.dropout_rate = dropout_rate
        
        self.rgcn1 = RGCNConv(in_channels, hidden_dim, num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        self.fc = nn.Linear(gru_hidden_dim, hidden_dim)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = self.dropout1(x)
        x = F.relu(self.rgcn2(x, edge_index, edge_type))
        x = self.dropout2(x)
        
        x_pooled = global_max_pool(x, batch)
        
        return x, x_pooled
    
    def temporal_forward(self, pooled_sequence, hidden=None):
        gru_out, hidden = self.gru(pooled_sequence, hidden)
        global_feature = self.fc(gru_out)
        global_feature = self.dropout_fc(global_feature)
        
        return global_feature, hidden


class LocalLearningUnit(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_relations, dropout_rate=0.5):
        super(LocalLearningUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.dropout_rate = dropout_rate
        
        self.rgcn = RGCNConv(in_channels, hidden_dim, num_relations)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        self.fc = nn.Linear(gru_hidden_dim, hidden_dim)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.rgcn(x, edge_index, edge_type))
        x = self.dropout(x)
        
        return x
    
    def temporal_forward(self, node_sequence, hidden=None):
        gru_out, hidden = self.gru(node_sequence, hidden)
        local_feature = self.fc(gru_out)
        local_feature = self.dropout_fc(local_feature)
        
        return local_feature, hidden


class TemporalFraudDetector(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_cards, num_emails, dropout_rate=0.5):
        super(TemporalFraudDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_cards = num_cards
        self.num_emails = num_emails
        
        self.card_embedding = nn.Embedding(num_cards, in_channels)
        self.email_embedding = nn.Embedding(num_emails, in_channels)
        
        self.glu = GlobalLearningUnit(in_channels, hidden_dim, gru_hidden_dim, num_relations=4, dropout_rate=dropout_rate)
        self.llu = LocalLearningUnit(in_channels, hidden_dim, gru_hidden_dim, num_relations=4, dropout_rate=dropout_rate)
        
        self.fc_final = nn.Linear(hidden_dim * 2, 1)
        self.dropout_final = nn.Dropout(dropout_rate)
        
    def forward(self, temporal_graphs, batch_size=7):
        device = next(self.parameters()).device
        
        all_predictions = []
        all_labels = []
        
        for i in range(0, len(temporal_graphs), batch_size):
            batch_graphs = temporal_graphs[i:i+batch_size]
            batch_preds, batch_labels = self._forward_batch(batch_graphs)
            all_predictions.append(batch_preds)
            all_labels.append(batch_labels)
        
        return torch.cat(all_predictions), torch.cat(all_labels)
    
    def _forward_batch(self, temporal_graphs):
        device = next(self.parameters()).device
        
        global_pooled_features = []
        local_node_features_list = []
        transaction_counts = []
        all_labels = []
        
        for t, graph in enumerate(temporal_graphs):
            graph = graph.to(device)
            
            trans_x = graph['transaction'].x
            num_trans = trans_x.shape[0]
            transaction_counts.append(num_trans)
            all_labels.append(graph['transaction'].y)
            
            card_x = self.card_embedding(torch.arange(self.num_cards, device=device))
            email_x = self.email_embedding(torch.arange(self.num_emails, device=device))
            
            x_dict = {
                'transaction': trans_x,
                'card': card_x,
                'email': email_x
            }
            
            edge_index_list = []
            edge_type_list = []
            
            uses_card_edges = graph['transaction', 'uses_card', 'card'].edge_index
            edge_index_list.append(uses_card_edges + torch.tensor([[0], [num_trans]], device=device))
            edge_type_list.append(torch.zeros(uses_card_edges.shape[1], dtype=torch.long, device=device))
            
            has_email_edges = graph['transaction', 'has_email', 'email'].edge_index
            edge_index_list.append(has_email_edges + torch.tensor([[0], [num_trans + self.num_cards]], device=device))
            edge_type_list.append(torch.ones(has_email_edges.shape[1], dtype=torch.long, device=device))
            
            used_by_edges = graph['card', 'used_by', 'transaction'].edge_index
            edge_index_list.append(used_by_edges + torch.tensor([[num_trans], [0]], device=device))
            edge_type_list.append(torch.full((used_by_edges.shape[1],), 2, dtype=torch.long, device=device))
            
            belongs_to_edges = graph['email', 'belongs_to', 'transaction'].edge_index
            edge_index_list.append(belongs_to_edges + torch.tensor([[num_trans + self.num_cards], [0]], device=device))
            edge_type_list.append(torch.full((belongs_to_edges.shape[1],), 3, dtype=torch.long, device=device))
            
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_type = torch.cat(edge_type_list)
            
            x_all = torch.cat([trans_x, card_x, email_x], dim=0)
            
            total_nodes = num_trans + self.num_cards + self.num_emails
            batch = torch.zeros(total_nodes, dtype=torch.long, device=device)
            
            glu_node_features, glu_pooled = self.glu(x_all, edge_index, edge_type, batch)
            global_pooled_features.append(glu_pooled.unsqueeze(0))
            
            llu_node_features = self.llu(x_all, edge_index, edge_type)
            llu_trans_features = llu_node_features[:num_trans]
            local_node_features_list.append(llu_trans_features)
        
        # Process temporal sequences
        global_pooled_seq = torch.cat([f.squeeze(0) for f in global_pooled_features], dim=0)
        global_pooled_seq = global_pooled_seq.unsqueeze(0)
        global_fraud_features, _ = self.glu.temporal_forward(global_pooled_seq)
        global_fraud_features = global_fraud_features.squeeze(0)
        
        expanded_global_features = []
        for t, count in enumerate(transaction_counts):
            expanded_global_features.append(global_fraud_features[t].unsqueeze(0).expand(count, -1))
        expanded_global_features = torch.cat(expanded_global_features, dim=0)
        
        local_fraud_features_list = []
        for feats in local_node_features_list:
            local_seq = feats.unsqueeze(1)
            local_out, _ = self.llu.temporal_forward(local_seq)
            local_fraud_features_list.append(local_out.squeeze(1))
        
        local_fraud_features = torch.cat(local_fraud_features_list, dim=0)
        
        combined_features = torch.cat([expanded_global_features, local_fraud_features], dim=1)
        combined_features = self.dropout_final(combined_features)
        
        logits = self.fc_final(combined_features)
        predictions = torch.sigmoid(logits).squeeze(1)
        
        labels = torch.cat(all_labels, dim=0)
        
        return predictions, labels


if __name__ == "__main__":
    fraud_detector = FraudDetectionModel()
    results = fraud_detector.predict("demo.csv")
    print("\n", results.head(20))
    results.to_csv("fraud_predictions.csv", index=False)
    print("\nSaved to 'fraud_predictions.csv'")