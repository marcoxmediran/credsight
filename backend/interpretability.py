"""
Model Interpretability for R-GCN and ERGCN
Provides on-demand explanations for fraud predictions
"""

import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.config import ExplainerConfig, ModelConfig
import numpy as np
from typing import Dict, List, Any, Optional
import json
from functools import lru_cache
import time

class FraudExplainer:
    def __init__(self, rgcn_model, ergcn_model, graph_data):
        self.rgcn_model = rgcn_model
        self.ergcn_model = ergcn_model
        self.graph_data = graph_data
        
        # Initialize explainers for both models
        self.rgcn_explainer = self._setup_explainer(rgcn_model, "rgcn")
        self.ergcn_explainer = self._setup_explainer(ergcn_model, "ergcn")
        
        # Cache for explanations
        self.explanation_cache = {}
    
    def _setup_explainer(self, model, model_name):
        """Setup GNNExplainer for the given model"""
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='classification',
                task_level='node',
                return_type='probs'
            )
        )
        return explainer
    
    @lru_cache(maxsize=1000)
    def explain_transaction(self, transaction_id: int, model_type: str = "both") -> Dict[str, Any]:
        """
        Generate explanation for a specific transaction
        
        Args:
            transaction_id: ID of transaction to explain
            model_type: "rgcn", "ergcn", or "both"
        
        Returns:
            Dictionary containing explanation data
        """
        cache_key = f"{transaction_id}_{model_type}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        start_time = time.time()
        
        # Find node index for transaction
        node_idx = self._get_node_index(transaction_id)
        if node_idx is None:
            return {"error": f"Transaction {transaction_id} not found"}
        
        explanations = {}
        
        if model_type in ["rgcn", "both"]:
            explanations["rgcn"] = self._explain_single_model(
                node_idx, self.rgcn_explainer, "R-GCN"
            )
        
        if model_type in ["ergcn", "both"]:
            explanations["ergcn"] = self._explain_single_model(
                node_idx, self.ergcn_explainer, "ERGCN"
            )
        
        result = {
            "transaction_id": transaction_id,
            "explanations": explanations,
            "computation_time": time.time() - start_time
        }
        
        # Cache result
        self.explanation_cache[cache_key] = result
        return result
    
    def _explain_single_model(self, node_idx: int, explainer, model_name: str) -> Dict[str, Any]:
        """Generate explanation for single model"""
        try:
            # Generate explanation
            explanation = explainer(
                x=self.graph_data.x,
                edge_index=self.graph_data.edge_index,
                index=node_idx
            )
            
            # Extract feature importance
            feature_importance = self._extract_feature_importance(explanation, node_idx)
            
            # Extract edge importance
            edge_importance = self._extract_edge_importance(explanation, node_idx)
            
            # Get prediction confidence
            with torch.no_grad():
                if model_name == "R-GCN":
                    pred = self.rgcn_model(self.graph_data.x, self.graph_data.edge_index)
                else:
                    pred = self.ergcn_model(self.graph_data.x, self.graph_data.edge_index)
                
                confidence = F.softmax(pred[node_idx], dim=0)
                prediction = torch.argmax(confidence).item()
                fraud_probability = confidence[1].item()
            
            return {
                "model": model_name,
                "prediction": prediction,
                "fraud_probability": fraud_probability,
                "feature_importance": feature_importance,
                "edge_importance": edge_importance,
                "top_reasons": self._generate_top_reasons(feature_importance, prediction)
            }
            
        except Exception as e:
            return {"error": f"Explanation failed for {model_name}: {str(e)}"}
    
    def _extract_feature_importance(self, explanation, node_idx: int) -> List[Dict[str, Any]]:
        """Extract and rank feature importance"""
        if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
            feature_scores = explanation.node_mask[node_idx].cpu().numpy()
            
            # Feature names (customize based on your dataset)
            feature_names = [
                "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4",
                "card5", "card6", "addr1", "addr2", "dist1", "dist2",
                "P_emaildomain", "R_emaildomain", "C1", "C2", "C3", "C4", "C5",
                "D1", "D2", "D3", "D4", "D5", "V1", "V2", "V3", "V4", "V5"
            ]
            
            # Create feature importance list
            importance_list = []
            for i, score in enumerate(feature_scores):
                if i < len(feature_names):
                    importance_list.append({
                        "feature": feature_names[i],
                        "importance": float(score),
                        "rank": i + 1
                    })
            
            # Sort by importance (descending)
            importance_list.sort(key=lambda x: abs(x["importance"]), reverse=True)
            
            # Add ranks
            for i, item in enumerate(importance_list):
                item["rank"] = i + 1
            
            return importance_list[:10]  # Top 10 features
        
        return []
    
    def _extract_edge_importance(self, explanation, node_idx: int) -> List[Dict[str, Any]]:
        """Extract important edges/relationships"""
        if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
            edge_scores = explanation.edge_mask.cpu().numpy()
            edge_index = self.graph_data.edge_index.cpu().numpy()
            
            # Find edges connected to target node
            connected_edges = []
            for i, (src, dst) in enumerate(edge_index.T):
                if src == node_idx or dst == node_idx:
                    connected_node = dst if src == node_idx else src
                    connected_edges.append({
                        "connected_transaction": int(connected_node),
                        "edge_importance": float(edge_scores[i]),
                        "relationship_type": self._get_relationship_type(src, dst)
                    })
            
            # Sort by importance
            connected_edges.sort(key=lambda x: abs(x["edge_importance"]), reverse=True)
            return connected_edges[:5]  # Top 5 connections
        
        return []
    
    def _get_relationship_type(self, src: int, dst: int) -> str:
        """Determine relationship type between nodes"""
        # Customize based on your graph construction
        return "similar_transaction"
    
    def _generate_top_reasons(self, feature_importance: List[Dict], prediction: int) -> List[str]:
        """Generate human-readable explanations"""
        if not feature_importance:
            return ["No explanation available"]
        
        reasons = []
        top_features = feature_importance[:3]
        
        fraud_indicators = {
            "TransactionAmt": "Transaction amount",
            "card1": "Card information",
            "addr1": "Billing address",
            "P_emaildomain": "Email domain",
            "C1": "Counting feature",
            "D1": "Time delta",
            "V1": "Vesta feature"
        }
        
        for feat in top_features:
            feature_name = feat["feature"]
            importance = feat["importance"]
            
            if prediction == 1:  # Fraud
                if importance > 0:
                    reason = f"High {fraud_indicators.get(feature_name, feature_name)} indicates fraud risk"
                else:
                    reason = f"Unusual {fraud_indicators.get(feature_name, feature_name)} pattern detected"
            else:  # Legitimate
                if importance > 0:
                    reason = f"Normal {fraud_indicators.get(feature_name, feature_name)} supports legitimacy"
                else:
                    reason = f"Typical {fraud_indicators.get(feature_name, feature_name)} behavior"
            
            reasons.append(reason)
        
        return reasons
    
    def _get_node_index(self, transaction_id: int) -> Optional[int]:
        """Map transaction ID to node index"""
        # Implement based on your graph construction
        # This is a placeholder - replace with actual mapping logic
        if hasattr(self.graph_data, 'transaction_ids'):
            try:
                return self.graph_data.transaction_ids.tolist().index(transaction_id)
            except ValueError:
                return None
        return None
    
    def get_explanation_summary(self, transaction_id: int) -> Dict[str, Any]:
        """Get a summary comparison of both models' explanations"""
        explanation = self.explain_transaction(transaction_id, "both")
        
        if "error" in explanation:
            return explanation
        
        rgcn_exp = explanation["explanations"].get("rgcn", {})
        ergcn_exp = explanation["explanations"].get("ergcn", {})
        
        return {
            "transaction_id": transaction_id,
            "model_agreement": rgcn_exp.get("prediction") == ergcn_exp.get("prediction"),
            "confidence_difference": abs(
                rgcn_exp.get("fraud_probability", 0) - ergcn_exp.get("fraud_probability", 0)
            ),
            "rgcn_prediction": rgcn_exp.get("prediction"),
            "ergcn_prediction": ergcn_exp.get("prediction"),
            "rgcn_confidence": rgcn_exp.get("fraud_probability"),
            "ergcn_confidence": ergcn_exp.get("fraud_probability"),
            "common_important_features": self._find_common_features(
                rgcn_exp.get("feature_importance", []),
                ergcn_exp.get("feature_importance", [])
            )
        }
    
    def _find_common_features(self, rgcn_features: List, ergcn_features: List) -> List[str]:
        """Find features important to both models"""
        rgcn_top = {f["feature"] for f in rgcn_features[:5]}
        ergcn_top = {f["feature"] for f in ergcn_features[:5]}
        return list(rgcn_top.intersection(ergcn_top))

# Global explainer instance
fraud_explainer = None

def initialize_explainer(rgcn_model, ergcn_model, graph_data):
    """Initialize the global explainer"""
    global fraud_explainer
    fraud_explainer = FraudExplainer(rgcn_model, ergcn_model, graph_data)

def get_transaction_explanation(transaction_id: int, model_type: str = "both") -> Dict[str, Any]:
    """Get explanation for a transaction"""
    if fraud_explainer is None:
        return {"error": "Explainer not initialized"}
    
    return fraud_explainer.explain_transaction(transaction_id, model_type)