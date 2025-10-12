"""
FastAPI backend for R-GCN and ERGCN model inference
Integrates with Google Colab trained models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from scipy.stats import ttest_rel
import pandas as pd
import uvicorn
from interpretability import initialize_explainer, get_transaction_explanation

app = FastAPI(title="Fraud Detection Model API")

# CORS middleware for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    TransactionID: int
    TrueLabel: int

class TransactionBatch(BaseModel):
    transactions: List[Transaction]

class ModelMetrics(BaseModel):
    recall: float
    f1: float
    auc: float

class AnalysisResult(BaseModel):
    transactions: List[Dict[str, Any]]
    metrics: Dict[str, Any]

# Global model storage
models = {
    'rgcn': None,
    'ergcn': None
}

def load_models():
    """Load pre-trained models from .pth files"""
    try:
        # Replace with your actual model loading logic
        models['rgcn'] = torch.load('models/rgcn_model.pth', map_location='cpu')
        models['ergcn'] = torch.load('models/ergcn_model.pth', map_location='cpu')
        
        # Set models to evaluation mode
        models['rgcn'].eval()
        models['ergcn'].eval()
        
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        # Use mock models for development
        models['rgcn'] = "mock_rgcn"
        models['ergcn'] = "mock_ergcn"

def predict_batch(model_name: str, transactions: List[Transaction]) -> List[int]:
    """
    Perform batch prediction using specified model
    Replace this with your actual model inference logic
    """
    if models[model_name] == f"mock_{model_name}":
        # Mock predictions for development
        return [np.random.choice([0, 1], p=[0.7, 0.3]) for _ in transactions]
    
    # Actual model inference would go here
    # Example structure:
    # 1. Preprocess transaction data
    # 2. Convert to model input format
    # 3. Run inference
    # 4. Return predictions
    
    predictions = []
    with torch.no_grad():
        for transaction in transactions:
            # Your preprocessing and inference logic here
            # pred = model(processed_input)
            # predictions.append(pred.item())
            pass
    
    return predictions

def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float] = None) -> ModelMetrics:
    """Calculate evaluation metrics"""
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Use predictions as probabilities if no probabilities provided
    if y_prob is None:
        y_prob = y_pred
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Handle case where only one class is present
        auc = 0.5
    
    return ModelMetrics(recall=recall, f1=f1, auc=auc)

def statistical_significance_test(rgcn_preds: List[int], ergcn_preds: List[int], true_labels: List[int]) -> float:
    """
    Perform paired t-test to determine statistical significance
    between R-GCN and ERGCN performance
    """
    # Calculate accuracy for each prediction
    rgcn_correct = [1 if pred == true else 0 for pred, true in zip(rgcn_preds, true_labels)]
    ergcn_correct = [1 if pred == true else 0 for pred, true in zip(ergcn_preds, true_labels)]
    
    # Perform paired t-test
    _, p_value = ttest_rel(ergcn_correct, rgcn_correct)
    
    return p_value

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    # Initialize interpretability after models are loaded
    if models['rgcn'] != "mock_rgcn" and models['ergcn'] != "mock_ergcn":
        # Initialize with your graph data
        # initialize_explainer(models['rgcn'], models['ergcn'], graph_data)

@app.get("/")
async def root():
    return {"message": "Fraud Detection Model API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "rgcn": models['rgcn'] is not None,
            "ergcn": models['ergcn'] is not None
        }
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_transactions(batch: TransactionBatch):
    """
    Analyze transactions using both R-GCN and ERGCN models
    """
    try:
        transactions = batch.transactions
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")
        
        # Extract true labels
        true_labels = [t.TrueLabel for t in transactions]
        
        # Get predictions from both models
        rgcn_predictions = predict_batch('rgcn', transactions)
        ergcn_predictions = predict_batch('ergcn', transactions)
        
        # Calculate metrics for both models
        rgcn_metrics = calculate_metrics(true_labels, rgcn_predictions)
        ergcn_metrics = calculate_metrics(true_labels, ergcn_predictions)
        
        # Calculate statistical significance
        p_value = statistical_significance_test(rgcn_predictions, ergcn_predictions, true_labels)
        
        # Prepare response
        processed_transactions = []
        for i, transaction in enumerate(transactions):
            processed_transactions.append({
                "TransactionID": transaction.TransactionID,
                "TrueLabel": transaction.TrueLabel,
                "RGCN": rgcn_predictions[i],
                "ERGCN": ergcn_predictions[i]
            })
        
        result = AnalysisResult(
            transactions=processed_transactions,
            metrics={
                "RGCN": rgcn_metrics.dict(),
                "ERGCN": ergcn_metrics.dict(),
                "p_value": p_value
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/explain/{transaction_id}")
async def explain_prediction(transaction_id: int, model_type: str = "both"):
    """
    Get explanation for a specific transaction prediction
    """
    try:
        explanation = get_transaction_explanation(transaction_id, model_type)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/explain/{transaction_id}/summary")
async def explain_prediction_summary(transaction_id: int):
    """
    Get summary comparison of both models' explanations
    """
    try:
        from interpretability import fraud_explainer
        if fraud_explainer is None:
            raise HTTPException(status_code=503, detail="Explainer not initialized")
        
        summary = fraud_explainer.get_explanation_summary(transaction_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation summary failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)