from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import pandas as pd
from io import StringIO

from basemodels import AnalysisResult, TransactionBatch
from RGCN import analyze_transactions

app = FastAPI(title="Credsight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Fraud Detection Model API", "status": "running"}

@app.get("/health")
async def health_check():
    """Check the health status of the API"""
    return {
        "status": "healthy",
        "message": "RGCN model ready for analysis"
    }

@app.get("/explain/{transaction_id}")
async def explain_transaction(transaction_id: int, model_type: str = "both"):
    """
    Get explanation for a specific transaction prediction.
    Returns mock explanation data for now.
    """
    try:
        # Simplified explanation data - only what's needed
        explanation = {
            "transaction_id": transaction_id,
            "explanations": {
                "rgcn": {
                    "model": "R-GCN",
                    "prediction": 1 if transaction_id % 3 == 0 else 0,
                    "fraud_probability": 0.75 if transaction_id % 3 == 0 else 0.25
                },
                "ergcn": {
                    "model": "ERGCN", 
                    "prediction": 0 if transaction_id % 2 == 0 else 1,
                    "fraud_probability": 0.3 if transaction_id % 2 == 0 else 0.8
                }
            }
        }
        
        return explanation
        
    except Exception as e:
        print(f"Error in explain_transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_transactions_endpoint(batch: TransactionBatch):
    """
    Analyze transactions using RGCN model.
    This serves as a gateway for model inference - accepts transaction data from frontend,
    converts it to CSV format, passes to RGCN.py for processing, and returns predictions.
    ERGCN will be implemented SOON.
    """
    try:
        # Validate input
        if not batch.transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")
        
        # Convert TransactionBatch to CSV format
        transactions_data = []
        for transaction in batch.transactions:
            # Use all attributes from the transaction object
            tx_dict = transaction.dict()
            transactions_data.append(tx_dict)
        
        # Convert to DataFrame and then to CSV string
        df = pd.DataFrame(transactions_data)
        print(f"Received columns: {list(df.columns)}")
        csv_data = df.to_csv(index=False)
        
        print(f"Processing {len(batch.transactions)} transactions with RGCN model...")
        
        # Call RGCN analysis function
        rgcn_results = analyze_transactions(csv_data)
        
        # Convert RGCN results to the expected API format
        processed_transactions = []
        for result in rgcn_results['transactions']:
            # Handle both protocols: with and without ground truth
            true_label = None
            if result['Actual_isFraud'] is not None and not pd.isna(result['Actual_isFraud']):
                true_label = int(result['Actual_isFraud'])
            
            # Handle NaN TransactionID
            transaction_id = result['TransactionID']
            if pd.isna(transaction_id):
                continue  # Skip rows with missing TransactionID
            
            processed_transactions.append({
                "TransactionID": int(transaction_id),
                "TrueLabel": true_label,
                "RGCN_Prediction": int(result['Predicted_isFraud']),
                "RGCN_Confidence": float(result['Fraud_Probability']),
                # ERGCN not yet implemented - set to null
                "ERGCN_Prediction": None,
                "ERGCN_Confidence": None
            })
        
        # Prepare final response
        # Handle None metrics for no ground truth cases
        rgcn_metrics = rgcn_results['metrics']
        result = AnalysisResult(
            transactions=processed_transactions,
            metrics={
                "RGCN": {
                    "recall": rgcn_metrics['recall'],
                    "f1": rgcn_metrics['f1'],
                    "auc": rgcn_metrics['auc']
                },
                # ERGCN not yet implemented - set to null
                "ERGCN": None, 
                "summary": {
                    "total_transactions": int(rgcn_results['summary']['total_transactions']),
                    "fraud_detected_by_RGCN": int(rgcn_results['summary']['fraud_detected_by_RGCN']),
                    "legitimate_detected_by_RGCN": int(rgcn_results['summary']['legitimate_detected_by_RGCN']),
                    "fraud_rate_of_RGCN": float(rgcn_results['summary']['fraud_rate'])
                }
            }
        )
        
        # Print completion message with proper None handling
        f1 = result.metrics['RGCN']['f1']
        recall = result.metrics['RGCN']['recall']
        auc = result.metrics['RGCN']['auc']
        
        if f1 is not None and recall is not None and auc is not None:
            print(f"Analysis complete. F1: {f1:.3f}, Recall: {recall:.3f}, AUC: {auc:.3f}")
        else:
            print("Analysis complete. No ground truth available - metrics not calculated.")
        return result
        
    except Exception as e:
        print(f"Error in analyze_transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)