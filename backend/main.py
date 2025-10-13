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
from model import FraudDetectionModel
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import tempfile
import os

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
        
        print(f"Processing {len(batch.transactions)} transactions with RGCN and ERGCN models...")
        
        # Call RGCN analysis function
        rgcn_results = analyze_transactions(csv_data)
        
        # Call FraudDetectionModel
        try:
            # Save CSV to temp file for model
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp:
                tmp.write(csv_data)
                tmp_path = tmp.name
            
            # Initialize and run model
            detector = FraudDetectionModel(model_path="ERGCN_files/training_v12_model.pth")
            results_df = detector.predict(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Convert model results to expected format
            ergcn_results = {
                'transactions': [],
                'metrics': {}
            }
            
            for _, row in results_df.iterrows():
                ergcn_results['transactions'].append({
                    'TransactionID': row['TransactionID'],
                    'Predicted_isFraud': int(row['Predicted_Label']),
                    'Fraud_Probability': float(row['Raw_Score'])
                })
            
            # Calculate metrics
            true_labels = results_df['True_Label'].values
            predicted_labels = results_df['Predicted_Label'].values
            raw_scores = results_df['Raw_Score'].values
            
            ergcn_results['metrics'] = {
                'recall': float(recall_score(true_labels, predicted_labels)),
                'f1': float(f1_score(true_labels, predicted_labels)),
                'auc': float(roc_auc_score(true_labels, raw_scores))
            }
            
            print(f"ERGCN analysis completed successfully")
        except Exception as e:
            print(f"ERGCN analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # Create fallback ERGCN results
            ergcn_results = {
                'transactions': [{
                    'TransactionID': tx['TransactionID'],
                    'Predicted_isFraud': 0,
                    'Fraud_Probability': 0.0
                } for tx in rgcn_results['transactions']],
                'metrics': {'f1': None, 'recall': None, 'auc': None}
            }
        
        # Convert results to the expected API format
        processed_transactions = []
        for i, rgcn_result in enumerate(rgcn_results['transactions']):
            # Get corresponding ERGCN result
            ergcn_result = ergcn_results['transactions'][i] if i < len(ergcn_results['transactions']) else {
                'Predicted_isFraud': 0, 'Fraud_Probability': 0.0
            }
            
            # Handle both protocols: with and without ground truth
            true_label = None
            if rgcn_result['Actual_isFraud'] is not None and not pd.isna(rgcn_result['Actual_isFraud']):
                true_label = int(rgcn_result['Actual_isFraud'])
            
            # Handle NaN TransactionID
            transaction_id = rgcn_result['TransactionID']
            if pd.isna(transaction_id):
                continue  # Skip rows with missing TransactionID
            
            processed_transactions.append({
                "TransactionID": int(transaction_id),
                "TrueLabel": true_label,
                "RGCN_Prediction": int(rgcn_result['Predicted_isFraud']),
                "RGCN_Confidence": float(rgcn_result['Fraud_Probability']),
                "ERGCN_Prediction": int(ergcn_result['Predicted_isFraud']),
                "ERGCN_Confidence": float(ergcn_result['Fraud_Probability'])
            })
        
        # Prepare final response
        # Handle None metrics for no ground truth cases
        rgcn_metrics = rgcn_results['metrics']
        ergcn_metrics = ergcn_results['metrics']
        
        # Calculate ERGCN summary statistics
        ergcn_fraud_count = sum(1 for tx in processed_transactions if tx['ERGCN_Prediction'] == 1)
        ergcn_legitimate_count = len(processed_transactions) - ergcn_fraud_count
        ergcn_fraud_rate = ergcn_fraud_count / len(processed_transactions) if len(processed_transactions) > 0 else 0.0
        
        result = AnalysisResult(
            transactions=processed_transactions,
            metrics={
                "RGCN": {
                    "recall": rgcn_metrics['recall'],
                    "f1": rgcn_metrics['f1'],
                    "auc": rgcn_metrics['auc']
                },
                "ERGCN": {
                    "recall": ergcn_metrics['recall'],
                    "f1": ergcn_metrics['f1'],
                    "auc": ergcn_metrics['auc']
                },
                "summary": {
                    "total_transactions": int(rgcn_results['summary']['total_transactions']),
                    "fraud_detected_by_RGCN": int(rgcn_results['summary']['fraud_detected_by_RGCN']),
                    "legitimate_detected_by_RGCN": int(rgcn_results['summary']['legitimate_detected_by_RGCN']),
                    "fraud_rate_of_RGCN": float(rgcn_results['summary']['fraud_rate']),
                    "fraud_detected_by_ERGCN": ergcn_fraud_count,
                    "legitimate_detected_by_ERGCN": ergcn_legitimate_count,
                    "fraud_rate_of_ERGCN": float(ergcn_fraud_rate)
                }
            }
        )
        
        # Print completion message with proper None handling
        rgcn_f1 = result.metrics['RGCN']['f1']
        rgcn_recall = result.metrics['RGCN']['recall']
        rgcn_auc = result.metrics['RGCN']['auc']
        ergcn_f1 = result.metrics['ERGCN']['f1']
        ergcn_recall = result.metrics['ERGCN']['recall']
        ergcn_auc = result.metrics['ERGCN']['auc']
        
        if rgcn_f1 is not None and rgcn_recall is not None and rgcn_auc is not None:
            print(f"Analysis complete. RGCN - F1: {rgcn_f1:.3f}, Recall: {rgcn_recall:.3f}, AUC: {rgcn_auc:.3f}")
            if ergcn_f1 is not None and ergcn_recall is not None and ergcn_auc is not None:
                print(f"ERGCN - F1: {ergcn_f1:.3f}, Recall: {ergcn_recall:.3f}, AUC: {ergcn_auc:.3f}")
        else:
            print("Analysis complete. No ground truth available - metrics not calculated.")
        return result
        
    except Exception as e:
        print(f"Error in analyze_transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)