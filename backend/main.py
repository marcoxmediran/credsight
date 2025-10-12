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
            tx_dict = {
                "TransactionID": transaction.TransactionID,
                "isFraud": transaction.TrueLabel,  # Use TrueLabel as isFraud for RGCN
                # Add default values for required features (RGCN will handle missing ones)
                "C14": 0, "C1": 0, "V317": 0, "V308": 0, "card6": "unknown", "V306": 0, "V126": 0,
                "C11": 0, "C6": 0, "V282": 0, "TransactionAmt": 0, "V53": 0, "P_emaildomain": "unknown",
                "card5": 0, "C13": 0, "V35": 0, "V280": 0, "V279": 0, "card2": 0, "V258": 0, "M6": "unknown",
                "V90": 0, "V82": 0, "TransactionDT": 0, "C2": 0, "V87": 0, "V294": 0, "C12": 0, "V313": 0, "id_06": "unknown"
            }
            transactions_data.append(tx_dict)
        
        # Convert to DataFrame and then to CSV string
        df = pd.DataFrame(transactions_data)
        csv_data = df.to_csv(index=False)
        
        print(f"Processing {len(batch.transactions)} transactions with RGCN model...")
        
        # Call RGCN analysis function
        rgcn_results = analyze_transactions(csv_data)
        
        # Convert RGCN results to the expected API format
        processed_transactions = []
        for result in rgcn_results['transactions']:
            processed_transactions.append({
                "TransactionID": result['TransactionID'],
                "TrueLabel": result['Actual_isFraud'],
                "RGCN_Prediction": result['Predicted_isFraud'],
                "RGCN_Confidence": result['Fraud_Probability'],
                # ERGCN not yet implemented - set to null
                "ERGCN_Prediction": None,
                "ERGCN_Confidence": None
            })
        
        # Prepare final response
        result = AnalysisResult(
            transactions=processed_transactions,
            metrics={
                "RGCN": {
                    "recall": rgcn_results['metrics']['recall'],
                    "f1": rgcn_results['metrics']['f1'],
                    "auc": rgcn_results['metrics']['auc']
                },
                # ERGCN not yet implemented - set to null
                "ERGCN": None, 
                "summary": {
                    "total_transactions": rgcn_results['summary']['total_transactions'],
                    "fraud_detected_by_RGCN": rgcn_results['summary']['fraud_detected_by_RGCN'],
                    "legitimate_detected_by_RGCN": rgcn_results['summary']['legitimate_detected_by_RGCN'],
                    "fraud_rate_of_RGCN": rgcn_results['summary']['fraud_rate']
                }
            }
        )
        
        print("=====================================================================")
        print("---------------------------------------------------------------------")
        print(f"Analysis complete. Processed {len(batch.transactions)} transactions.")
        print(f"Recall: {result.metrics['RGCN']['recall']}, F1: {result.metrics['RGCN']['f1']}, AUC: {result.metrics['RGCN']['auc']}")
        print("=====================================================================")
        print("---------------------------------------------------------------------")
        return result
        
    except Exception as e:
        print(f"Error in analyze_transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)