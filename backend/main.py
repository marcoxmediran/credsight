from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import pandas as pd
from io import StringIO

from basemodels import AnalysisResult
from RGCN import analyze_transactions as analyze_transactions_rgcn
from ERGCN import analyze_transactions as analyze_transactions_ergcn

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
async def analyze_transactions_endpoint(file: UploadFile = File(...)):
    """
    Analyze transactions using RGCN and ERGCN models.
    Accepts CSV file directly from frontend - no transformation needed.
    This eliminates the double transformation issue (CSV -> JSON -> CSV).
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read CSV content directly
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        
        print("\n" + "="*80)
        print("RECEIVED CSV FILE")
        print("="*80)
        print(f"Filename: {file.filename}")
        print(f"CSV data length: {len(csv_data)} characters")
        print(f"CSV first 500 chars: {csv_data[:500]}")
        
        # Quick validation - check if CSV has data
        lines = csv_data.strip().split('\n')
        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="CSV file must contain at least a header and one data row")
        
        print(f"CSV rows: {len(lines)} (including header)")
        print(f"Expected data rows: {len(lines) - 1}")
        
        # Call RGCN analysis function
        print("\n" + "="*80)
        print("CALLING RGCN MODEL")
        print("="*80)
        rgcn_results = analyze_transactions_rgcn(csv_data)
        
        # Call ERGCN analysis function
        print("\n" + "="*80)
        print("CALLING ERGCN MODEL")
        print("="*80)
        try:
            ergcn_results = analyze_transactions_ergcn(csv_data)
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
        skipped_count = 0
        
        # Ensure both results have the same length
        min_length = min(len(rgcn_results['transactions']), len(ergcn_results['transactions']))
        max_length = max(len(rgcn_results['transactions']), len(ergcn_results['transactions']))
        
        if min_length != max_length:
            print(f"WARNING: Result length mismatch - RGCN: {len(rgcn_results['transactions'])}, ERGCN: {len(ergcn_results['transactions'])}")
        
        for i in range(min_length):
            rgcn_result = rgcn_results['transactions'][i]
            # Get corresponding ERGCN result
            ergcn_result = ergcn_results['transactions'][i] if i < len(ergcn_results['transactions']) else {
                'Predicted_isFraud': 0, 'Fraud_Probability': 0.0
            }
            
            # Handle both protocols: with and without ground truth
            true_label = None
            if rgcn_result.get('Actual_isFraud') is not None and not pd.isna(rgcn_result.get('Actual_isFraud')):
                true_label = int(rgcn_result['Actual_isFraud'])
            
            # Handle NaN TransactionID - convert to int safely
            transaction_id = rgcn_result.get('TransactionID')
            if transaction_id is None or pd.isna(transaction_id):
                skipped_count += 1
                print(f"WARNING: Skipping transaction at index {i} due to missing TransactionID")
                continue  # Skip rows with missing TransactionID
            
            try:
                transaction_id = int(transaction_id)
            except (ValueError, TypeError):
                skipped_count += 1
                print(f"WARNING: Skipping transaction at index {i} due to invalid TransactionID: {transaction_id}")
                continue
            
            processed_transactions.append({
                "TransactionID": transaction_id,
                "TrueLabel": true_label,
                "RGCN_Prediction": int(rgcn_result.get('Predicted_isFraud', 0)),
                "RGCN_Confidence": float(rgcn_result.get('Fraud_Probability', 0.0)),
                "ERGCN_Prediction": int(ergcn_result.get('Predicted_isFraud', 0)),
                "ERGCN_Confidence": float(ergcn_result.get('Fraud_Probability', 0.0))
            })
        
        if skipped_count > 0:
            print(f"WARNING: Skipped {skipped_count} transactions due to missing/invalid TransactionID")
            print(f"Processed {len(processed_transactions)} transactions out of {min_length} total results")
        
        # Prepare final response
        # Handle None metrics for no ground truth cases
        rgcn_metrics = rgcn_results['metrics']
        ergcn_metrics = ergcn_results['metrics']
        
        # IMPORTANT: Use metrics from model results (calculated on ALL input data)
        # These metrics are calculated on the full dataset sent to the models,
        # not on the filtered processed_transactions
        print(f"\nMetrics from models (calculated on {rgcn_results['summary']['total_transactions']} transactions):")
        print(f"  RGCN - F1: {rgcn_metrics.get('f1')}, Recall: {rgcn_metrics.get('recall')}, AUC: {rgcn_metrics.get('auc')}")
        print(f"  ERGCN - F1: {ergcn_metrics.get('f1')}, Recall: {ergcn_metrics.get('recall')}, AUC: {ergcn_metrics.get('auc')}")
        print(f"\nConfusion matrices from both models")
        print(f"RGCN")
        print(f"{rgcn_results['confusion_matrix']}")
        print(f"ERGCN")
        print(f"{ergcn_results['confusion_matrix']}")
        
        # Convert confusion matrices from numpy arrays to labeled objects
        def convert_confusion_matrix(cm):
            """Convert 2x2 numpy confusion matrix to labeled object format.
            
            Args:
                cm: numpy array of shape (2, 2) with format [[TN, FP], [FN, TP]]
                    or None if not available
            
            Returns:
                dict with keys: true_negative, false_positive, false_negative, true_positive
                or None if cm is None/empty
            """
            if cm is None:
                return None
            try:
                # Handle numpy array or list/tuple - convert to Python int for JSON serialization
                # Format: [[TN, FP], [FN, TP]]
                # Access as 2D array and convert to int
                return {
                    "true_negative": int(cm[0][0]),
                    "false_positive": int(cm[0][1]),
                    "false_negative": int(cm[1][0]),
                    "true_positive": int(cm[1][1])
                }
            except (IndexError, TypeError, ValueError, AttributeError) as e:
                print(f"Warning: Could not convert confusion matrix: {e}")
                return None
        
        # Extract and convert confusion matrices
        rgcn_cm = rgcn_results.get('confusion_matrix')
        ergcn_cm = ergcn_results.get('confusion_matrix')
        
        confusion_matrices = {
            "RGCN": convert_confusion_matrix(rgcn_cm),
            "ERGCN": convert_confusion_matrix(ergcn_cm)
        }
        
        # Calculate ERGCN summary statistics from processed transactions
        # Note: These counts may differ from model results if transactions were filtered
        ergcn_fraud_count = sum(1 for tx in processed_transactions if tx['ERGCN_Prediction'] == 1)
        ergcn_legitimate_count = len(processed_transactions) - ergcn_fraud_count
        ergcn_fraud_rate = ergcn_fraud_count / len(processed_transactions) * 100 if len(processed_transactions) > 0 else 0.0
        
        # Use model summary for RGCN (more accurate as it's from the full dataset)
        # But also calculate from processed for consistency check
        rgcn_fraud_from_processed = sum(1 for tx in processed_transactions if tx['RGCN_Prediction'] == 1)
        
        if skipped_count > 0:
            print(f"\nWARNING: Summary statistics may differ due to {skipped_count} skipped transactions")
            print(f"  Model processed: {rgcn_results['summary']['total_transactions']} transactions")
            print(f"  Returned to API: {len(processed_transactions)} transactions")
        
        # Build metrics dict with confusion matrices
        metrics_dict = {
            "RGCN": {
                "recall": rgcn_metrics.get('recall'),
                "f1": rgcn_metrics.get('f1'),
                "auc": rgcn_metrics.get('auc')
            },
            "ERGCN": {
                "recall": ergcn_metrics.get('recall'),
                "f1": ergcn_metrics.get('f1'),
                "auc": ergcn_metrics.get('auc')
            },
            "summary": {
                "total_transactions": int(rgcn_results['summary']['total_transactions']),
                "fraud_detected_by_RGCN": int(rgcn_results['summary']['fraud_detected_by_RGCN']),
                "legitimate_detected_by_RGCN": int(rgcn_results['summary']['legitimate_detected_by_RGCN']),
                "fraud_rate_of_RGCN": float(rgcn_results['summary']['fraud_rate']),
                "fraud_detected_by_ERGCN": ergcn_fraud_count,
                "legitimate_detected_by_ERGCN": ergcn_legitimate_count,
                "fraud_rate_of_ERGCN": float(ergcn_fraud_rate)
            },
            "confusion_matrices": confusion_matrices
        }
        
        # Debug: Print confusion matrices before creating result
        print(f"DEBUG: Confusion matrices being added: {confusion_matrices}")
        print(f"DEBUG: Metrics dict keys: {metrics_dict.keys()}")
        
        result = AnalysisResult(
            transactions=processed_transactions,
            metrics=metrics_dict
        )
        
        # Debug: Verify confusion_matrices are in the result
        print(f"DEBUG: Result metrics keys: {result.metrics.keys()}")
        if 'confusion_matrices' in result.metrics:
            print(f"DEBUG: Confusion matrices in result: {result.metrics['confusion_matrices']}")
        else:
            print("DEBUG: WARNING - confusion_matrices NOT in result.metrics!")
        
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
