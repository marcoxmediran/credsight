from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Transaction(BaseModel):
    class Config:
        extra = 'allow'  # Allow additional fields from CSV
    
    TransactionID: int
    TrueLabel: Optional[int] = None
    isFraud: Optional[int] = None

class TransactionBatch(BaseModel):
    transactions: List[Transaction]

class ModelMetrics(BaseModel):
    recall: Optional[float]
    f1: Optional[float]
    auc: Optional[float]

class AnalysisResult(BaseModel):
    transactions: List[Dict[str, Any]]
    metrics: Dict[str, Any]