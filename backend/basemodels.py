from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Transaction(BaseModel):
    TransactionID: int
    TrueLabel: Optional[int] = None

class TransactionBatch(BaseModel):
    transactions: List[Transaction]

class ModelMetrics(BaseModel):
    recall: float
    f1: float
    auc: float

class AnalysisResult(BaseModel):
    transactions: List[Dict[str, Any]]
    metrics: Dict[str, Any]