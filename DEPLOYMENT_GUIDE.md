# Fraud Detection Analysis - Deployment Guide

## Overview
This guide covers deploying your R-GCN and ERGCN models for integration with the Next.js frontend.

## Architecture
```
Next.js Frontend → API Route → FastAPI Backend → PyTorch Models
```

## Deployment Options

### Option 1: Google Colab + ngrok (Recommended for Development)

1. **Setup Colab Environment**
   ```python
   # In your Colab notebook
   !pip install fastapi uvicorn pyngrok torch-geometric scikit-learn scipy
   !ngrok authtoken YOUR_NGROK_TOKEN  # Get from https://ngrok.com/
   ```

2. **Upload Backend Files to Colab**
   - Upload `backend/model_inference.py` and `backend/colab_integration.py`
   - Upload your trained model files (.pth)

3. **Start the Server**
   ```python
   from colab_integration import start_colab_server
   
   model_paths = {
       "rgcn": "/content/drive/MyDrive/models/rgcn_model.pth",
       "ergcn": "/content/drive/MyDrive/models/ergcn_model.pth"
   }
   
   start_colab_server(model_paths, ngrok_token="your_token")
   ```

4. **Update Frontend Configuration**
   - Copy the ngrok URL from Colab output
   - Update `.env.local`: `BACKEND_URL=https://your-id.ngrok.io`

### Option 2: Local Development

1. **Setup Python Environment**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Place Model Files**
   ```
   backend/
   ├── models/
   │   ├── rgcn_model.pth
   │   └── ergcn_model.pth
   └── model_inference.py
   ```

3. **Start Backend**
   ```bash
   python model_inference.py
   ```

4. **Start Frontend**
   ```bash
   npm run dev
   ```

### Option 3: AWS Production Deployment

1. **Lambda Function Setup**
   - Package your models and dependencies
   - Use AWS Lambda Layers for PyTorch
   - Deploy FastAPI using Mangum adapter

2. **API Gateway Configuration**
   - Create REST API
   - Configure CORS for your domain
   - Set up custom domain (optional)

3. **S3 Model Storage**
   ```python
   import boto3
   
   s3 = boto3.client('s3')
   
   def load_model_from_s3(bucket, key):
       s3.download_file(bucket, key, '/tmp/model.pth')
       return torch.load('/tmp/model.pth', map_location='cpu')
   ```

## Model Integration Requirements

### Data Preprocessing
Your models should expect this input format:
```python
class Transaction(BaseModel):
    TransactionID: int
    TrueLabel: int  # 0 = legitimate, 1 = fraud
```

### Expected Output Format
```json
{
  "transactions": [
    {
      "TransactionID": 1001,
      "TrueLabel": 0,
      "RGCN": 0,
      "ERGCN": 1
    }
  ],
  "metrics": {
    "RGCN": {
      "recall": 0.34,
      "f1": 0.46,
      "auc": 0.89
    },
    "ERGCN": {
      "recall": 0.46,
      "f1": 0.61,
      "auc": 0.93
    },
    "p_value": 0.021
  }
}
```

## Model Loading Template

Replace the `predict_batch` function in `model_inference.py`:

```python
def predict_batch(model_name: str, transactions: List[Transaction]) -> List[int]:
    model = models[model_name]
    predictions = []
    
    with torch.no_grad():
        for transaction in transactions:
            # 1. Preprocess transaction data
            features = preprocess_transaction(transaction)
            
            # 2. Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32)
            
            # 3. Model inference
            output = model(input_tensor)
            
            # 4. Convert to prediction (0 or 1)
            prediction = torch.sigmoid(output).round().int().item()
            predictions.append(prediction)
    
    return predictions

def preprocess_transaction(transaction: Transaction):
    # Implement your preprocessing logic here
    # This should match your training preprocessing
    pass
```

## Statistical Significance Testing

The system automatically performs paired t-tests between models:
- H0: No difference between R-GCN and ERGCN performance
- H1: Significant difference exists
- p < 0.05 indicates statistical significance

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure FastAPI CORS middleware is configured
   - Check frontend URL in allowed origins

2. **Model Loading Errors**
   - Verify model file paths
   - Check PyTorch version compatibility
   - Ensure models are in evaluation mode

3. **Memory Issues**
   - Use CPU inference for large batches
   - Implement batch processing for large datasets

4. **Timeout Errors**
   - Increase request timeout in Next.js
   - Optimize model inference speed
   - Consider async processing for large datasets

## Performance Optimization

1. **Model Optimization**
   ```python
   # Quantize models for faster inference
   model_quantized = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **Batch Processing**
   ```python
   # Process transactions in batches
   batch_size = 32
   for i in range(0, len(transactions), batch_size):
       batch = transactions[i:i+batch_size]
       predictions.extend(model_inference(batch))
   ```

3. **Caching**
   - Cache model predictions for identical inputs
   - Use Redis for distributed caching

## Security Considerations

1. **API Authentication**
   - Implement API keys for production
   - Use JWT tokens for user authentication

2. **Input Validation**
   - Validate transaction data format
   - Sanitize file uploads

3. **Rate Limiting**
   - Implement request rate limiting
   - Monitor API usage

## Monitoring and Logging

1. **Performance Metrics**
   - Track inference latency
   - Monitor model accuracy over time
   - Log prediction distributions

2. **Error Handling**
   - Comprehensive error logging
   - Graceful fallback mechanisms
   - Health check endpoints

## Next Steps

1. Train your R-GCN and ERGCN models in Colab
2. Save models as .pth files
3. Deploy using Option 1 (Colab + ngrok) for testing
4. Move to AWS for production deployment
5. Implement additional features as needed

For questions or issues, refer to the FastAPI and PyTorch documentation.