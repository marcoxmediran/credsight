"""
Google Colab integration script for model deployment
Use this in your Colab notebook to expose models via ngrok
"""

import torch
import torch_geometric
from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn
import threading
import time
from model_inference import app, load_models

def setup_colab_environment():
    """Setup Colab environment for model serving"""
    
    # Install required packages in Colab
    colab_setup = """
    !pip install fastapi uvicorn pyngrok torch-geometric
    
    # Set ngrok auth token (get from https://ngrok.com/)
    !ngrok authtoken YOUR_NGROK_TOKEN
    """
    
    print("Run this in your Colab notebook:")
    print(colab_setup)

def start_colab_server(model_paths: dict, ngrok_token: str = None):
    """
    Start FastAPI server in Colab with ngrok tunnel
    
    Args:
        model_paths: Dict with 'rgcn' and 'ergcn' model file paths
        ngrok_token: Your ngrok authentication token
    """
    
    # Load your trained models
    global models
    try:
        models['rgcn'] = torch.load(model_paths['rgcn'], map_location='cpu')
        models['ergcn'] = torch.load(model_paths['ergcn'], map_location='cpu')
        
        models['rgcn'].eval()
        models['ergcn'].eval()
        
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Set ngrok token if provided
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"🌐 Public URL: {public_url}")
    print(f"📋 Update your Next.js API endpoint to: {public_url}/analyze")
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example usage in Colab:
colab_example = '''
# In your Colab notebook:

from colab_integration import start_colab_server

# Define paths to your trained models
model_paths = {
    "rgcn": "/content/drive/MyDrive/models/rgcn_model.pth",
    "ergcn": "/content/drive/MyDrive/models/ergcn_model.pth"
}

# Start the server (replace with your actual ngrok token)
start_colab_server(model_paths, ngrok_token="your_ngrok_token_here")
'''

if __name__ == "__main__":
    print("Colab Integration Setup")
    print("=" * 50)
    print(colab_example)