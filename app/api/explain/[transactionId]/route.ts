import { NextRequest, NextResponse } from 'next/server'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ transactionId: string }> }
) {
  try {
    const { transactionId } = await params
    const { searchParams } = new URL(request.url)
    const modelType = searchParams.get('model_type') || 'both'
    
    const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'
    
    const response = await fetch(
      `${BACKEND_URL}/explain/${transactionId}?model_type=${modelType}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )
    
    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`)
    }
    
    const explanation = await response.json()
    return NextResponse.json(explanation)
    
  } catch (error) {
    console.error('Explanation API error:', error)
    
    // Fallback mock explanation for development
    const mockExplanation = {
      transaction_id: parseInt(transactionId),
      explanations: {
        rgcn: {
          model: "R-GCN",
          prediction: Math.random() > 0.5 ? 1 : 0,
          fraud_probability: Math.random(),
          feature_importance: [
            { feature: "TransactionAmt", importance: Math.random() * 0.8, rank: 1 },
            { feature: "card1", importance: Math.random() * 0.6, rank: 2 },
            { feature: "addr1", importance: Math.random() * 0.4, rank: 3 },
            { feature: "P_emaildomain", importance: Math.random() * 0.3, rank: 4 },
            { feature: "C1", importance: Math.random() * 0.2, rank: 5 }
          ],
          edge_importance: [
            { connected_transaction: 1001, edge_importance: 0.7, relationship_type: "similar_transaction" },
            { connected_transaction: 1002, edge_importance: 0.5, relationship_type: "similar_transaction" }
          ],
          top_reasons: [
            "High transaction amount indicates fraud risk",
            "Unusual card information pattern detected",
            "Billing address shows suspicious activity"
          ]
        },
        ergcn: {
          model: "ERGCN",
          prediction: Math.random() > 0.5 ? 1 : 0,
          fraud_probability: Math.random(),
          feature_importance: [
            { feature: "TransactionAmt", importance: Math.random() * 0.9, rank: 1 },
            { feature: "D1", importance: Math.random() * 0.7, rank: 2 },
            { feature: "V1", importance: Math.random() * 0.5, rank: 3 },
            { feature: "card2", importance: Math.random() * 0.4, rank: 4 },
            { feature: "C2", importance: Math.random() * 0.3, rank: 5 }
          ],
          edge_importance: [
            { connected_transaction: 1003, edge_importance: 0.8, relationship_type: "similar_transaction" },
            { connected_transaction: 1004, edge_importance: 0.6, relationship_type: "similar_transaction" }
          ],
          top_reasons: [
            "Transaction amount and timing suggest fraud",
            "Enhanced temporal features show anomaly",
            "Graph relationships indicate suspicious network"
          ]
        }
      },
      computation_time: 0.5
    }
    
    return NextResponse.json(mockExplanation)
  }
}