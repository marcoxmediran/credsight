"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2, AlertTriangle, CheckCircle, X } from "lucide-react"
import { useState } from "react"

interface ExplanationModalProps {
  isOpen: boolean
  onClose: () => void
  transactionId: number | null
  explanationData: any
  isLoading: boolean
}

export default function ExplanationModal({
  isOpen,
  onClose,
  transactionId,
  explanationData,
  isLoading
}: ExplanationModalProps) {
  const [activeTab, setActiveTab] = useState("comparison")
  
  if (!isOpen || !transactionId) return null

  const renderFeatureImportance = (features: any[], modelName: string) => {
    if (!features || features.length === 0) {
      return <p className="text-muted-foreground">No feature importance data available</p>
    }

    return (
      <div className="space-y-3">
        {features.slice(0, 8).map((feature, index) => (
          <div key={index} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">{feature.feature}</span>
              <span className="text-sm text-muted-foreground">
                {(feature.importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div 
                className="bg-primary h-2 rounded-full transition-all duration-300" 
                style={{ width: `${Math.abs(feature.importance) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    )
  }

  const renderTopReasons = (reasons: string[], prediction: number) => {
    if (!reasons || reasons.length === 0) {
      return <p className="text-muted-foreground">No explanation available</p>
    }

    return (
      <div className="space-y-3">
        {reasons.map((reason, index) => (
          <div key={index} className="flex items-start gap-3 p-3 bg-muted rounded-lg">
            {prediction === 1 ? (
              <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
            ) : (
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
            )}
            <p className="text-sm">{reason}</p>
          </div>
        ))}
      </div>
    )
  }

  const renderModelExplanation = (modelData: any, modelName: string) => {
    if (!modelData || modelData.error) {
      return (
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground">
              {modelData?.error || "No explanation available"}
            </p>
          </CardContent>
        </Card>
      )
    }

    const confidenceColor = modelData.fraud_probability > 0.5 ? "text-red-500" : "text-green-500"
    const predictionLabel = modelData.prediction === 1 ? "Fraud" : "Legitimate"

    return (
      <div className="space-y-6">
        {/* Prediction Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{modelName} Prediction</span>
              <Badge variant={modelData.prediction === 1 ? "destructive" : "secondary"}>
                {predictionLabel}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Fraud Probability</span>
                  <span className={`text-sm font-bold ${confidenceColor}`}>
                    {(modelData.fraud_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-3">
                  <div 
                    className="bg-primary h-3 rounded-full transition-all duration-300" 
                    style={{ width: `${modelData.fraud_probability * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Top Reasons */}
        <Card>
          <CardHeader>
            <CardTitle>Key Factors</CardTitle>
          </CardHeader>
          <CardContent>
            {renderTopReasons(modelData.top_reasons, modelData.prediction)}
          </CardContent>
        </Card>

        {/* Feature Importance */}
        <Card>
          <CardHeader>
            <CardTitle>Feature Importance</CardTitle>
          </CardHeader>
          <CardContent>
            {renderFeatureImportance(modelData.feature_importance, modelName)}
          </CardContent>
        </Card>

        {/* Connected Transactions */}
        {modelData.edge_importance && modelData.edge_importance.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Related Transactions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {modelData.edge_importance.slice(0, 3).map((edge: any, index: number) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-sm">Transaction {edge.connected_transaction}</span>
                    <span className="text-sm text-muted-foreground">
                      Influence: {(edge.edge_importance * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-background border rounded-lg shadow-lg max-w-4xl max-h-[90vh] overflow-y-auto w-full mx-4">
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-lg font-semibold">
            Confidence Details - Transaction {transactionId}
          </h2>
          <Button variant="ghost" size="sm" onClick={onClose} className="cursor-pointer">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="p-6">
          {/* Threshold Formula */}
          <div className="mb-6 p-4 bg-muted rounded-lg">
            <h3 className="text-sm font-semibold mb-2">Fraud Detection Formula:</h3>
            <p className="text-sm text-muted-foreground">
              If <span className="font-mono">Confidence ≥ Threshold</span> → <span className="text-red-500 font-medium">Fraud</span>
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              R-GCN: ≥50% | ERGCN: ≥60%
            </p>
          </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="ml-2">Generating interpretation...</span>
          </div>
        ) : explanationData?.error ? (
          <div className="text-center py-12">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-lg font-medium">Interpretation Failed</p>
            <p className="text-muted-foreground">{explanationData.error}</p>
          </div>
        ) : explanationData?.explanations ? (
          <div className="w-full">
            <div className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-blue-500">R-GCN Model</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span>Prediction:</span>
                        <Badge variant={explanationData.explanations.rgcn?.prediction === 1 ? "destructive" : "secondary"}>
                          {explanationData.explanations.rgcn?.prediction === 1 ? "Fraud" : "Legitimate"}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>Confidence:</span>
                        <span className="font-bold">
                          {((explanationData.explanations.rgcn?.fraud_probability || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-purple-500">ERGCN Model</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span>Prediction:</span>
                        <Badge variant={explanationData.explanations.ergcn?.prediction === 1 ? "destructive" : "secondary"}>
                          {explanationData.explanations.ergcn?.prediction === 1 ? "Fraud" : "Legitimate"}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>Confidence:</span>
                        <span className="font-bold">
                          {((explanationData.explanations.ergcn?.fraud_probability || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Model Thresholds</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-blue-500 font-medium">R-GCN Threshold:</span>
                      <div className="flex items-center gap-2">
                        <span className="font-bold">50%</span>
                        {((explanationData.explanations.rgcn?.fraud_probability || 0) * 100) >= 50 && (
                          <Badge variant="destructive" className="text-xs">Fraud</Badge>
                        )}
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-purple-500 font-medium">ERGCN Threshold:</span>
                      <div className="flex items-center gap-2">
                        <span className="font-bold">60%</span>
                        {((explanationData.explanations.ergcn?.fraud_probability || 0) * 100) >= 60 && (
                          <Badge variant="destructive" className="text-xs">Fraud</Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No interpretation data available</p>
          </div>
        )}
        </div>
      </div>
    </div>
  )
}