"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2, AlertTriangle, CheckCircle, X } from "lucide-react"
import { useState } from "react"

type TransactionSnapshot = {
  RGCN?: number | null
  ERGCN?: number | null
  RGCN_Confidence?: number | null
  ERGCN_Confidence?: number | null
}

interface ExplanationModalProps {
  isOpen: boolean
  onClose: () => void
  transactionId: number | null
  explanationData: any
  isLoading: boolean
  transactionDetails?: TransactionSnapshot | null
}

export default function ExplanationModal({
  isOpen,
  onClose,
  transactionId,
  explanationData,
  isLoading,
  transactionDetails
}: ExplanationModalProps) {
  const [activeTab, setActiveTab] = useState("comparison")
  
  if (!isOpen || !transactionId) return null

  const getModelSnapshot = (modelKey: "rgcn" | "ergcn") => {
    const explanationModel = explanationData?.explanations?.[modelKey]
    const fallbackPrediction = modelKey === "rgcn" ? transactionDetails?.RGCN : transactionDetails?.ERGCN
    const fallbackConfidence = modelKey === "rgcn" ? transactionDetails?.RGCN_Confidence : transactionDetails?.ERGCN_Confidence
    const base = {
      prediction: fallbackPrediction ?? null,
      fraud_probability: fallbackConfidence ?? null
    }
    if (!explanationModel) {
      return base
    }
    return {
      ...base,
      ...explanationModel
    }
  }

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

    const probability = typeof modelData.fraud_probability === "number" ? modelData.fraud_probability : null
    const confidenceColor = probability !== null && probability > 0.5 ? "text-red-500" : "text-green-500"
    const predictionValue = modelData.prediction
    const predictionLabel = predictionValue === 1 ? "Fraud" : predictionValue === 0 ? "Legitimate" : "Unknown"
    const badgeVariant = predictionValue === 1 ? "destructive" : predictionValue === 0 ? "secondary" : "outline"

    return (
      <div className="space-y-6">
        {/* Prediction Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{modelName} Prediction</span>
              <Badge variant={badgeVariant}>
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
                    {probability !== null ? `${(probability * 100).toFixed(1)}%` : "N/A"}
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-3">
                  <div 
                    className="bg-primary h-3 rounded-full transition-all duration-300" 
                    style={{ width: probability !== null ? `${Math.min(Math.max(probability * 100, 0), 100)}%` : "0%" }}
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

  const buildSnapshotSummary = (snapshot: { prediction?: number | null, fraud_probability?: number | null }, threshold = 0.5) => {
    const probability = typeof snapshot?.fraud_probability === "number" ? snapshot.fraud_probability : null
    return {
      probability,
      confidenceText: probability !== null ? `${(probability * 100).toFixed(1)}%` : "N/A",
      badgeVariant: snapshot?.prediction === 1 ? "destructive" : snapshot?.prediction === 0 ? "secondary" : "outline",
      predictionLabel: snapshot?.prediction === 1 ? "Fraud" : snapshot?.prediction === 0 ? "Legitimate" : "Unknown",
      meetsThreshold: probability !== null && probability >= threshold
    }
  }

  const rgcnSnapshot = getModelSnapshot("rgcn")
  const ergcnSnapshot = getModelSnapshot("ergcn")
  const rgcnSummary = buildSnapshotSummary(rgcnSnapshot, 0.5)
  const ergcnSummary = buildSnapshotSummary(ergcnSnapshot, 0.6)

  const hasSnapshotData = Boolean(transactionDetails || explanationData?.explanations)
  const hasDetailedInsights = Boolean(explanationData?.explanations)

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

        {isLoading && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="ml-2 text-sm text-muted-foreground">Generating interpretation...</span>
          </div>
        )}

        {explanationData?.error ? (
          <div className="text-center py-12">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-lg font-medium">Interpretation Failed</p>
            <p className="text-muted-foreground">{explanationData.error}</p>
          </div>
        ) : hasSnapshotData ? (
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
                        <Badge variant={rgcnSummary.badgeVariant}>
                          {rgcnSummary.predictionLabel}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>Confidence:</span>
                        <span className="font-bold">{rgcnSummary.confidenceText}</span>
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
                        <Badge variant={ergcnSummary.badgeVariant}>
                          {ergcnSummary.predictionLabel}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>Confidence:</span>
                        <span className="font-bold">{ergcnSummary.confidenceText}</span>
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
                        {rgcnSummary.meetsThreshold && (
                          <Badge variant="destructive" className="text-xs">Fraud</Badge>
                        )}
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-purple-500 font-medium">ERGCN Threshold:</span>
                      <div className="flex items-center gap-2">
                        <span className="font-bold">60%</span>
                        {ergcnSummary.meetsThreshold && (
                          <Badge variant="destructive" className="text-xs">Fraud</Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {hasDetailedInsights ? (
                <div className="grid gap-6 md:grid-cols-2">
                  {renderModelExplanation(explanationData.explanations.rgcn, "R-GCN Detailed Insights")}
                  {renderModelExplanation(explanationData.explanations.ergcn, "ERGCN Detailed Insights")}
                </div>
              ) : (
                <div className="text-center text-sm text-muted-foreground py-6">
                  Detailed explanations will appear once available.
                </div>
              )}
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