"use client"

import type React from "react"

import { useState, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AnimatedCard } from "@/components/AnimatedCard"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Upload, Download, ArrowUpDown, Search, ChevronLeft, ChevronRight, Info, HelpCircle, Expand, AlertCircle, AlertTriangle } from "lucide-react"
import Link from "next/link"
import LiquidEther from "@/components/LiquidEther"
import ModelEvaluationPlaceholder from "@/components/ModelEvaluationPlaceholder"
import ExplanationModal from "@/components/ExplanationModal"

type TransactionData = {
  TransactionID: number
  TrueLabel?: number | null
  RGCN?: number | null
  ERGCN?: number | null
  RGCN_Confidence?: number | null
  ERGCN_Confidence?: number | null
  [key: string]: any  // Allow any additional CSV columns
}

type ModelMetrics = {
  recall: number | null
  f1: number | null
  auc: number | null
}

type SummaryMetrics = {
  total_transactions: number
  fraud_detected_by_RGCN: number
  legitimate_detected_by_RGCN: number
  fraud_rate_of_RGCN?: number
  fraud_detected_by_ERGCN?: number
  legitimate_detected_by_ERGCN?: number
  fraud_rate_of_ERGCN?: number
}

type AnalysisMetrics = {
  RGCN: ModelMetrics
  ERGCN: ModelMetrics | null
  p_value: number
  summary?: SummaryMetrics
}

type AnalysisResult = {
  transactions: TransactionData[]
  metrics: AnalysisMetrics
  error?: string
}

type FilterType = "all" | "legit" | "fraud" | "both_correct" | "both_incorrect" | "rgcn_correct" | "ergcn_correct" | "rgcn_incorrect" | "ergcn_incorrect" | "rgcn_fraud" | "rgcn_legit" | "ergcn_fraud" | "ergcn_legit"

export default function AnalyzePage() {
  const [data, setData] = useState<TransactionData[]>([])
  const [filteredData, setFilteredData] = useState<TransactionData[]>([])
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [filterType, setFilterType] = useState<FilterType>("all")
  const [sortConfig, setSortConfig] = useState<{
    key: keyof TransactionData
    direction: "asc" | "desc"
  } | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(10)
  const [showExplanation, setShowExplanation] = useState(false)
  const [selectedTransaction, setSelectedTransaction] = useState<number | null>(null)
  const [selectedTransactionData, setSelectedTransactionData] = useState<TransactionData | null>(null)
  const [explanationData, setExplanationData] = useState<any>(null)
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false)
  const [isProcessingFile, setIsProcessingFile] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [hasGroundTruth, setHasGroundTruth] = useState(true)
  const [hasUploadedCSV, setHasUploadedCSV] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [ergcnWarning, setErgcnWarning] = useState<string | null>(null)

  const analyzeWithModels = async (csvFile: File) => {
    setIsAnalyzing(true)
    setAnalysisError(null)
    setErgcnWarning(null)
    try {
      // Send CSV file directly to backend via FormData
      const formData = new FormData()
      formData.append('file', csvFile)

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      })
      const payload = await response.json()
      
      if (!response.ok || payload?.error) {
        const errorMessage = payload?.error || 'Analysis failed'
        setAnalysisError(errorMessage)
        return
      }
      
      const result: AnalysisResult = payload
      const containsGroundTruth = result.transactions?.some((tx) => tx.TrueLabel !== null && tx.TrueLabel !== undefined)
      
      setAnalysisResult(result)
      setData(result.transactions)
      setFilteredData(result.transactions)
      setHasGroundTruth(containsGroundTruth)
      
      const ergcnMetrics = result.metrics?.ERGCN
      const hasErgcnMetrics = Boolean(ergcnMetrics && (ergcnMetrics.recall !== null || ergcnMetrics.f1 !== null || ergcnMetrics.auc !== null))
      const hasErgcnConfidence = result.transactions?.some((tx) => typeof tx.ERGCN_Confidence === 'number')
      
      if (!hasErgcnMetrics || !hasErgcnConfidence) {
        setErgcnWarning("ERGCN analysis data is unavailable. Displaying R-GCN results only.")
      }
    } catch (error) {
      console.error('Analysis failed:', error)
      setAnalysisError('Analysis failed. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = useCallback((file: File) => {
    if (file.type !== "text/csv" && !file.name.endsWith('.csv')) {
      alert("Please upload a CSV file")
      return
    }

    // Check file size and warn for large files
    const fileSizeMB = file.size / (1024 * 1024)
    if (fileSizeMB > 800) {
      alert(`File too large (${fileSizeMB.toFixed(1)}MB). Maximum supported size is 800MB. Please split your file or use a smaller sample.`)
      return
    } else if (fileSizeMB > 100) {
      const proceed = confirm(`Large file detected (${fileSizeMB.toFixed(1)}MB). Processing may take several minutes. Continue?`)
      if (!proceed) return
    }

    // Quick validation: Read first few lines to check for TransactionID column
    setIsProcessingFile(true)
    setProcessingProgress(0)

    const reader = new FileReader()
    
    reader.onload = async (e) => {
      try {
        const text = e.target?.result as string
        const lines = text.split('\n').filter(line => line.trim())
        
        if (lines.length < 2) {
          alert("CSV file must contain at least a header and one data row")
          setIsProcessingFile(false)
          return
        }

        // Quick validation: Check for TransactionID column
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase())
        const hasTransactionID = headers.some(h => h.includes('transactionid'))
        
        if (!hasTransactionID) {
          alert("CSV must contain TransactionID column")
          setIsProcessingFile(false)
          return
        }

        // Check if ground truth column exists (optional)
        const hasGroundTruth = headers.some(h => h.includes('isfraud') || h.includes('truelabel'))
        setHasGroundTruth(hasGroundTruth)

        setProcessingProgress(100)
        setHasUploadedCSV(true)
        
        // Send file directly to backend - no parsing/transformation needed!
        await analyzeWithModels(file)
        
      } catch (error) {
        console.error('File validation error:', error)
        alert("Error validating file. Please ensure it's a valid CSV file.")
      } finally {
        setIsProcessingFile(false)
        setProcessingProgress(0)
      }
    }
    
    reader.onerror = () => {
      alert("Error reading file. Please try again.")
      setIsProcessingFile(false)
      setProcessingProgress(0)
    }
    
    // Only read first 10KB for validation (header + a few rows)
    const blob = file.slice(0, 10240) // First 10KB
    reader.readAsText(blob)
    
    // After validation, send the full file
    // Note: We validate first, then send the full file in analyzeWithModels
  }, [])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileUpload(file)
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileUpload(file)
  }

  const applyFiltersAndSearch = useCallback(() => {
    let result = [...data]

    if (filterType === "legit" && hasGroundTruth) {
      result = result.filter((item) => item.TrueLabel === 0)
    } else if (filterType === "fraud" && hasGroundTruth) {
      result = result.filter((item) => item.TrueLabel === 1)
    } else if (filterType === "both_correct" && analysisResult) {
      result = result.filter((item) => 
        (item.RGCN === item.TrueLabel) && (item.ERGCN === item.TrueLabel)
      )
    } else if (filterType === "both_incorrect" && analysisResult) {
      result = result.filter((item) => 
        (item.RGCN !== item.TrueLabel) && (item.ERGCN !== item.TrueLabel)
      )
    } else if (filterType === "rgcn_correct" && analysisResult) {
      result = result.filter((item) => item.RGCN === item.TrueLabel)
    } else if (filterType === "ergcn_correct" && analysisResult) {
      result = result.filter((item) => item.ERGCN === item.TrueLabel)
    } else if (filterType === "rgcn_incorrect" && analysisResult) {
      result = result.filter((item) => item.RGCN !== item.TrueLabel)
    } else if (filterType === "ergcn_incorrect" && analysisResult) {
      result = result.filter((item) => item.ERGCN !== item.TrueLabel)
    } else if (filterType === "rgcn_fraud" && analysisResult) {
      result = result.filter((item) => item.RGCN === 1)
    } else if (filterType === "rgcn_legit" && analysisResult) {
      result = result.filter((item) => item.RGCN === 0)
    } else if (filterType === "ergcn_fraud" && analysisResult) {
      result = result.filter((item) => item.ERGCN === 1)
    } else if (filterType === "ergcn_legit" && analysisResult) {
      result = result.filter((item) => item.ERGCN === 0)
    }

    const normalizedQuery = searchQuery.trim()
    if (normalizedQuery) {
      result = result.filter((item) => item.TransactionID.toString().includes(normalizedQuery))
    }

    if (sortConfig) {
      result.sort((a, b) => {
        const aValue = a[sortConfig.key]
        const bValue = b[sortConfig.key]

        if (typeof aValue === "number" && typeof bValue === "number") {
          return sortConfig.direction === "asc" ? aValue - bValue : bValue - aValue
        }

        return 0
      })
    }

    setFilteredData(result)
    setCurrentPage(1)
  }, [data, filterType, searchQuery, sortConfig, analysisResult, hasGroundTruth])

  // Re-apply filters, search, and sort whenever inputs change to avoid stale state issues
  useEffect(() => {
    applyFiltersAndSearch()
  }, [applyFiltersAndSearch])

  const handleSort = (key: keyof TransactionData) => {
    setSortConfig((current) => {
      if (!current || current.key !== key) {
        return { key, direction: "asc" }
      }
      if (current.direction === "asc") {
        return { key, direction: "desc" }
      }
      return null
    })
  }

  const explainTransaction = async (transaction: TransactionData) => {
    setIsLoadingExplanation(true)
    setExplanationData(null)
    setSelectedTransaction(transaction.TransactionID)
    setSelectedTransactionData(transaction)
    setShowExplanation(true)
    
    try {
      const response = await fetch(`/api/explain/${transaction.TransactionID}?model_type=both`)
      
      if (response.ok) {
        const explanation = await response.json()
        setExplanationData(explanation)
      } else {
        // Use fallback mock data when API fails
        const mockExplanation = {
          transaction_id: transaction.TransactionID,
          explanations: {
            rgcn: {
              model: "R-GCN",
              prediction: transaction.RGCN ?? (Math.random() > 0.5 ? 1 : 0),
              fraud_probability: transaction.RGCN_Confidence ?? Math.random(),
              feature_importance: [
                { feature: "TransactionAmt", importance: Math.random() * 0.8, rank: 1 },
                { feature: "card1", importance: Math.random() * 0.6, rank: 2 },
                { feature: "addr1", importance: Math.random() * 0.4, rank: 3 }
              ],
              top_reasons: [
                "Transaction amount pattern analysis",
                "Card usage behavior assessment",
                "Address verification results"
              ]
            },
            ergcn: {
              model: "ERGCN",
              prediction: transaction.ERGCN ?? (Math.random() > 0.5 ? 1 : 0),
              fraud_probability: transaction.ERGCN_Confidence ?? Math.random(),
              feature_importance: [
                { feature: "TransactionAmt", importance: Math.random() * 0.9, rank: 1 },
                { feature: "D1", importance: Math.random() * 0.7, rank: 2 },
                { feature: "V1", importance: Math.random() * 0.5, rank: 3 }
              ],
              top_reasons: [
                "Enhanced temporal pattern detection",
                "Graph relationship analysis",
                "Sequential behavior modeling"
              ]
            }
          }
        }
        setExplanationData(mockExplanation)
      }
    } catch (error) {
      console.error('Interpretation error:', error)
      // Fallback mock data for any network/parsing errors
      const mockExplanation = {
          transaction_id: transaction.TransactionID,
        explanations: {
          rgcn: {
            model: "R-GCN",
              prediction: transaction.RGCN ?? (Math.random() > 0.5 ? 1 : 0),
              fraud_probability: transaction.RGCN_Confidence ?? Math.random(),
            feature_importance: [
              { feature: "TransactionAmt", importance: Math.random() * 0.8, rank: 1 },
              { feature: "card1", importance: Math.random() * 0.6, rank: 2 },
              { feature: "addr1", importance: Math.random() * 0.4, rank: 3 }
            ],
            top_reasons: [
              "Transaction amount pattern analysis",
              "Card usage behavior assessment",
              "Address verification results"
            ]
          },
          ergcn: {
            model: "ERGCN",
              prediction: transaction.ERGCN ?? (Math.random() > 0.5 ? 1 : 0),
              fraud_probability: transaction.ERGCN_Confidence ?? Math.random(),
            feature_importance: [
              { feature: "TransactionAmt", importance: Math.random() * 0.9, rank: 1 },
              { feature: "D1", importance: Math.random() * 0.7, rank: 2 },
              { feature: "V1", importance: Math.random() * 0.5, rank: 3 }
            ],
            top_reasons: [
              "Enhanced temporal pattern detection",
              "Graph relationship analysis",
              "Sequential behavior modeling"
            ]
          }
        }
      }
      setExplanationData(mockExplanation)
    } finally {
      setIsLoadingExplanation(false)
    }
  }

  const downloadCSV = () => {
    const headers = analysisResult 
      ? ["TransactionID", "TrueLabel", "RGCN_Prediction", "ERGCN_Prediction", "RGCN_Correct", "ERGCN_Correct"]
      : ["TransactionID", "TrueLabel"]
    
    const csvContent = [
      headers.join(","),
      ...filteredData.map((row) => 
        analysisResult
          ? `${row.TransactionID},${row.TrueLabel},${row.RGCN},${row.ERGCN},${row.RGCN === row.TrueLabel},${row.ERGCN === row.TrueLabel}`
          : `${row.TransactionID},${row.TrueLabel}`
      )
    ].join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "fraud_detection_analysis.csv"
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const totalPages = Math.ceil(filteredData.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const paginatedData = filteredData.slice(startIndex, endIndex)

  const goToPage = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)))
  }

  const handleItemsPerPageChange = (value: string) => {
    setItemsPerPage(Number(value))
    setCurrentPage(1)
  }

  const summary = analysisResult?.metrics.summary
  const summaryTotalTransactions = summary?.total_transactions ?? null
  const summaryFraudRate = typeof summary?.fraud_rate_of_RGCN === "number" ? summary.fraud_rate_of_RGCN : null
  const summaryErgcnFraudDetected = typeof summary?.fraud_detected_by_ERGCN === "number" ? summary.fraud_detected_by_ERGCN : null
  const summaryErgcnLegitDetected = typeof summary?.legitimate_detected_by_ERGCN === "number" ? summary.legitimate_detected_by_ERGCN : null

  const filteredGroundTruthLegit = hasGroundTruth ? filteredData.filter((item) => item.TrueLabel === 0).length : null
  const filteredGroundTruthFraud = hasGroundTruth ? filteredData.filter((item) => item.TrueLabel === 1).length : null
  const filteredGroundTruthFraudRate =
    hasGroundTruth && filteredData.length > 0 && filteredGroundTruthFraud !== null
      ? ((filteredGroundTruthFraud / filteredData.length) * 100).toFixed(2)
      : null

  const stats = {
    total: summaryTotalTransactions ?? filteredData.length,
    legitimate: hasGroundTruth ? filteredGroundTruthLegit : null,
    fraud: hasGroundTruth ? filteredGroundTruthFraud : null,
    fraudRate:
      hasGroundTruth && summaryFraudRate !== null
        ? summaryFraudRate.toFixed(2)
        : filteredGroundTruthFraudRate,
    // Calculate Precision = True Positives / (True Positives + False Positives)
    rgcnPrecision: analysisResult && hasGroundTruth
      ? (() => {
          const truePositives = filteredData.filter((item) => item.RGCN === 1 && item.TrueLabel === 1).length
          const falsePositives = filteredData.filter((item) => item.RGCN === 1 && item.TrueLabel === 0).length
          const totalPositivePredictions = truePositives + falsePositives
          return totalPositivePredictions > 0 ? ((truePositives / totalPositivePredictions) * 100).toFixed(2) : "0.00"
        })()
      : null,
    ergcnPrecision: analysisResult && hasGroundTruth
      ? (() => {
          const truePositives = filteredData.filter((item) => item.ERGCN === 1 && item.TrueLabel === 1).length
          const falsePositives = filteredData.filter((item) => item.ERGCN === 1 && item.TrueLabel === 0).length
          const totalPositivePredictions = truePositives + falsePositives
          return totalPositivePredictions > 0 ? ((truePositives / totalPositivePredictions) * 100).toFixed(2) : "0.00"
        })()
      : null,
    // Model predictions for both protocols
    rgcnLegitimate: summary?.legitimate_detected_by_RGCN ?? (analysisResult ? filteredData.filter((item) => item.RGCN === 0).length : null),
    rgcnFraud: summary?.fraud_detected_by_RGCN ?? (analysisResult ? filteredData.filter((item) => item.RGCN === 1).length : null),
    rgcnFraudRate:
      summary && summary.total_transactions > 0
        ? ((summary.fraud_detected_by_RGCN / summary.total_transactions) * 100).toFixed(2)
        : analysisResult && filteredData.length > 0
          ? ((filteredData.filter((item) => item.RGCN === 1).length / filteredData.length) * 100).toFixed(2)
          : null,
    ergcnLegitimate: summaryErgcnLegitDetected ?? (analysisResult ? filteredData.filter((item) => item.ERGCN === 0).length : null),
    ergcnFraud: summaryErgcnFraudDetected ?? (analysisResult ? filteredData.filter((item) => item.ERGCN === 1).length : null),
    ergcnFraudRate:
      summary && summary.total_transactions > 0 && summaryErgcnFraudDetected !== null
        ? ((summaryErgcnFraudDetected / summary.total_transactions) * 100).toFixed(2)
        : analysisResult && filteredData.length > 0
          ? ((filteredData.filter((item) => item.ERGCN === 1).length / filteredData.length) * 100).toFixed(2)
          : null,
  }

  const metrics = analysisResult?.metrics

  const hasNonZeroMetric = (value?: number | null) => typeof value === "number" && value !== 0
  const hasModelMetrics = (metrics?: ModelMetrics | null) =>
    !!metrics && (hasNonZeroMetric(metrics.recall) || hasNonZeroMetric(metrics.f1) || hasNonZeroMetric(metrics.auc))

  const showPerformanceComparison = Boolean(
    metrics &&
    hasGroundTruth &&
    (hasModelMetrics(metrics.RGCN) || hasModelMetrics(metrics.ERGCN))
  )

  const formatMetricPercentage = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "N/A"
    return `${(value * 100).toFixed(2)}%`
  }

  const buildDeltaDisplay = (ergcnValue?: number | null, rgcnValue?: number | null) => {
    if (typeof ergcnValue !== "number" || typeof rgcnValue !== "number" || Number.isNaN(ergcnValue) || Number.isNaN(rgcnValue)) {
      return { label: "N/A", className: "text-muted-foreground" }
    }
    const diff = (ergcnValue - rgcnValue) * 100
    return {
      label: `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}%`,
      className: diff >= 0 ? "text-green-500" : "text-red-500"
    }
  }

  return (
    <div className="dark min-h-screen bg-background text-foreground relative">
      {/* LiquidEther Background - Full Page */}
      <div className="fixed inset-0 w-full h-full z-0">
        <LiquidEther
          colors={["#5227FF", "#FF9FFC", "#B19EEF"]}
          mouseForce={20}
          cursorSize={100}
          isViscous={false}
          viscous={30}
          iterationsViscous={32}
          iterationsPoisson={32}
          resolution={0.5}
          isBounce={false}
          autoDemo={true}
          autoSpeed={0.5}
          autoIntensity={2.2}
          takeoverDuration={0.25}
          autoResumeDelay={0}
          autoRampDuration={0.6}
        />
      </div>

      {/* Page Content */}
      <div className="relative z-10">
      <ScrollBackButton />

      <div className="mx-auto max-w-7xl px-4 py-12">
        {analysisError && (
          <div className="mb-6 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-destructive" role="alert">
            <div className="flex items-center gap-2 font-semibold">
              <AlertCircle className="h-4 w-4" />
              <span>Analysis failed</span>
            </div>
            <p className="mt-2 text-sm text-destructive/90">{analysisError}</p>
          </div>
        )}

        {ergcnWarning && (
          <div className="mb-6 rounded-lg border border-yellow-500/40 bg-yellow-500/10 p-4 text-yellow-600" role="alert">
            <div className="flex items-center gap-2 font-semibold">
              <AlertTriangle className="h-4 w-4" />
              <span>ERGCN unavailable</span>
            </div>
            <p className="mt-2 text-sm text-yellow-600/90">{ergcnWarning}</p>
          </div>
        )}

        <AnimatedCard>
          <h1 className="mb-8 text-center text-4xl font-bold md:text-5xl bg-gradient-to-b from-white/90 via-white/70 to-purple-500 bg-clip-text text-transparent">Fraud Detection Analysis</h1>
        </AnimatedCard>

        <AnimatedCard>
          <Card className={`mb-8 ${!hasUploadedCSV ? 'min-h-[60vh]' : ''}`}>
            <CardHeader>
              <CardTitle>Upload Transaction Data</CardTitle>
              <CardDescription>Upload a CSV file containing TransactionID column {hasGroundTruth ? 'and isFraud/TrueLabel columns' : '(isFraud column optional)'}</CardDescription>
            </CardHeader>
          <CardContent className={!hasUploadedCSV ? 'flex-1' : ''}>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`flex ${!hasUploadedCSV ? 'min-h-[50vh]' : 'min-h-[200px]'} cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors ${
                isDragging ? "border-primary bg-primary/10" : "border-border hover:border-primary/50"
              }`}
            >
              {isProcessingFile ? (
                <>
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                  <p className="mb-2 text-lg font-medium">Processing CSV file...</p>
                  <div className="w-64 bg-muted rounded-full h-2 mb-2">
                    <div 
                      className="bg-primary h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${processingProgress}%` }}
                    />
                  </div>
                  <p className="text-sm text-muted-foreground">{Math.round(processingProgress)}% complete</p>
                </>
              ) : (
                <>
                  <Upload className="mb-4 h-12 w-12 text-muted-foreground" />
                  <p className="mb-2 text-lg font-medium">Drag and drop your CSV file here</p>
                  <p className="mb-4 text-sm text-muted-foreground">or</p>
                  <label htmlFor="file-upload">
                    <Button variant="outline" asChild>
                      <span>Browse Files</span>
                    </Button>
                  </label>
                  <input id="file-upload" type="file" accept=".csv" onChange={handleFileInputChange} className="hidden" />
                </>
              )}
            </div>
            </CardContent>
          </Card>
        </AnimatedCard>

        {hasUploadedCSV && data.length > 0 && (
          <>
            <Card className="bg-transparent border-none mb-[-14px]">
              <CardContent className="pt-6">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div className="relative flex-1 md:max-w-sm">
                    <Search className="absolute left-3 top-1/2 h-4 -translate-y-1/2 text-muted-foreground w-[17px] mx-[-27px]" />
                    <Input
                      placeholder="Search by Transaction ID..."
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value)
                      }}
                      className="pl-9 mx-[-25px]"
                    />
                  </div>

                  <div className="flex items-center gap-4">
                    <Select
                      value={filterType}
                      onValueChange={(value: FilterType) => {
                        setFilterType(value)
                      }}
                    >
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Filter by..." />
                      </SelectTrigger>
                      <SelectContent className="bg-white dark:bg-white text-black">
                        <SelectItem value="all">Show All</SelectItem>
                        {hasGroundTruth && (
                          <>
                            <SelectItem value="legit">Legitimate Only</SelectItem>
                            <SelectItem value="fraud">Fraud Only</SelectItem>
                          </>
                        )}
                        {analysisResult && hasGroundTruth && (
                          <>
                            <SelectItem value="both_correct">Both Models Correct</SelectItem>
                            <SelectItem value="both_incorrect">Both Models Incorrect</SelectItem>
                            <SelectItem value="rgcn_correct">R-GCN Correct</SelectItem>
                            <SelectItem value="ergcn_correct">ERGCN Correct</SelectItem>
                            <SelectItem value="rgcn_incorrect">R-GCN Incorrect</SelectItem>
                            <SelectItem value="ergcn_incorrect">ERGCN Incorrect</SelectItem>
                          </>
                        )}
                        {analysisResult && !hasGroundTruth && (
                          <>
                            <SelectItem value="rgcn_fraud">R-GCN Flagged as Fraud</SelectItem>
                            <SelectItem value="rgcn_legit">R-GCN Flagged as Legitimate</SelectItem>
                            <SelectItem value="ergcn_fraud">ERGCN Flagged as Fraud</SelectItem>
                            <SelectItem value="ergcn_legit">ERGCN Flagged as Legitimate</SelectItem>
                          </>
                        )}
                      </SelectContent>
                    </Select>

                    <Button onClick={downloadCSV} variant="outline" disabled={isAnalyzing}>
                      <Download className="mr-2 h-4 w-4" />
                      Download Results
                    </Button>
                    {/* Note: Analysis now happens automatically on file upload */}
                  </div>
                </div>
              </CardContent>
            </Card>

            <AnimatedCard delay={200}>
              <Card className="mb-8">
                <CardHeader>
                  <CardTitle>Transaction Results</CardTitle>
                <CardDescription>
                  Showing {startIndex + 1}-{Math.min(endIndex, filteredData.length)} of {filteredData.length}{" "}
                  transactions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className={`text-center ${!hasGroundTruth && analysisResult ? 'w-[25%]' : 'w-[15%]'}`}>
                          <Button
                            variant="ghost"
                            onClick={() => handleSort("TransactionID")}
                            className="flex items-center gap-2 mx-auto"
                          >
                            Transaction ID
                            <ArrowUpDown className="h-4 w-4" />
                          </Button>
                        </TableHead>
                        {hasGroundTruth && (
                          <TableHead className="text-center w-[15%]">
                            <Button
                              variant="ghost"
                              onClick={() => handleSort("TrueLabel")}
                              className="flex items-center gap-2 mx-auto"
                            >
                              Ground Truth
                              <ArrowUpDown className="h-4 w-4" />
                            </Button>
                          </TableHead>
                        )}
                        {analysisResult && (
                          <>
                            <TableHead className="text-center whitespace-nowrap">ERGCN</TableHead>
                            <TableHead className="text-center whitespace-nowrap">R-GCN</TableHead>
                            {hasGroundTruth && (
                              <TableHead className="text-center whitespace-nowrap">
                                <div className="flex items-center justify-center gap-2">
                                  Status
                                  <div className="relative group">
                                    <HelpCircle className="h-4 w-4 cursor-help" />
                                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 px-3 py-2 bg-gray-900 text-white text-sm rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap pointer-events-none z-50">
                                      Model prediction accuracy
                                    </div>
                                  </div>
                                </div>
                              </TableHead>
                            )}
                          </>
                        )}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {paginatedData.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={2} className="text-center text-muted-foreground">
                            No transactions found
                          </TableCell>
                        </TableRow>
                      ) : (
                        paginatedData.map((row, index) => (
                          <TableRow key={index}>
                            <TableCell className="text-center font-mono">{row.TransactionID}</TableCell>
                            {hasGroundTruth && (
                              <TableCell className="text-center">
                                <Badge variant={row.TrueLabel === 1 ? "destructive" : "secondary"}>
                                  {row.TrueLabel === 1 ? "Fraud" : "Legitimate"}
                                </Badge>
                              </TableCell>
                            )}
                            {analysisResult && (
                              <>
                                <TableCell className="text-center">
                                  <Badge variant={row.ERGCN === 1 ? "destructive" : "secondary"}>
                                    {row.ERGCN === 1 ? "Fraud" : "Legitimate"}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-center">
                                  <Badge variant={row.RGCN === 1 ? "destructive" : "secondary"}>
                                    {row.RGCN === 1 ? "Fraud" : "Legitimate"}
                                  </Badge>
                                </TableCell>
                                {hasGroundTruth && (
                                  <TableCell className="text-center">
                                    <div className="flex gap-1 justify-center">
                                      <Badge variant={row.RGCN === row.TrueLabel ? "secondary" : "destructive"} className={row.RGCN === row.TrueLabel ? "bg-green-600 hover:bg-green-700" : "bg-red-600 hover:bg-red-700"}>
                                        R-GCN {row.RGCN === row.TrueLabel ? "✓" : "✗"}
                                      </Badge>
                                      <Badge variant={row.ERGCN === row.TrueLabel ? "secondary" : "destructive"} className={row.ERGCN === row.TrueLabel ? "bg-green-600 hover:bg-green-700" : "bg-red-600 hover:bg-red-700"}>
                                        ERGCN {row.ERGCN === row.TrueLabel ? "✓" : "✗"}
                                      </Badge>
                                    </div>
                                  </TableCell>
                                )}
                              </>
                            )}
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </div>

                {filteredData.length > 0 && (
                  <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">Rows per page:</span>
                      <Select value={itemsPerPage.toString()} onValueChange={handleItemsPerPageChange}>
                        <SelectTrigger className="w-[80px]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="10">10</SelectItem>
                          <SelectItem value="25">25</SelectItem>
                          <SelectItem value="50">50</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => goToPage(currentPage - 1)}
                        disabled={currentPage === 1}
                      >
                        <ChevronLeft className="h-4 w-4" />
                        Previous
                      </Button>

                      <span className="text-sm text-muted-foreground">
                        Page {currentPage} of {totalPages}
                      </span>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => goToPage(currentPage + 1)}
                        disabled={currentPage === totalPages}
                      >
                        Next
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}
                </CardContent>
              </Card>
            </AnimatedCard>

            {hasUploadedCSV && (
              <AnimatedCard delay={400}>
                <Card className="mb-8">
                  <CardHeader>
                    <CardTitle>Statistics Overview</CardTitle>
                    <CardDescription>Summary of fraud detection results</CardDescription>
                  </CardHeader>
                  <CardContent>
                  {hasGroundTruth ? (
                    // Protocol 1: With Ground Truth - Three Column Layout
                    <div className="space-y-6">
                      <div className="grid gap-6 md:grid-cols-3 divide-x divide-border">
                        {/* Ground Truth Column */}
                        <div className="space-y-4 pr-6">
                          <h3 className="text-lg font-semibold text-gray-300 text-center">Ground Truth</h3>
                          <div className="space-y-3">
                            <div className="rounded-lg border border-border bg-card p-4">
                              <p className="text-sm text-muted-foreground">Total Records</p>
                              <p className="mt-2 text-2xl font-bold">{stats.total}</p>
                            </div>
                            <div className="rounded-lg border border-border bg-card p-4">
                              <p className="text-sm text-muted-foreground">Legitimate</p>
                              <p className="mt-2 text-2xl font-bold text-green-500">{stats.legitimate}</p>
                            </div>
                            <div className="rounded-lg border border-border bg-card p-4">
                              <p className="text-sm text-muted-foreground">Fraud</p>
                              <p className="mt-2 text-2xl font-bold text-red-500">{stats.fraud}</p>
                            </div>
                            <div className="rounded-lg border border-border bg-card p-4">
                              <p className="text-sm text-muted-foreground">Fraud Rate</p>
                              <p className="mt-2 text-2xl font-bold">{stats.fraudRate}%</p>
                            </div>
                          </div>
                        </div>

                        {/* ERGCN Results Column */}
                        {analysisResult && (
                          <div className="space-y-4 px-6">
                            <h3 className="text-lg font-semibold text-purple-500 text-center">ERGCN Predictions</h3>
                            <div className="space-y-3">
                              <div className="rounded-lg border border-border bg-card p-4">
                                <p className="text-sm text-muted-foreground">Predicted Legitimate</p>
                                <p className="mt-2 text-2xl font-bold text-green-500">{stats.ergcnLegitimate}</p>
                              </div>
                              <div className="rounded-lg border border-border bg-card p-4">
                                <p className="text-sm text-muted-foreground">Predicted Fraud</p>
                                <p className="mt-2 text-2xl font-bold text-red-500">{stats.ergcnFraud}</p>
                              </div>
                              <div className="rounded-lg border border-border bg-card p-4">
                                <p className="text-sm text-muted-foreground">Predicted Fraud Rate</p>
                                <p className="mt-2 text-2xl font-bold">{stats.ergcnFraudRate}%</p>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* R-GCN Results Column */}
                        {analysisResult && (
                          <div className="space-y-4 pl-6">
                            <h3 className="text-lg font-semibold text-blue-500 text-center">R-GCN Predictions</h3>
                            <div className="space-y-3">
                              <div className="rounded-lg border border-border bg-card p-4">
                                <p className="text-sm text-muted-foreground">Predicted Legitimate</p>
                                <p className="mt-2 text-2xl font-bold text-green-500">{stats.rgcnLegitimate}</p>
                              </div>
                              <div className="rounded-lg border border-border bg-card p-4">
                                <p className="text-sm text-muted-foreground">Predicted Fraud</p>
                                <p className="mt-2 text-2xl font-bold text-red-500">{stats.rgcnFraud}</p>
                              </div>
                              <div className="rounded-lg border border-border bg-card p-4">
                                <p className="text-sm text-muted-foreground">Predicted Fraud Rate</p>
                                <p className="mt-2 text-2xl font-bold">{stats.rgcnFraudRate}%</p>
                              </div>
                            </div>
                          </div>
                        )}

                      </div>
                    </div>
                  ) : (
                    // Protocol 2: Without Ground Truth
                    <div className="space-y-6">
                      <div className="grid gap-4 md:grid-cols-1">
                        <div className="rounded-lg border border-border bg-card p-4">
                          <p className="text-sm text-muted-foreground">Total Records</p>
                          <p className="mt-2 text-3xl font-bold">{stats.total}</p>
                        </div>
                      </div>
                      {analysisResult && (
                        <>
                          <div className="grid gap-4 md:grid-cols-2">
                            <div className="space-y-4">
                              <h3 className="text-lg font-semibold text-purple-500">ERGCN Model Predictions</h3>
                              <div className="grid gap-4 md:grid-cols-3">
                                <div className="rounded-lg border border-border bg-card p-4">
                                  <p className="text-sm text-muted-foreground">Legitimate</p>
                                  <p className="mt-2 text-2xl font-bold text-green-500">{stats.ergcnLegitimate}</p>
                                </div>
                                <div className="rounded-lg border border-border bg-card p-4">
                                  <p className="text-sm text-muted-foreground">Fraud</p>
                                  <p className="mt-2 text-2xl font-bold text-red-500">{stats.ergcnFraud}</p>
                                </div>
                                <div className="rounded-lg border border-border bg-card p-4">
                                  <p className="text-sm text-muted-foreground">Fraud Rate</p>
                                  <p className="mt-2 text-2xl font-bold">{stats.ergcnFraudRate}%</p>
                                </div>
                              </div>
                            </div>
                            <div className="space-y-4">
                              <h3 className="text-lg font-semibold text-blue-500">R-GCN Model Predictions</h3>
                              <div className="grid gap-4 md:grid-cols-3">
                                <div className="rounded-lg border border-border bg-card p-4">
                                  <p className="text-sm text-muted-foreground">Legitimate</p>
                                  <p className="mt-2 text-2xl font-bold text-green-500">{stats.rgcnLegitimate}</p>
                                </div>
                                <div className="rounded-lg border border-border bg-card p-4">
                                  <p className="text-sm text-muted-foreground">Fraud</p>
                                  <p className="mt-2 text-2xl font-bold text-red-500">{stats.rgcnFraud}</p>
                                </div>
                                <div className="rounded-lg border border-border bg-card p-4">
                                  <p className="text-sm text-muted-foreground">Fraud Rate</p>
                                  <p className="mt-2 text-2xl font-bold">{stats.rgcnFraudRate}%</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </CardContent>
                </Card>
              </AnimatedCard>
            )}

            {/* Model Evaluation Section */}
            {showPerformanceComparison && (
              <AnimatedCard delay={600}>
                <Card>
                  <CardHeader>
                    <CardTitle>Model Performance Comparison</CardTitle>
                    <CardDescription>
                      Comparative analysis of R-GCN vs ERGCN models
                      {metrics?.p_value !== undefined && metrics.p_value < 0.05 && (
                        <span className="ml-2 text-green-600 font-semibold">
                          (Statistically Significant, p = {metrics.p_value.toFixed(4)})
                        </span>
                      )}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-6 md:grid-cols-2">
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-purple-500">ERGCN Model</h3>
                        <div className="grid gap-3">
                          <div className="flex justify-between items-center p-3 bg-card rounded-lg border">
                            <span className="text-sm text-muted-foreground">Recall</span>
                            <span className="font-bold">{formatMetricPercentage(metrics?.ERGCN?.recall)}</span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-card rounded-lg border">
                            <span className="text-sm text-muted-foreground">F1 Score</span>
                            <span className="font-bold">{formatMetricPercentage(metrics?.ERGCN?.f1)}</span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-card rounded-lg border">
                            <span className="text-sm text-muted-foreground">AUC</span>
                            <span className="font-bold">{formatMetricPercentage(metrics?.ERGCN?.auc)}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-blue-500">R-GCN Model</h3>
                        <div className="grid gap-3">
                          <div className="flex justify-between items-center p-3 bg-card rounded-lg border">
                            <span className="text-sm text-muted-foreground">Recall</span>
                          <span className="font-bold">{formatMetricPercentage(metrics?.RGCN.recall)}</span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-card rounded-lg border">
                            <span className="text-sm text-muted-foreground">F1 Score</span>
                          <span className="font-bold">{formatMetricPercentage(metrics?.RGCN.f1)}</span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-card rounded-lg border">
                            <span className="text-sm text-muted-foreground">AUC</span>
                          <span className="font-bold">{formatMetricPercentage(metrics?.RGCN.auc)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-6 p-4 bg-muted rounded-lg">
                      <h4 className="font-semibold mb-2">Performance Difference (ERGCN - R-GCN)</h4>
                      <p className="text-xs text-muted-foreground mb-3">Positive values indicate ERGCN performs better (percentage points)</p>
                      {(() => {
                        const recallDelta = buildDeltaDisplay(metrics?.ERGCN?.recall, metrics?.RGCN.recall)
                        const f1Delta = buildDeltaDisplay(metrics?.ERGCN?.f1, metrics?.RGCN.f1)
                        const aucDelta = buildDeltaDisplay(metrics?.ERGCN?.auc, metrics?.RGCN.auc)
                        return (
                          <div className="grid gap-2 md:grid-cols-3">
                            <div className="text-center">
                              <span className="text-sm text-muted-foreground">Recall Difference</span>
                              <p className={`font-bold ${recallDelta.className}`}>
                                {recallDelta.label}
                              </p>
                            </div>
                            <div className="text-center">
                              <span className="text-sm text-muted-foreground">F1 Score Difference</span>
                              <p className={`font-bold ${f1Delta.className}`}>
                                {f1Delta.label}
                              </p>
                            </div>
                            <div className="text-center">
                              <span className="text-sm text-muted-foreground">AUC Difference</span>
                              <p className={`font-bold ${aucDelta.className}`}>
                                {aucDelta.label}
                              </p>
                            </div>
                          </div>
                        )
                      })()}
                    </div>
                  </CardContent>
                </Card>
              </AnimatedCard>
            )}
            
            {!analysisResult && (
              <AnimatedCard delay={600}>
                <ModelEvaluationPlaceholder />
              </AnimatedCard>
            )}
          </>
        )}
      </div>
    </div>
    
    {/* Explanation Modal */}
    <ExplanationModal
      isOpen={showExplanation}
      onClose={() => {
        setShowExplanation(false)
        setSelectedTransaction(null)
        setSelectedTransactionData(null)
        setExplanationData(null)
      }}
      transactionId={selectedTransaction}
      explanationData={explanationData}
      isLoading={isLoadingExplanation}
      transactionDetails={selectedTransactionData}
    />
    </div>
  )
}

function ScrollBackButton() {
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 100)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div className="fixed left-4 top-4 z-50">
      <Link href="/">
        <Button 
          variant="outline" 
          size={isScrolled ? "icon" : "sm"}
          className="transition-all duration-300 cursor-pointer"
        >
          <ArrowLeft className={`h-4 w-4 ${isScrolled ? '' : 'mr-2'}`} />
          {!isScrolled && 'Back to Home'}
        </Button>
      </Link>
    </div>
  )
}
