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
import { ArrowLeft, Upload, Download, ArrowUpDown, Search, ChevronLeft, ChevronRight } from "lucide-react"
import Link from "next/link"
import LiquidEther from "@/components/LiquidEther"
import ModelEvaluationPlaceholder from "@/components/ModelEvaluationPlaceholder"

type TransactionData = {
  TransactionID: number
  isFraud: boolean
}

type FilterType = "all" | "legit" | "fraud"

// Example transaction data (for demonstration purposes)
const EXAMPLE_DATA: TransactionData[] = [
  { TransactionID: 1001, isFraud: false },
  { TransactionID: 1002, isFraud: false },
  { TransactionID: 1003, isFraud: true },
  { TransactionID: 1004, isFraud: false },
  { TransactionID: 1005, isFraud: false },
  { TransactionID: 1006, isFraud: false },
  { TransactionID: 1007, isFraud: true },
  { TransactionID: 1008, isFraud: false },
  { TransactionID: 1009, isFraud: false },
  { TransactionID: 1010, isFraud: false },
  { TransactionID: 1011, isFraud: true },
  { TransactionID: 1012, isFraud: false },
  { TransactionID: 1013, isFraud: false },
  { TransactionID: 1014, isFraud: false },
  { TransactionID: 1015, isFraud: true },
  { TransactionID: 1016, isFraud: false },
  { TransactionID: 1017, isFraud: false },
  { TransactionID: 1018, isFraud: false },
  { TransactionID: 1019, isFraud: true },
  { TransactionID: 1020, isFraud: false },
  { TransactionID: 1021, isFraud: false },
  { TransactionID: 1022, isFraud: false },
  { TransactionID: 1023, isFraud: false },
  { TransactionID: 1024, isFraud: true },
  { TransactionID: 1025, isFraud: false },
  { TransactionID: 1026, isFraud: false },
  { TransactionID: 1027, isFraud: false },
  { TransactionID: 1028, isFraud: true },
  { TransactionID: 1029, isFraud: false },
  { TransactionID: 1030, isFraud: false },
]

export default function AnalyzePage() {
  const [data, setData] = useState<TransactionData[]>(EXAMPLE_DATA)
  const [filteredData, setFilteredData] = useState<TransactionData[]>(EXAMPLE_DATA)
  const [searchQuery, setSearchQuery] = useState("")
  const [filterType, setFilterType] = useState<FilterType>("all")
  const [sortConfig, setSortConfig] = useState<{
    key: keyof TransactionData
    direction: "asc" | "desc"
  } | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(10)

  const handleFileUpload = useCallback((file: File) => {
    if (file.type !== "text/csv") {
      alert("Please upload a CSV file")
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target?.result as string
      const rows = text.split("\n").filter((row) => row.trim())
      const headers = rows[0].split(",").map((h) => h.trim())

      const transactionIdIndex = headers.findIndex((h) => h.toLowerCase().includes("transactionid"))
      const isFraudIndex = headers.findIndex((h) => h.toLowerCase().includes("isfraud"))

      if (transactionIdIndex === -1 || isFraudIndex === -1) {
        alert("CSV must contain TransactionID and isFraud columns")
        return
      }

      const parsedData: TransactionData[] = rows.slice(1).map((row) => {
        const values = row.split(",").map((v) => v.trim())
        return {
          TransactionID: Number.parseInt(values[transactionIdIndex]),
          isFraud: values[isFraudIndex].toLowerCase() === "true" || values[isFraudIndex] === "1",
        }
      })

      setData(parsedData)
      setFilteredData(parsedData)
    }
    reader.readAsText(file)
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

    if (filterType === "legit") {
      result = result.filter((item) => !item.isFraud)
    } else if (filterType === "fraud") {
      result = result.filter((item) => item.isFraud)
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

        if (typeof aValue === "boolean" && typeof bValue === "boolean") {
          return sortConfig.direction === "asc"
            ? aValue === bValue
              ? 0
              : aValue
                ? 1
                : -1
            : bValue === aValue
              ? 0
              : bValue
                ? 1
                : -1
        }

        return 0
      })
    }

    setFilteredData(result)
    setCurrentPage(1)
  }, [data, filterType, searchQuery, sortConfig])

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

  const downloadCSV = () => {
    const headers = ["TransactionID", "isFraud"]
    const csvContent = [headers.join(","), ...filteredData.map((row) => `${row.TransactionID},${row.isFraud}`)].join(
      "\n",
    )

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "fraud_detection_results.csv"
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

  const stats = {
    total: filteredData.length,
    legitimate: filteredData.filter((item) => !item.isFraud).length,
    fraud: filteredData.filter((item) => item.isFraud).length,
    fraudRate:
      filteredData.length > 0
        ? ((filteredData.filter((item) => item.isFraud).length / filteredData.length) * 100).toFixed(2)
        : "0.00",
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
        <AnimatedCard>
          <h1 className="mb-8 text-center text-4xl font-bold md:text-5xl bg-gradient-to-b from-white/90 via-white/70 to-purple-500 bg-clip-text text-transparent">Fraud Detection Analysis</h1>
        </AnimatedCard>

        <AnimatedCard>
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>Upload Transaction Data</CardTitle>
              <CardDescription>Upload a CSV file containing TransactionID and isFraud columns</CardDescription>
            </CardHeader>
          <CardContent>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors ${
                isDragging ? "border-primary bg-primary/10" : "border-border hover:border-primary/50"
              }`}
            >
              <Upload className="mb-4 h-12 w-12 text-muted-foreground" />
              <p className="mb-2 text-lg font-medium">Drag and drop your CSV file here</p>
              <p className="mb-4 text-sm text-muted-foreground">or</p>
              <label htmlFor="file-upload">
                <Button variant="outline" asChild>
                  <span>Browse Files</span>
                </Button>
              </label>
              <input id="file-upload" type="file" accept=".csv" onChange={handleFileInputChange} className="hidden" />
            </div>
            </CardContent>
          </Card>
        </AnimatedCard>

        {data.length > 0 && (
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
                      <SelectContent>
                        <SelectItem value="all">Show All</SelectItem>
                        <SelectItem value="legit">Legitimate Only</SelectItem>
                        <SelectItem value="fraud">Fraud Only</SelectItem>
                      </SelectContent>
                    </Select>

                    <Button onClick={downloadCSV} variant="outline">
                      <Download className="mr-2 h-4 w-4" />
                      Download CSV
                    </Button>
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
                        <TableHead>
                          <Button
                            variant="ghost"
                            onClick={() => handleSort("TransactionID")}
                            className="flex items-center gap-2"
                          >
                            Transaction ID
                            <ArrowUpDown className="h-4 w-4" />
                          </Button>
                        </TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            onClick={() => handleSort("isFraud")}
                            className="flex items-center gap-2"
                          >
                            Is Fraud
                            <ArrowUpDown className="h-4 w-4" />
                          </Button>
                        </TableHead>
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
                            <TableCell className="font-mono">{row.TransactionID}</TableCell>
                            <TableCell>
                              <Badge variant={row.isFraud ? "destructive" : "secondary"}>
                                {row.isFraud ? "True" : "False"}
                              </Badge>
                            </TableCell>
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
                          <SelectItem value="100">100</SelectItem>
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

            <AnimatedCard delay={400}>
              <Card>
                <CardHeader>
                  <CardTitle>Statistics Overview</CardTitle>
                  <CardDescription>Summary of fraud detection results</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="rounded-lg border border-border bg-card p-4">
                      <p className="text-sm text-muted-foreground">Total Records</p>
                      <p className="mt-2 text-3xl font-bold">{stats.total}</p>
                    </div>
                    <div className="rounded-lg border border-border bg-card p-4">
                      <p className="text-sm text-muted-foreground">Legitimate</p>
                      <p className="mt-2 text-3xl font-bold text-green-500">{stats.legitimate}</p>
                    </div>
                    <div className="rounded-lg border border-border bg-card p-4">
                      <p className="text-sm text-muted-foreground">Fraud</p>
                      <p className="mt-2 text-3xl font-bold text-red-500">{stats.fraud}</p>
                    </div>
                    <div className="rounded-lg border border-border bg-card p-4">
                      <p className="text-sm text-muted-foreground">Fraud Rate</p>
                      <p className="mt-2 text-3xl font-bold">{stats.fraudRate}%</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </AnimatedCard>

            {/* Model Evaluation Section */}
            <AnimatedCard delay={600}>
              <ModelEvaluationPlaceholder />
            </AnimatedCard>
          </>
        )}
      </div>
    </div>
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
