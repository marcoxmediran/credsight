import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface ModelEvaluationProps {
  f1Score?: number
  recall?: number
  aucRoc?: number
  confusionMatrix?: number[][]
  classificationReport?: {
    precision: number
    recall: number
    f1Score: number
    support: number
  }[]
  trainingLossData?: number[]
  rocCurveData?: { fpr: number[]; tpr: number[] }
  testMetricsData?: {
    epochs: number[]
    f1: number[]
    recall: number[]
    auc: number[]
  }
}

export default function ModelEvaluationPlaceholder({
  f1Score = 0.0702,
  recall = 0.8197,
  aucRoc = 0.6360,
  confusionMatrix = [[16150, 23893], [200, 909]],
}: ModelEvaluationProps) {
  const classificationData = [
    { label: 'Legitimate', precision: 0.99, recall: 0.40, f1Score: 0.57, support: 40043 },
    { label: 'Fraud', precision: 0.04, recall: 0.82, f1Score: 0.07, support: 1109 },
    { label: 'Accuracy', precision: null, recall: null, f1Score: 0.41, support: 41152 },
    { label: 'Macro Avg', precision: 0.51, recall: 0.61, f1Score: 0.32, support: 41152 },
    { label: 'Weighted Avg', precision: 0.96, recall: 0.41, f1Score: 0.56, support: 41152 }
  ]

  return (
    <div className="mt-12 space-y-8">
      {/* Final Evaluation */}
      <Card>
        <CardHeader>
          <CardTitle>Final Evaluation on Test Set</CardTitle>
          <CardDescription>Model performance metrics on unseen test data</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{f1Score.toFixed(3)}</div>
                <p className="text-sm text-muted-foreground">F1 Score</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{recall.toFixed(3)}</div>
                <p className="text-sm text-muted-foreground">Recall</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{aucRoc.toFixed(3)}</div>
                <p className="text-sm text-muted-foreground">AUC-ROC</p>
              </CardContent>
            </Card>
          </div>

          {/* Confusion Matrix & Classification Report */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Confusion Matrix</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3 w-full">
                  {confusionMatrix.map((row, i) =>
                    row.map((value, j) => (
                      <div
                        key={`${i}-${j}`}
                        className={`p-8 text-center font-mono text-2xl rounded ${
                          i === j ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {value.toLocaleString()}
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Classification Report</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 font-medium">Class</th>
                        <th className="text-right py-2 font-medium">Precision</th>
                        <th className="text-right py-2 font-medium">Recall</th>
                        <th className="text-right py-2 font-medium">F1-Score</th>
                        <th className="text-right py-2 font-medium">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {classificationData.map((row, i) => (
                        <tr key={i} className={`${i >= 2 ? 'border-t' : ''}`}>
                          <td className={`py-2 ${i >= 2 ? 'font-medium' : ''}`}>{row.label}</td>
                          <td className="text-right py-2">
                            {row.precision !== null ? row.precision.toFixed(2) : '-'}
                          </td>
                          <td className="text-right py-2">
                            {row.recall !== null ? row.recall.toFixed(2) : '-'}
                          </td>
                          <td className="text-right py-2">{row.f1Score.toFixed(2)}</td>
                          <td className="text-right py-2">{row.support.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card>
        <CardHeader>
          <CardTitle>Model Visualizations</CardTitle>
          <CardDescription>Training progress and model performance charts</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Training Loss Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-48 bg-muted rounded flex items-center justify-center">
                  <p className="text-muted-foreground">Loading line chart...</p>
                </div>
              </CardContent>
            </Card>

            {/* Confusion Matrix Heatmap */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Confusion Matrix Heatmap</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-48 bg-muted rounded flex items-center justify-center">
                  <p className="text-muted-foreground">Loading heatmap...</p>
                </div>
              </CardContent>
            </Card>

            {/* Test Metrics Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Test Metrics over Training</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-48 bg-muted rounded flex items-center justify-center">
                  <p className="text-muted-foreground">Loading metrics chart...</p>
                </div>
              </CardContent>
            </Card>

            {/* ROC Curve */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">ROC Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-48 bg-muted rounded flex items-center justify-center">
                  <p className="text-muted-foreground">Loading ROC curve...</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}