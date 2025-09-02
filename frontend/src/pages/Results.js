import React, { useState } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const Results = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');

  // Mock data for demonstration
  const mockData = {
    performanceMetrics: {
      accuracy: 95.8,
      sensitivity: 94.2,
      specificity: 97.1,
      totalPredictions: 1247,
    },
    classificationData: [
      { name: 'No DR', value: 652, color: '#66bb6a' },
      { name: 'Mild', value: 234, color: '#42a5f5' },
      { name: 'Moderate', value: 198, color: '#ffa726' },
      { name: 'Severe', value: 89, color: '#ff7043' },
      { name: 'Proliferative', value: 74, color: '#ef5350' }
    ],
    timeSeriesData: [
      { date: '2024-01-01', predictions: 45, accuracy: 94.2 },
      { date: '2024-01-02', predictions: 52, accuracy: 95.1 },
      { date: '2024-01-03', predictions: 38, accuracy: 96.3 },
      { date: '2024-01-04', predictions: 61, accuracy: 94.8 },
      { date: '2024-01-05', predictions: 47, accuracy: 97.2 },
      { date: '2024-01-06', predictions: 55, accuracy: 95.5 },
      { date: '2024-01-07', predictions: 43, accuracy: 96.1 }
    ],
    recentPredictions: [
      {
        id: '1',
        timestamp: '2024-01-07 14:30',
        classification: 'No DR',
        confidence: 97.3,
        riskLevel: 'Low',
        patientId: 'P-2024-001'
      },
      {
        id: '2',
        timestamp: '2024-01-07 14:15',
        classification: 'Moderate',
        confidence: 89.2,
        riskLevel: 'Medium',
        patientId: 'P-2024-002'
      },
      {
        id: '3',
        timestamp: '2024-01-07 13:45',
        classification: 'Proliferative DR',
        confidence: 94.7,
        riskLevel: 'High',
        patientId: 'P-2024-003'
      }
    ]
  };

  const getRiskBadgeClass = (riskLevel) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return 'risk-badge low';
      case 'medium': return 'risk-badge medium';
      case 'high': return 'risk-badge high';
      case 'critical': return 'risk-badge critical';
      default: return 'risk-badge';
    }
  };

  return (
    <div className="results-page">
      <div className="results-container">

        {/* Header */}
        <div className="results-header">
          <div className="header-content">
            <h1 className="page-title">
              📊 Analysis Results & Metrics
            </h1>
            <p className="page-description">
              Comprehensive analytics and performance metrics for diabetic retinopathy classification
            </p>
          </div>

          <div className="header-actions">
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="timeframe-select"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last 90 Days</option>
            </select>

            <button className="btn btn-secondary">
              📥 Export Report
            </button>
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className="metrics-overview">
          <div className="metric-card primary">
            <div className="metric-icon">✅</div>
            <div className="metric-content">
              <div className="metric-value">{mockData.performanceMetrics.accuracy}%</div>
              <div className="metric-label">Overall Accuracy</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon">👁️</div>
            <div className="metric-content">
              <div className="metric-value">{mockData.performanceMetrics.totalPredictions}</div>
              <div className="metric-label">Total Predictions</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon">📈</div>
            <div className="metric-content">
              <div className="metric-value">{mockData.performanceMetrics.sensitivity}%</div>
              <div className="metric-label">Sensitivity</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon">👥</div>
            <div className="metric-content">
              <div className="metric-value">{mockData.performanceMetrics.specificity}%</div>
              <div className="metric-label">Specificity</div>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="charts-section">

          {/* Performance Over Time */}
          <div className="chart-card">
            <div className="chart-header">
              <h3 className="chart-title">Performance Trends</h3>
            </div>
            <div className="chart-content">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={mockData.timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#757575"
                    fontSize={12}
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis stroke="#757575" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#0066cc" 
                    strokeWidth={3}
                    dot={{ fill: '#0066cc', r: 4 }}
                    activeDot={{ r: 6, fill: '#0066cc' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Classification Distribution */}
          <div className="chart-card">
            <div className="chart-header">
              <h3 className="chart-title">DR Classification Distribution</h3>
            </div>
            <div className="chart-content">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={mockData.classificationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {mockData.classificationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>

              {/* Legend */}
              <div className="chart-legend">
                {mockData.classificationData.map((item, index) => (
                  <div key={index} className="legend-item">
                    <div 
                      className="legend-color" 
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="legend-label">{item.name}</span>
                    <span className="legend-value">({item.value})</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Recent Predictions */}
        <div className="recent-predictions">
          <div className="section-header">
            <h3 className="section-title">Recent Predictions</h3>
            <div className="section-actions">
              <button className="btn btn-secondary btn-small">
                🔍 Filter
              </button>
            </div>
          </div>

          <div className="predictions-table">
            <div className="table-header">
              <div className="header-cell">Timestamp</div>
              <div className="header-cell">Patient ID</div>
              <div className="header-cell">Classification</div>
              <div className="header-cell">Confidence</div>
              <div className="header-cell">Risk Level</div>
              <div className="header-cell">Actions</div>
            </div>

            {mockData.recentPredictions.map((prediction) => (
              <div key={prediction.id} className="table-row">
                <div className="table-cell">
                  📅 <span>{prediction.timestamp}</span>
                </div>
                <div className="table-cell">
                  <span className="patient-id">{prediction.patientId}</span>
                </div>
                <div className="table-cell">
                  <span className="classification">{prediction.classification}</span>
                </div>
                <div className="table-cell">
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${prediction.confidence}%` }}
                    ></div>
                    <span className="confidence-text">{prediction.confidence}%</span>
                  </div>
                </div>
                <div className="table-cell">
                  <span className={getRiskBadgeClass(prediction.riskLevel)}>
                    {prediction.riskLevel === 'High' && '⚠️ '}
                    {prediction.riskLevel}
                  </span>
                </div>
                <div className="table-cell">
                  <button className="btn-icon">
                    👁️
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;