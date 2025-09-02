import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import axios from 'axios';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState('efficientnet');
  const [useEnsemble, setUseEnsemble] = useState(false);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const error = rejectedFiles[0].errors[0];
      toast.error(`File rejected: ${error.message}`);
      return;
    }

    const file = acceptedFiles[0];
    if (file) {
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File too large. Maximum size is 10MB.');
        return;
      }

      setSelectedFile(file);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);

      toast.success('Image uploaded successfully!');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024
  });

  const removeFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setAnalysisResult(null);
    setUploadProgress(0);
  };

  const analyzeImage = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first.');
      return;
    }

    setIsAnalyzing(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const endpoint = useEnsemble ? '/predict/ensemble' : '/predict';
      const params = useEnsemble ? {} : { model_name: selectedModel };

      const response = await axios.post(
        `http://localhost:8000${endpoint}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          params,
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(progress);
          },
        }
      );

      setAnalysisResult(response.data);
      toast.success('Analysis completed successfully!');

    } catch (error) {
      console.error('Analysis error:', error);
      toast.error(error.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">

        {/* Header */}
        <div className="upload-header">
          <h1 className="page-title">
            🔬 Retinal Image Analysis
          </h1>
          <p className="page-description">
            Upload a fundus image for AI-powered diabetic retinopathy detection and classification
          </p>
        </div>

        {/* Model Selection */}
        <div className="model-selection">
          <h3 className="selection-title">Analysis Settings</h3>
          <div className="selection-options">
            <div className="option-group">
              <label className="checkbox-option">
                <input
                  type="checkbox"
                  checked={useEnsemble}
                  onChange={(e) => setUseEnsemble(e.target.checked)}
                />
                <span className="checkmark"></span>
                <div className="option-content">
                  <span className="option-title">Ensemble Model</span>
                  <span className="option-description">
                    Use multiple models for higher accuracy (recommended for clinical use)
                  </span>
                </div>
              </label>
            </div>

            {!useEnsemble && (
              <div className="model-selector">
                <label className="selector-label">Single Model:</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="model-select"
                >
                  <option value="efficientnet">EfficientNet (Fast & Accurate)</option>
                  <option value="vit">Vision Transformer (High Precision)</option>
                  <option value="hybrid">Hybrid CNN-ViT (Best Performance)</option>
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Upload Area */}
        <div className="upload-section">
          {!selectedFile ? (
            <div
              className={`dropzone ${isDragActive ? 'active' : ''}`}
              {...getRootProps()}
            >
              <input {...getInputProps()} />
              <div className="dropzone-content">
                <div className="dropzone-icon">
                  📁
                </div>

                <div className="dropzone-text">
                  <h3 className="dropzone-title">
                    {isDragActive ? 'Drop your image here' : 'Upload Fundus Image'}
                  </h3>
                  <p className="dropzone-description">
                    Drag and drop a retinal fundus image, or click to browse
                  </p>
                  <div className="file-requirements">
                    <span>Supported formats: JPEG, PNG, BMP, TIFF</span>
                    <span>Maximum size: 10MB</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="image-preview-card">
              <div className="preview-header">
                <div className="preview-info">
                  <span>🖼️ {selectedFile.name}</span>
                  <span className="file-size">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
                <button onClick={removeFile} className="remove-btn">
                  ✕
                </button>
              </div>

              <div className="preview-image">
                <img src={preview} alt="Preview" />
              </div>

              <div className="preview-actions">
                <button
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="btn btn-primary btn-large"
                >
                  {isAnalyzing ? '🔄 Analyzing...' : '🧠 Start Analysis ⚡'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Analysis Progress */}
        {isAnalyzing && (
          <div className="analysis-progress">
            <div className="progress-content">
              <div className="progress-header">
                <span className="progress-text">Analyzing retinal image...</span>
              </div>
              <div className="progress-bar-container">
                <div className="progress-bar-track">
                  <div 
                    className="progress-bar-fill"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <div className="progress-percentage">
                  {Math.round(uploadProgress)}%
                </div>
              </div>
              <div className="progress-steps">
                <div className="step completed">
                  <span>✓ Image uploaded</span>
                </div>
                <div className={`step ${uploadProgress > 50 ? 'completed' : 'active'}`}>
                  <span>🔄 AI processing</span>
                </div>
                <div className="step">
                  <span>👁️ Generating results</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {analysisResult && (
          <div className="results-section">
            <div className="results-card">
              <div className="results-header">
                <div className="results-title">
                  ✅ Analysis Complete
                </div>
                <div className="results-meta">
                  Model: {analysisResult.model_used} • 
                  Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div className="results-content">
                <div className="primary-result">
                  <div className="result-classification">
                    <span className="classification-label">Classification:</span>
                    <span className={`classification-value ${analysisResult.risk_level.toLowerCase()}`}>
                      {analysisResult.severity_description}
                    </span>
                  </div>

                  <div className="risk-indicator">
                    <span className="risk-label">Risk Level:</span>
                    <span className={`risk-badge ${analysisResult.risk_level.toLowerCase()}`}>
                      {analysisResult.requires_immediate_attention && '⚠️ '}
                      {analysisResult.risk_level}
                    </span>
                  </div>
                </div>

                {/* Clinical Recommendations */}
                {analysisResult.recommendations && (
                  <div className="recommendations">
                    <h4 className="recommendations-title">Clinical Recommendations:</h4>
                    <ul className="recommendations-list">
                      {analysisResult.recommendations.map((recommendation, index) => (
                        <li key={index} className="recommendation-item">
                          {recommendation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Urgent Alert */}
                {analysisResult.requires_immediate_attention && (
                  <div className="urgent-alert">
                    <span>⚠️</span>
                    <div className="alert-content">
                      <span className="alert-title">Immediate Medical Attention Required</span>
                      <span className="alert-text">
                        This case requires urgent ophthalmologist consultation.
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;