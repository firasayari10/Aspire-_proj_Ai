import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './ImageDropzone.css';

const ImageDropzone = () => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [aiAnalysis, setAiAnalysis] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setProcessedImage(null);
      setAiAnalysis(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    maxFiles: 1
  });

  const processImage = async () => {
    if (!image) return;
    try {
      setProcessing(true);
      setError(null);

      const formData = new FormData();
      formData.append('image', image);

      const response = await axios.post('http://localhost:5000/api/process-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setProcessedImage({
          image: `data:image/jpeg;base64,${response.data.processed_image}`,
          detections: response.data.detections
        });
        setAiAnalysis(response.data.ai_analysis);
      } else {
        throw new Error('Processing failed');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const removeImage = () => {
    setImage(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setProcessedImage(null);
    setAiAnalysis(null);
  };

  const groupDetectionsByModel = (detections) => {
    return detections.reduce((acc, det) => {
      if (!acc[det.model]) {
        acc[det.model] = [];
      }
      acc[det.model].push(det);
      return acc;
    }, {});
  };

  return (
    <div className="image-dropzone-container">
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <p>{isDragActive ? "Drop the image here..." : "Drag and drop an image here, or click to select a file"}</p>
      </div>

      {error && <div className="error-message">{error}</div>}

      {previewUrl && (
        <div className="preview-card">
          <img 
            src={processedImage?.image || previewUrl} 
            alt="preview" 
            className="preview-image"
          />
          <div className="preview-controls">
            <button 
              className="process-button"
              onClick={processImage}
              disabled={processing}
            >
              {processing ? 'Processing...' : 'Process'}
            </button>
            <button 
              className="remove-button"
              onClick={removeImage}
            >
              ×
            </button>
          </div>

          {processedImage?.detections && (
            <div className="results-container">
              <div className="detections-box">
                <h3>Detections</h3>
                {Object.entries(groupDetectionsByModel(processedImage.detections)).map(([model, detections]) => (
                  <div key={model} className="model-results">
                    <h4>
                      {model === 'waste' ? 'Waste Detection' :
                       model === 'medwaste' ? 'Medical Waste Detection' :
                       'Battery Detection'}
                    </h4>
                    {detections.map((det, i) => (
                      <div key={i} className="detection-item">
                        <strong>{det.class_name}</strong> – Confidence: {(det.confidence * 100).toFixed(1)}%
                      </div>
                    ))}
                  </div>
                ))}
              </div>

              {aiAnalysis?.success && (
                <div className="ai-analysis-box">
                  <h3>AI Analysis</h3>
                  <div className="ai-analysis-content">
                    {aiAnalysis.analysis
                      .split('\n')
                      .filter(line => line.trim() && !line.match(/^\d+\./))
                      .map((line, i) => (
                        <p key={i}>{line}</p>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageDropzone;
