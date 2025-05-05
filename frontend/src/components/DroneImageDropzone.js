import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './ImageDropzone.css';

const DroneImageDropzone = () => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setProcessedImage(null);
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

      const response = await axios.post('http://localhost:5000/api/process-drone-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setProcessedImage({
          image: `data:image/jpeg;base64,${response.data.image}`,
          detections: response.data.detections
        });
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
  };

  return (
    <div className="image-dropzone-container">
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <p>{isDragActive ? "Drop the drone image here..." : "Drag and drop a drone image here, or click to select a file"}</p>
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
                {processedImage.detections.length === 0 && <p>No waste detected.</p>}
                {processedImage.detections.map((det, i) => (
                  <div key={i} className="detection-item">
                    <strong>{det.class_name}</strong> – Confidence: {(det.confidence * 100).toFixed(1)}%
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DroneImageDropzone; 