// E:\PNEUMONIA-DETECTION\frontend\src\components\ImageUploader.jsx

import React, { useState } from 'react';
import ResultDisplay from './ResultDisplay'; // Ensure this path is correct

function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null); // State for image preview
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPrediction(null); // Clear previous prediction
    setError(null); // Clear previous errors

    // Create a URL for image preview
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select an X-ray image first.');
      return;
    }

    setLoading(true);
    setError(null); // Clear errors before new attempt
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setPrediction(data);
        setError(null);
      } else {
        // Handle server-side errors (e.g., if the file format is wrong)
        setError(data.detail || 'An unknown error occurred during prediction.');
      }
    } catch (err) {
      // Handle network errors (e.g., backend not running)
      setError('Failed to connect to the server. Please ensure the backend API is running at http://localhost:8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Custom File Input Area */}
      <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-blue-300 rounded-lg cursor-pointer hover:border-blue-500 transition-colors duration-200 min-h-[120px]">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden" // Hide the default input
          id="file-upload" // Link label to input
        />
        <label htmlFor="file-upload" className="flex flex-col items-center justify-center w-full h-full text-blue-600 cursor-pointer">
          {selectedFile ? (
            <>
              {/* File selected state */}
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-semibold text-lg text-center break-words px-2">
                {selectedFile.name}
              </span>
              <span className="text-sm text-gray-500">
                ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </span>
            </>
          ) : (
            <>
              {/* No file selected state */}
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="font-semibold text-lg text-center">
                Click to select an X-ray image
              </span>
              <span className="text-sm text-gray-500 mt-1">
                (PNG, JPG, JPEG recommended)
              </span>
            </>
          )}
        </label>
      </div>

      {/* Image Preview */}
      {previewUrl && (
        <div className="mt-4 flex justify-center">
          <img
            src={previewUrl}
            alt="Image Preview"
            className="max-w-xs max-h-48 object-contain rounded-lg shadow-md border border-gray-200"
          />
        </div>
      )}

      {/* Predict Button */}
      <button
        onClick={handleSubmit}
        disabled={loading || !selectedFile} // Disable if loading or no file selected
        className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-all duration-300 ease-in-out
          ${loading || !selectedFile // Conditional styling for disabled/loading states
            ? 'bg-blue-300 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
          }`}
      >
        {loading ? (
          <span className="flex items-center justify-center">
            {/* Simple Loading Spinner */}
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing X-ray...
          </span>
        ) : 'Predict Pneumonia'}
      </button>

      {/* Error Message Display */}
      {error && (
        <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-center shadow-sm">
          <span className="font-medium">Error:</span> {error}
        </div>
      )}

      {/* Prediction Result Display */}
      {prediction && <ResultDisplay prediction={prediction} />}
    </div>
  );
}

export default ImageUploader;