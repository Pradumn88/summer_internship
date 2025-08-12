import React, { useState, useRef } from 'react';

function ImageUploader({ onPrediction, darkMode }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const imgRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPrediction(null);
    setError(null);
    setZoomLevel(1);

    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleMouseMove = (e) => {
    if (!imgRef.current) return;
    
    const { left, top, width, height } = imgRef.current.getBoundingClientRect();
    const x = ((e.pageX - left) / width) * 100;
    const y = ((e.pageY - top) / height) * 100;
    setPosition({ x, y });
  };

  const handleZoom = (direction) => {
    setZoomLevel(prev => {
      const newLevel = direction === 'in' ? Math.min(prev + 0.5, 3) : Math.max(prev - 0.5, 1);
      return newLevel;
    });
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select an X-ray image first.');
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Server response was not ok');
      }
      
      const data = await response.json();
      onPrediction(data);
      setError(null);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* File Input Area */}
      <div className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-lg cursor-pointer min-h-[120px] transition-colors ${
        darkMode ? 'border-gray-600 hover:border-blue-500' : 'border-blue-300 hover:border-blue-500'
      }`}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label htmlFor="file-upload" className="flex flex-col items-center justify-center w-full h-full cursor-pointer">
          {selectedFile ? (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-semibold text-lg text-center break-words px-2">
                {selectedFile.name}
              </span>
              <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </span>
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="font-semibold text-lg text-center">
                Click to select an X-ray image
              </span>
              <span className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                (PNG, JPG, JPEG recommended)
              </span>
            </>
          )}
        </label>
      </div>

      {/* Image Preview with Zoom Controls */}
      {previewUrl && (
        <div className="mt-4">
          <div 
            className="relative overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 max-w-md mx-auto"
            onMouseMove={handleMouseMove}
          >
            <img
              ref={imgRef}
              src={previewUrl}
              alt="Image Preview"
              className="w-full object-contain"
              style={{
                transform: `scale(${zoomLevel})`,
                transformOrigin: `${position.x}% ${position.y}%`,
                transition: 'transform 0.3s ease',
                maxHeight: '300px'
              }}
            />
            
            {/* Zoom Controls */}
            <div className="absolute bottom-4 right-4 flex space-x-2">
              <button 
                onClick={() => handleZoom('out')}
                disabled={zoomLevel <= 1}
                className={`p-2 rounded-full ${
                  darkMode ? 'bg-gray-800/80 text-white' : 'bg-white/80 text-gray-800'
                } shadow-md backdrop-blur-sm ${zoomLevel <= 1 ? 'opacity-50' : ''}`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                </svg>
              </button>
              <button 
                onClick={() => handleZoom('in')}
                disabled={zoomLevel >= 3}
                className={`p-2 rounded-full ${
                  darkMode ? 'bg-gray-800/80 text-white' : 'bg-white/80 text-gray-800'
                } shadow-md backdrop-blur-sm ${zoomLevel >= 3 ? 'opacity-50' : ''}`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </button>
            </div>
          </div>
          
          <div className="text-center mt-2 text-sm text-gray-500 dark:text-gray-400">
            Scroll or use buttons to zoom • Click and drag to pan
          </div>
        </div>
      )}

      {/* Predict Button */}
      <button
        onClick={handleSubmit}
        disabled={loading || !selectedFile}
        className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-all duration-300 ease-in-out ${
          loading || !selectedFile 
            ? 'bg-blue-300 cursor-not-allowed dark:bg-blue-900/50' 
            : 'bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800'
        } shadow-lg transform hover:scale-[1.02] active:scale-[0.98]`}
      >
        {loading ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing X-ray...
          </span>
        ) : (
          <span className="flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            Predict Pneumonia
          </span>
        )}
      </button>

      {/* Error Message */}
      {error && (
        <div className={`mt-4 p-3 rounded-lg text-center shadow-sm ${
          darkMode ? 'bg-red-900/30 text-red-200 border border-red-700' : 'bg-red-100 text-red-700 border border-red-300'
        }`}>
          <span className="font-medium">Error:</span> {error}
        </div>
      )}
    </div>
  );
}

export default ImageUploader;