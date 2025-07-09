// E:\PNEUMONIA-DETECTION\frontend\src\components\ResultDisplay.jsx

import React from 'react';

function ResultDisplay({ prediction }) {
  if (!prediction) return null; // Don't render if no prediction data

  const isPneumonia = prediction.prediction === 'PNEUMONIA';
  
  // Conditional styling based on diagnosis
  const resultContainerClass = isPneumonia 
    ? 'bg-red-50 border-red-400 text-red-800' // For Pneumonia
    : 'bg-green-50 border-green-400 text-green-800'; // For Normal

  // Conditional icon based on diagnosis (using simple SVG icons)
  const icon = isPneumonia ? (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-red-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ) : (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-green-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );

  return (
    <div className={`mt-8 p-5 rounded-xl border-2 ${resultContainerClass} text-center shadow-md`}>
      <h2 className="text-2xl font-bold mb-4 flex items-center justify-center">
        {icon} Prediction Result
      </h2>
      <p className="text-xl mb-2">
        Diagnosis: <span className="font-extrabold">{prediction.prediction}</span>
      </p>
      <p className="text-xl">
        Confidence: <span className="font-extrabold">{(prediction.confidence * 100).toFixed(2)}%</span>
      </p>
      {isPneumonia && (
        <p className="text-sm mt-4 text-red-700 font-medium">
          <span className="text-red-900 font-semibold">Important:</span> This is an AI-generated prediction and should not replace professional medical advice.
        </p>
      )}
    </div>
  );
}

export default ResultDisplay;