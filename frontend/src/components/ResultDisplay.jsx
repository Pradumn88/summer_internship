import React, { useState, useEffect } from 'react';

function ResultDisplay({ prediction, darkMode }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  if (!prediction) return null;

  const isPneumonia = prediction.prediction === 'PNEUMONIA';
  const confidencePercentage = (prediction.confidence * 100).toFixed(2);
  
  // Conditional styling
  const resultContainerClass = isPneumonia 
    ? darkMode 
      ? 'bg-gradient-to-br from-red-900/20 to-red-800/10 border-red-700 text-red-200' 
      : 'bg-gradient-to-br from-red-50 to-red-100 border-red-400 text-red-800'
    : darkMode 
      ? 'bg-gradient-to-br from-green-900/20 to-green-800/10 border-green-700 text-green-200' 
      : 'bg-gradient-to-br from-green-50 to-green-100 border-green-400 text-green-800';

  // Conditional icon
  const icon = isPneumonia ? (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-red-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ) : (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-green-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );

  // Handle feedback submission
  const handleFeedback = (isAccurate) => {
    // In a real app, you would send this to your backend
    console.log(`User feedback: ${isAccurate ? 'Accurate' : 'Inaccurate'}`);
    setFeedbackSubmitted(true);
    setShowFeedback(false);
  };

  return (
    <div className={`mt-8 p-6 rounded-2xl border-2 ${resultContainerClass} text-center shadow-xl transition-all duration-700 ease-out transform ${
      isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
    }`}>
      <div className="flex items-center justify-center mb-4">
        {icon}
        <h2 className="text-2xl font-bold">Prediction Result</h2>
      </div>
      
      <div className="bg-white/20 dark:bg-black/20 backdrop-blur-sm p-4 rounded-xl mb-4">
        <p className="text-xl mb-2">
          Diagnosis: <span className="font-extrabold text-2xl">{prediction.prediction}</span>
        </p>
        <p className="text-xl flex items-center justify-center">
          Confidence: 
          <span className="font-extrabold ml-2 text-2xl">{confidencePercentage}%</span>
          <button 
            className="ml-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 relative"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            onClick={() => setShowTooltip(!showTooltip)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {showTooltip && (
              <div className={`absolute left-full ml-2 mt-2 p-3 rounded-lg shadow-lg text-sm max-w-xs z-10 ${
                darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800'
              }`}>
                Confidence measures how certain the AI is about its prediction. 
                It's calculated based on patterns learned from thousands of X-ray images.
              </div>
            )}
          </button>
        </p>
      </div>

      {isPneumonia && (
        <div className={`p-4 rounded-lg mb-4 ${
          darkMode ? 'bg-red-900/30 border border-red-800' : 'bg-red-100 border border-red-200'
        }`}>
          <p className="font-medium">
            <span className="font-semibold">Important:</span> This is an AI-generated prediction and should not replace professional medical advice.
          </p>
        </div>
      )}
      
      {/* Feedback Section */}
      {!feedbackSubmitted ? (
        <div className="mt-4">
          <button 
            onClick={() => setShowFeedback(!showFeedback)}
            className={`text-sm px-3 py-1 rounded-full ${
              darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'
            } transition-colors`}
          >
            {showFeedback ? 'Cancel Feedback' : 'Was this helpful?'}
          </button>
          
          {showFeedback && (
            <div className="mt-3 flex justify-center space-x-3">
              <button 
                onClick={() => handleFeedback(true)}
                className="px-4 py-2 bg-green-500 text-white rounded-full flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Yes
              </button>
              <button 
                onClick={() => handleFeedback(false)}
                className="px-4 py-2 bg-red-500 text-white rounded-full flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                No
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="mt-4 text-green-500 dark:text-green-400 text-sm">
          Thank you for your feedback!
        </div>
      )}
      
      {/* Comparison Examples */}
      <div className="mt-6 pt-4 border-t border-gray-300 dark:border-gray-700">
        <h3 className="font-bold mb-3">Compare with Examples</h3>
        <div className="flex justify-center space-x-4">
          <div className="text-center">
            <div className={`w-24 h-24 mx-auto rounded-lg overflow-hidden border-2 ${
              darkMode ? 'border-blue-500' : 'border-blue-400'
            }`}>
              <div className="bg-gray-200 border-2 border-dashed rounded-xl w-full h-full" />
            </div>
            <p className="text-sm mt-1">Normal X-ray</p>
          </div>
          <div className="text-center">
            <div className={`w-24 h-24 mx-auto rounded-lg overflow-hidden border-2 ${
              darkMode ? 'border-red-500' : 'border-red-400'
            }`}>
              <div className="bg-gray-200 border-2 border-dashed rounded-xl w-full h-full" />
            </div>
            <p className="text-sm mt-1">Pneumonia X-ray</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultDisplay;