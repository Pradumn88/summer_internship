import React from 'react';

function HistoryPanel({ history, darkMode }) {
  if (history.length === 0) {
    return (
      <div className="text-center py-10">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-xl font-medium mt-4">No History Yet</h3>
        <p className={`mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          Your past predictions will appear here
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Recent Predictions</h2>
      <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
        {history.map((item, index) => {
          const isPneumonia = item.prediction === 'PNEUMONIA';
          const confidence = (item.confidence * 100).toFixed(2);
          const date = new Date().toLocaleDateString();
          
          return (
            <div 
              key={index} 
              className={`p-4 rounded-xl border ${
                darkMode 
                  ? isPneumonia 
                    ? 'bg-red-900/20 border-red-800' 
                    : 'bg-green-900/20 border-green-800'
                  : isPneumonia 
                    ? 'bg-red-50 border-red-200' 
                    : 'bg-green-50 border-green-200'
              }`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <div className="flex items-center">
                    <span className={`text-lg font-bold ${
                      isPneumonia 
                        ? darkMode ? 'text-red-400' : 'text-red-700'
                        : darkMode ? 'text-green-400' : 'text-green-700'
                    }`}>
                      {item.prediction}
                    </span>
                    <span className="ml-3 bg-gray-200 dark:bg-gray-700 text-xs px-2 py-1 rounded-full">
                      {confidence}%
                    </span>
                  </div>
                  <div className="text-sm mt-2 text-gray-500 dark:text-gray-400">
                    Analyzed on {date}
                  </div>
                </div>
                <button className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default HistoryPanel;