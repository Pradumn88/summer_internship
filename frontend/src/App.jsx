import React, { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import HistoryPanel from './components/HistoryPanel';
import InfoCards from './components/InfoCards';
import TeamInfo from './components/TeamInfo';
import './index.css';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload', 'history', 'info', 'team'

  // Initialize dark mode from localStorage or system preference
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setDarkMode(savedDarkMode || systemPrefersDark);
    
    // Load history from localStorage
    const savedHistory = localStorage.getItem('predictionHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Save dark mode preference
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  // Handle new prediction
  const handleNewPrediction = (result) => {
    setPrediction(result);
    
    // Add to history (keep last 5)
    const newHistory = [result, ...history.slice(0, 4)];
    setHistory(newHistory);
    localStorage.setItem('predictionHistory', JSON.stringify(newHistory));
  };

  return (
    <div className={`min-h-screen flex flex-col items-center justify-between p-4 sm:p-6 transition-colors duration-300 ${
      darkMode 
        ? 'bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100' 
        : 'bg-gradient-to-br from-blue-50 to-indigo-100 text-gray-800'
    }`}>
      <div className="w-full max-w-6xl">
        {/* Header */}
        <header className="w-full text-center mb-8 md:mb-12">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-4xl sm:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-700 tracking-tight drop-shadow-sm">
              Pneumonia Insight AI
            </h1>
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-full bg-white/20 backdrop-blur-sm shadow-lg"
              aria-label="Toggle dark mode"
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
          <p className={`text-md sm:text-lg ${
            darkMode ? 'text-gray-300' : 'text-gray-700'
          } leading-relaxed max-w-2xl mx-auto`}>
            Upload a chest X-ray for an instant, AI-powered pneumonia diagnosis
          </p>
        </header>

        {/* Main Content */}
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Left Panel - Tabs Content */}
          <div className="flex-1">
            <div className={`bg-white dark:bg-gray-800/80 backdrop-blur-sm p-6 sm:p-8 rounded-2xl shadow-2xl w-full border ${
              darkMode ? 'border-gray-700' : 'border-gray-200'
            } transition-all duration-500 ease-in-out transform hover:scale-[1.01]`}>
              {/* Tab Navigation */}
              <div className="flex mb-6 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
                <button 
                  className={`py-2 px-4 font-medium whitespace-nowrap ${
                    activeTab === 'upload' 
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-500' 
                      : 'text-gray-500 dark:text-gray-400'
                  }`}
                  onClick={() => setActiveTab('upload')}
                >
                  Diagnosis
                </button>
                <button 
                  className={`py-2 px-4 font-medium whitespace-nowrap ${
                    activeTab === 'history' 
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-500' 
                      : 'text-gray-500 dark:text-gray-400'
                  }`}
                  onClick={() => setActiveTab('history')}
                >
                  History
                </button>
                <button 
                  className={`py-2 px-4 font-medium whitespace-nowrap ${
                    activeTab === 'info' 
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-500' 
                      : 'text-gray-500 dark:text-gray-400'
                  }`}
                  onClick={() => setActiveTab('info')}
                >
                  Learn More
                </button>
                <button 
                  className={`py-2 px-4 font-medium whitespace-nowrap ${
                    activeTab === 'team' 
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-500' 
                      : 'text-gray-500 dark:text-gray-400'
                  }`}
                  onClick={() => setActiveTab('team')}
                >
                  Our Team
                </button>
              </div>
              
              {/* Tab Content */}
              {activeTab === 'upload' && (
                <>
                  <ImageUploader 
                    onPrediction={handleNewPrediction} 
                    darkMode={darkMode} 
                  />
                  {prediction && <ResultDisplay prediction={prediction} darkMode={darkMode} />}
                </>
              )}
              
              {activeTab === 'history' && (
                <HistoryPanel history={history} darkMode={darkMode} />
              )}
              
              {activeTab === 'info' && (
                <InfoCards darkMode={darkMode} />
              )}
              
              {activeTab === 'team' && (
                <TeamInfo darkMode={darkMode} />
              )}
            </div>
          </div>
          
          {/* Right Panel - About This AI */}
          <div className="lg:w-1/3">
            <div className={`bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-2xl shadow-xl border ${
              darkMode ? 'border-gray-700' : 'border-gray-200'
            }`}>
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                About This AI
              </h2>
              <div className="space-y-4">
                <div className="p-4 bg-blue-50/50 dark:bg-blue-900/20 rounded-lg">
                  <h3 className="font-bold mb-2">How It Works</h3>
                  <p className="text-sm">
                    Our AI uses a deep learning model trained on thousands of chest X-rays 
                    to identify patterns associated with pneumonia. It analyzes your uploaded 
                    image and provides a diagnosis with confidence level.
                  </p>
                </div>
                
                <div className="p-4 bg-green-50/50 dark:bg-green-900/20 rounded-lg">
                  <h3 className="font-bold mb-2">Data Privacy</h3>
                  <p className="text-sm">
                    Your X-ray images are processed temporarily and never stored on our servers. 
                    All analysis happens on your device or in secure cloud processing.
                  </p>
                </div>
                
                <div className="p-4 bg-yellow-50/50 dark:bg-yellow-900/20 rounded-lg">
                  <h3 className="font-bold mb-2">Important Notice</h3>
                  <p className="text-sm">
                    This tool provides AI-generated analysis and should not replace professional 
                    medical advice. Always consult a healthcare provider for medical decisions.
                  </p>
                </div>
                
                <div className="mt-6">
                  <h3 className="font-bold mb-2">Trusted Resources</h3>
                  <div className="flex flex-wrap gap-2">
                    <a href="https://www.who.int/publications/i/item/9789241507813" target="_blank" rel="noopener noreferrer" className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition">
                      WHO
                    </a>
                    <a href="https://www.cdc.gov/pneumonia/about/index.html" target="_blank" rel="noopener noreferrer" className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition">
                      CDC
                    </a>
                    <a href="https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonia" target="_blank" rel="noopener noreferrer" className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition">
                      American Lung Association
                    </a>
                  </div>
                </div>
                
                <div className="pt-4 mt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
                  <p>AI Model Version: PneumoNet v2.1</p>
                  <p>Last Updated: August 2025</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className={`mt-10 text-center text-sm ${
        darkMode ? 'text-gray-400' : 'text-gray-600'
      }`}>
        &copy; {new Date().getFullYear()} Pneumonia Insight AI. All rights reserved.
      </footer>
    </div>
  );
}

export default App;