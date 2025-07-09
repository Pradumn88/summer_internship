// E:\PNENEUMONIA-DETECTION\frontend\src\App.jsx

import React from 'react';
import ImageUploader from './components/ImageUploader'; // Ensure this path is correct
import './index.css'; // Your Tailwind CSS import

function App() {
  return (
    // Main container with a subtle gradient background and centered content
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center justify-center p-4 sm:p-6">
      
      {/* Header Section */}
      <header className="w-full max-w-2xl text-center mb-8 md:mb-12">
        <h1 className="text-4xl sm:text-5xl font-extrabold text-blue-800 tracking-tight mb-4 drop-shadow-sm">
          Pneumonia Insight AI
        </h1>
        <p className="text-md sm:text-lg text-gray-700 leading-relaxed">
          Upload a chest X-ray image for an instant, AI-powered pneumonia diagnosis.
        </p>
      </header>

      {/* Main Content Card (where the uploader will sit) */}
      <div className="bg-white p-6 sm:p-8 rounded-xl shadow-2xl w-full max-w-md border border-gray-200 transform hover:scale-102 transition-transform duration-300 ease-in-out">
        <ImageUploader />
      </div>

      {/* Footer Section */}
      <footer className="mt-10 text-gray-600 text-sm text-center">
        &copy; {new Date().getFullYear()} AI-Powered Diagnosis. All rights reserved.
      </footer>
    </div>
  );
}

export default App;