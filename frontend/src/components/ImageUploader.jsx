import React, { useState } from "react";

// Load backend URL from .env
// ImageUploader.jsx (top of file)
const API_URL =
  `${process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000"}/predict`;


function ImageUploader({ darkMode, onPrediction, onPreview }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setError("");

    if (f && onPreview) {
      const reader = new FileReader();
      reader.onloadend = () => onPreview(reader.result);
      reader.readAsDataURL(f);
    } else if (!f && onPreview) {
      onPreview(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a chest X-ray image first.");
      return;
    }

    if (!API_URL) {
      setError("Backend URL is missing. Check your .env file.");
      return;
    }

    setError("");
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Prediction failed");
      }

      const data = await res.json();

      const result = {
        id: Date.now(),
        fileName: file.name,
        prediction: data.prediction,
        confidence: data.confidence, 
        probabilities: data.probabilities,
        gradcam: data.gradcam || null,
        createdAt: new Date().toISOString(),
      };

      onPrediction(result);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong");
    } finally {
      setUploading(false);
    }
  };

  const cardStyle = darkMode
    ? "bg-slate-900 border-slate-700"
    : "bg-white border-slate-200";

  return (
    <div className={`rounded-2xl border p-4 sm:p-5 shadow-sm ${cardStyle}`}>
      <h2 className="text-lg font-semibold mb-2">Upload Chest X-ray</h2>
      <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mb-3">
        JPEG / PNG chest X-ray image. The model will classify it into COVID,
        Normal, Pneumonia or TB and generate a Grad-CAM heatmap.
      </p>

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <label className="cursor-pointer inline-flex items-center justify-center px-4 py-2 rounded-xl border border-dashed border-indigo-400 bg-indigo-50/60 text-indigo-700 text-sm hover:bg-indigo-100 dark:bg-slate-900 dark:border-indigo-500/70 dark:text-indigo-300">
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          📁 Choose X-ray
        </label>

        <button
          onClick={handleUpload}
          disabled={uploading}
          className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-sm font-medium shadow hover:bg-indigo-700 disabled:opacity-60"
        >
          {uploading ? "Analyzing…" : "Run AI Detection"}
        </button>
      </div>

      <p className="mt-2 text-xs text-slate-500 dark:text-slate-500">
        {file ? `Selected: ${file.name}` : "No file selected yet."}
      </p>

      {error && (
        <p className="mt-2 text-xs text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}
    </div>
  );
}

export default ImageUploader;
