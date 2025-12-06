import React, { useState, useEffect } from "react";
import ImageUploader from "./components/ImageUploader";
import ResultDisplay from "./components/ResultDisplay";
import HistoryPanel from "./components/HistoryPanel";
import InfoCards from "./components/InfoCards";
import TeamInfo from "./components/TeamInfo";

const CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TB"];

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState("detect"); // detect | history | info | team
  const [prediction, setPrediction] = useState(null);
  const [preview, setPreview] = useState(null);
  const [history, setHistory] = useState([]);

  // load prefs + history
  useEffect(() => {
    const savedDark = localStorage.getItem("darkMode");
    if (savedDark !== null) setDarkMode(savedDark === "true");

    const savedHist = localStorage.getItem("predictionHistory");
    if (savedHist) setHistory(JSON.parse(savedHist));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("darkMode", darkMode.toString());
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem("predictionHistory", JSON.stringify(history));
  }, [history]);

  const handleNewPrediction = (result) => {
    setPrediction(result);
    setActiveTab("detect");
    setHistory((prev) => [result, ...prev].slice(0, 20));
  };

  const handleHistorySelect = (item) => {
    setPrediction(item);
    setPreview(item.preview || null);
    setActiveTab("detect");
  };

  const bg = darkMode
    ? "bg-slate-950 text-slate-100"
    : "bg-slate-100 text-slate-900";

  return (
    <div className={`${bg} min-h-screen transition-colors`}>
      <header className="border-b border-slate-800/20 dark:border-slate-700/60 bg-gradient-to-r from-indigo-500/10 via-slate-500/5 to-teal-500/10 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold">
              PneumoVision&nbsp;
              <span className="text-indigo-500">X-Ray AI</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
              Multi-class chest X-ray classifier (COVID / Normal / Pneumonia / TB){" "}
              + Grad-CAM explainability
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-xs text-slate-500 dark:text-slate-400">
              {darkMode ? "Dark" : "Light"} mode
            </span>
            <button
              onClick={() => setDarkMode((d) => !d)}
              className="relative inline-flex h-8 w-14 items-center rounded-full border border-slate-300 dark:border-slate-600 bg-white/80 dark:bg-slate-900/70 shadow-sm"
            >
              <span
                className={`inline-flex h-6 w-6 transform items-center justify-center rounded-full bg-indigo-500 text-white text-xs shadow transition-transform ${
                  darkMode ? "translate-x-6" : "translate-x-1"
                }`}
              >
                {darkMode ? "🌙" : "☀️"}
              </span>
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 grid gap-6 lg:grid-cols-[2fr,1.2fr]">
        {/* Left: upload + result */}
        <section className="space-y-4">
          {/* Tabs */}
          <div className="flex gap-2 text-sm">
            {[
              { id: "detect", label: "Detection" },
              { id: "history", label: "History" },
              { id: "info", label: "Model Insights" },
              { id: "team", label: "Team" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 rounded-full border text-xs sm:text-sm transition ${
                  activeTab === tab.id
                    ? "bg-indigo-500 text-white border-indigo-500 shadow-sm"
                    : darkMode
                    ? "border-slate-700 text-slate-300 hover:bg-slate-800"
                    : "border-slate-300 text-slate-700 hover:bg-white"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === "detect" && (
            <>
              <ImageUploader
                darkMode={darkMode}
                onPrediction={handleNewPrediction}
                onPreview={setPreview}
              />
              <ResultDisplay
                prediction={prediction}
                preview={preview}
                darkMode={darkMode}
                classNames={CLASS_NAMES}
              />
            </>
          )}

          {activeTab === "history" && (
            <HistoryPanel
              history={history}
              darkMode={darkMode}
              onSelect={handleHistorySelect}
            />
          )}

          {activeTab === "info" && <InfoCards darkMode={darkMode} />}

          {activeTab === "team" && <TeamInfo darkMode={darkMode} />}
        </section>

        {/* Right: always show quick info + metrics */}
        <aside className="space-y-4">
          <InfoCards darkMode={darkMode} compact />
          <HistoryPanel
            history={history.slice(0, 5)}
            darkMode={darkMode}
            onSelect={handleHistorySelect}
            compact
          />
        </aside>
      </main>

      <footer className="py-4 text-center text-xs text-slate-500 dark:text-slate-500 border-t border-slate-200/40 dark:border-slate-800/60">
        Built as an enhanced IBM major project — multi-class X-ray AI with
        Grad-CAM, class balancing & model comparison.
      </footer>
    </div>
  );
}

export default App;
