import React from "react";

function TeamInfo({ darkMode }) {
  const cardStyle = darkMode
    ? "bg-slate-900 border-slate-700"
    : "bg-white border-slate-200";

  return (
    <div className={`rounded-2xl border p-4 shadow-sm text-sm ${cardStyle}`}>
      <h2 className="font-semibold mb-2">Team & Contributions</h2>
      <ul className="list-disc list-inside space-y-1.5 text-[13px]">
        <li>
          <span className="font-medium">Model Evolution:</span> Upgraded from
          binary Pneumonia detection to a 4-class system (COVID / Normal /
          Pneumonia / TB), added class-balanced training & fine-tuning.
        </li>
        <li>
          <span className="font-medium">Deep Learning:</span> Implemented
          MobileNetV2 baseline, DenseNet121 variant, and an ensemble strategy
          with focal loss for imbalanced data.
        </li>
        <li>
          <span className="font-medium">MLOps & Backend:</span> FastAPI
          inference API, prediction logging, Grad-CAM generation, JSON
          responses usable by any frontend.
        </li>
        <li>
          <span className="font-medium">Frontend & UX:</span> React + Tailwind
          dashboard, history panel, dual-view (X-ray + heatmap) and per-class
          probability bars for easier explanation in viva.
        </li>
      </ul>
    </div>
  );
}

export default TeamInfo;
