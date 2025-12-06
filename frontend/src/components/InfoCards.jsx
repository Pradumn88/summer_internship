import React from "react";

function InfoCards({ darkMode, compact }) {
  const cardStyle = darkMode
    ? "bg-slate-900 border-slate-700"
    : "bg-white border-slate-200";

  return (
    <div
      className={`rounded-2xl border p-4 shadow-sm space-y-3 ${
        cardStyle
      } ${compact ? "text-xs" : "text-sm"}`}
    >
      <h2 className="font-semibold text-sm mb-1">Project Snapshot</h2>
      <ul className="space-y-1.5">
        <li>
          <span className="font-medium">Task: </span>
          Multi-class chest X-ray classification (COVID, Normal, Pneumonia, TB).
        </li>
        <li>
          <span className="font-medium">Backbone: </span>
          MobileNetV2 + DenseNet121 ensemble, fine-tuned with heavy
          augmentation and class balancing.
        </li>
        <li>
          <span className="font-medium">Metrics (Test): </span>
          ~78% accuracy, ~0.80 precision, ~0.77 recall, ~0.88 PR-AUC.
        </li>
        <li>
          <span className="font-medium">Explainability: </span>
          Grad-CAM heatmaps over the lungs to highlight suspicious regions.
        </li>
      </ul>
      {!compact && (
        <p className="text-[11px] text-slate-500 dark:text-slate-500">
          This interface demonstrates how an AI model can assist radiologists by
          triaging cases and visually explaining its focus regions.
        </p>
      )}
    </div>
  );
}

export default InfoCards;
