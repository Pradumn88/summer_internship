import React from "react";

function ResultDisplay({ prediction, preview, darkMode, classNames }) {
  if (!prediction) {
    return (
      <div
        className={`mt-4 rounded-2xl border p-4 text-sm ${
          darkMode
            ? "border-slate-800 bg-slate-900 text-slate-300"
            : "border-slate-200 bg-white text-slate-700"
        }`}
      >
        Upload an X-ray and run detection to see results here.
      </div>
    );
  }

  const { prediction: label, confidence, probabilities, gradcam } = prediction;
  const confPct = (confidence * 100).toFixed(1);

  const isAlert = label === "COVID" || label === "TB" || label === "PNEUMONIA";

  const cardClasses = isAlert
    ? darkMode
      ? "from-rose-900/40 to-amber-900/30 border-rose-700/70"
      : "from-rose-50 to-amber-50 border-rose-300"
    : darkMode
    ? "from-emerald-900/30 to-sky-900/20 border-emerald-700/60"
    : "from-emerald-50 to-sky-50 border-emerald-300";

  return (
    <div
      className={`mt-4 rounded-2xl border bg-gradient-to-br p-4 sm:p-5 shadow-md ${cardClasses}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-semibold">Prediction Result</h2>
          <p className="text-xs text-slate-600 dark:text-slate-400">
            Model: MobileNetV2 + DenseNet121 (ensemble), trained on 4 classes
          </p>
        </div>
        <span className="inline-flex items-center text-xs px-2 py-1 rounded-full bg-slate-900/80 text-slate-100">
          {confPct}% confidence
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Images */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
            X-ray & Grad-CAM
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-900/5 flex items-center justify-center">
              {preview ? (
                <img
                  src={preview}
                  alt="Original X-ray"
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="text-[11px] text-slate-500 p-2 text-center">
                  Original preview not available
                </span>
              )}
            </div>
            <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-900/5 flex items-center justify-center">
              {gradcam ? (
                <img
                  src={`data:image/png;base64,${gradcam}`}
                  alt="Grad-CAM heatmap"
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="text-[11px] text-slate-500 p-2 text-center">
                  Grad-CAM not generated
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Text + per-class probs */}
        <div className="space-y-3">
          <div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              Predicted Class
            </div>
            <div className="mt-1 inline-flex items-center gap-2">
              <span className="text-xl font-bold">{label}</span>
              {isAlert && (
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-red-500/10 text-red-600 dark:text-red-300 border border-red-500/40">
                  Flagged — requires clinical review
                </span>
              )}
            </div>
          </div>

          <div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
              Per-class probability
            </div>
            <div className="space-y-1.5">
              {classNames.map((cls, i) => {
                const p = probabilities?.[i] ?? 0;
                const pct = (p * 100).toFixed(1);
                return (
                  <div key={cls} className="flex items-center gap-2">
                    <div className="w-20 text-[11px] font-medium">{cls}</div>
                    <div className="flex-1 h-2.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          cls === label
                            ? "bg-indigo-500"
                            : "bg-slate-400 dark:bg-slate-500"
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="w-10 text-[11px] text-right">{pct}%</div>
                  </div>
                );
              })}
            </div>
          </div>

          <p className="text-[11px] text-slate-500 dark:text-slate-500 leading-snug">
            This is an AI assistance system trained on retrospective chest X-ray
            datasets. It <span className="font-semibold">does not replace</span>{" "}
            a radiologist. All outputs must be interpreted by qualified medical
            professionals.
          </p>
        </div>
      </div>
    </div>
  );
}

export default ResultDisplay;
