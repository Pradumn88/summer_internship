import React from "react";

function HistoryPanel({ history, darkMode, onSelect, compact }) {
  const cardStyle = darkMode
    ? "bg-slate-900 border-slate-700"
    : "bg-white border-slate-200";

  return (
    <div
      className={`rounded-2xl border p-4 shadow-sm ${
        compact ? "text-xs" : "text-sm"
      } ${cardStyle}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold text-sm">
          {compact ? "Recent Predictions" : "Prediction History"}
        </h2>
        <span className="text-[11px] text-slate-500 dark:text-slate-500">
          {history.length} case{history.length !== 1 ? "s" : ""}
        </span>
      </div>

      {history.length === 0 ? (
        <p className="text-[11px] text-slate-500 dark:text-slate-500">
          No predictions yet.
        </p>
      ) : (
        <ul className="space-y-1.5 max-h-64 overflow-y-auto">
          {history.map((item) => {
            const created = item.createdAt
              ? new Date(item.createdAt)
              : new Date();
            return (
              <li
                key={item.id}
                onClick={() => onSelect && onSelect(item)}
                className={`flex items-center justify-between gap-2 rounded-xl px-2 py-1 cursor-pointer ${
                  darkMode
                    ? "hover:bg-slate-800"
                    : "hover:bg-slate-100 border border-transparent hover:border-slate-300"
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1">
                    <span className="font-medium text-[11px] truncate">
                      {item.prediction}
                    </span>
                    <span className="text-[10px] text-slate-500 truncate">
                      {item.fileName}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {created.toLocaleString()}
                  </div>
                </div>
                <span className="text-[11px] text-slate-500">
                  {(item.confidence * 100).toFixed(0)}%
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export default HistoryPanel;
