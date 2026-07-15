import { riskBorderClass, riskBadgeClass } from "../utils.js";

export default function ReviewPanel({ idea, assessments, onApprove }) {
  const highCount   = assessments.filter(a => a.risk_level?.toUpperCase() === "HIGH").length;
  const mediumCount = assessments.filter(a => a.risk_level?.toUpperCase() === "MEDIUM").length;
  const lowCount    = assessments.filter(a => a.risk_level?.toUpperCase() === "LOW").length;

  return (
    <>
      <h2 className="text-white font-bold text-xl mb-1">Phase 1 Complete — Review Findings</h2>
      <p className="text-slate-400 italic text-sm mb-4">&ldquo;{idea}&rdquo;</p>

      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 mb-5 text-sm text-blue-300">
      <strong>{assessments.length} patent{assessments.length !== 1 ? "s" : ""} assessed.</strong>{" "}
      Review the findings below. Click approve to summarize all available patents and generate a downloadable Markdown report.
      </div>

      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="bg-slate-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-red-400">{highCount}</div>
          <div className="text-xs text-slate-400 mt-1">High Risk</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-yellow-400">{mediumCount}</div>
          <div className="text-xs text-slate-400 mt-1">Medium Risk</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-400">{lowCount}</div>
          <div className="text-xs text-slate-400 mt-1">Low Risk</div>
        </div>
      </div>

      <div className="max-h-64 overflow-y-auto pr-1 mb-5 space-y-3">
        {assessments.length === 0 ? (
          <p className="text-slate-400 text-sm italic">No patents were assessed.</p>
        ) : (
          assessments.map((a) => (
            <div
              key={a.patent_id}
              className={`border-l-4 rounded-lg bg-slate-800/80 p-4 ${riskBorderClass(a.risk_level)}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-white font-mono text-sm">{a.patent_id}</span>
                <span className={`px-2 py-1 rounded-full text-xs font-bold ${riskBadgeClass(a.risk_level)}`}>
                  {a.risk_level}
                </span>
              </div>
              <p className="text-slate-400 text-xs mb-1">Overlap: {a.overlap_score}</p>
              <p className="text-slate-300 text-xs leading-relaxed">{a.reasoning}</p>
            </div>
          ))
        )}
      </div>

      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => onApprove(true)}
          className="flex-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-3 font-semibold"
        >
          Generate Full FTO Report →
        </button>
        <button
          type="button"
          onClick={() => onApprove(false)}
          className="bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-5 py-3"
        >
          Cancel
        </button>
      </div>
    </>
  );
}
