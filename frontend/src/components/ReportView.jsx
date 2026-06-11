import ReactMarkdown from "react-markdown";
import { riskBorderClass, riskBadgeClass } from "../utils.js";

export default function ReportView({ result, onDownload, onReset }) {
  const significant = result.risk_assessments ?? [];
  const cleared     = result.cleared_patents   ?? [];

  const highCount   = significant.filter(a => a.risk_level?.toUpperCase() === "HIGH").length;
  const mediumCount = significant.filter(a => a.risk_level?.toUpperCase() === "MEDIUM").length;
  const lowCount    = significant.filter(a => a.risk_level?.toUpperCase() === "LOW").length;

  return (
    <>
      <p className="text-slate-400 italic mb-6">&ldquo;{result.idea}&rdquo;</p>

      <div className="grid grid-cols-3 gap-3 mb-6">
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

      <h2 className="text-white font-bold text-lg mb-4">Patent Risk Assessments</h2>

      {significant.length === 0 ? (
        <p className="text-slate-400 text-sm mb-4 italic">
          No patents with significant technical overlap (score &gt; 0.2) were identified.
        </p>
      ) : (
        significant.map((assessment) => (
          <div
            key={assessment.patent_id}
            className={`border-l-4 rounded-lg bg-slate-800 p-4 mb-3 ${riskBorderClass(assessment.risk_level)}`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-white font-mono text-sm">{assessment.patent_id}</span>
              <span className={`px-2 py-1 rounded-full text-xs font-bold ${riskBadgeClass(assessment.risk_level)}`}>
                {assessment.risk_level}
              </span>
            </div>
            <p className="text-slate-400 text-sm mb-1">
              Overlap score: {assessment.overlap_score}
            </p>
            <p className="text-slate-300 text-sm">{assessment.reasoning}</p>
          </div>
        ))
      )}

      <h2 className="text-white font-bold text-lg mt-8 mb-4">FTO Report</h2>
      <div className="bg-slate-800 rounded-lg p-6 prose prose-invert prose-sm max-w-none overflow-auto max-h-96">
        <ReactMarkdown>{result.report}</ReactMarkdown>
      </div>

      {cleared.length > 0 && (
        <div className="mt-8">
          <h2 className="text-white font-bold text-lg mb-1">
            Patents Examined — No Significant Overlap
          </h2>
          <p className="text-slate-500 text-xs mb-4">
            These {cleared.length} patent{cleared.length !== 1 ? "s" : ""} were assessed
            but scored ≤ 0.2 overlap. They pose no identified risk and are listed here
            for audit purposes only.
          </p>
          <div className="rounded-lg border border-slate-700 overflow-hidden">
            {cleared.map((p, idx) => (
              <div
                key={p.patent_id}
                className={`flex items-center justify-between px-4 py-2 text-sm ${
                  idx % 2 === 0 ? "bg-slate-800" : "bg-slate-800/60"
                }`}
              >
                <span className="font-mono text-slate-300">{p.patent_id}</span>
                <div className="flex items-center gap-4 text-slate-500 text-xs">
                  <span>overlap: {p.overlap_score}</span>
                  <span className="uppercase">{p.risk_level}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-3 mt-6">
        <button
          type="button"
          onClick={onDownload}
          className="bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-4 py-2"
        >
          Download Report
        </button>
        <button
          type="button"
          onClick={onReset}
          className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2"
        >
          Analyze New Idea
        </button>
      </div>
    </>
  );
}
