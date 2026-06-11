export default function LoadingView({ phase, elapsed }) {
  const isPhase2 = phase === "generating_report";

  return (
    <div className="flex flex-col items-center text-center">
      <div className="animate-spin border-4 border-blue-500 border-t-transparent rounded-full w-12 h-12 mb-4" />
      <p className="text-white font-bold mb-2">
        {isPhase2 ? "Writing FTO Report..." : "Analyzing patents..."}
      </p>
      <p className="text-slate-400 mb-4">Time elapsed: {elapsed}s</p>
      <p className="text-slate-500 text-sm italic mb-6">
        {isPhase2
          ? "The AI analyst is drafting the formal FTO report. This typically takes 15–30 seconds."
          : "EPO patent search and LLM analysis typically takes 120–150 seconds. For niche topics, the system may conclude early with no results found."}
      </p>
      <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
        <div
          className="bg-blue-500 h-2 transition-all duration-1000"
          style={{ width: `${Math.min((elapsed / (isPhase2 ? 30 : 90)) * 100, 95)}%` }}
        />
      </div>
    </div>
  );
}
