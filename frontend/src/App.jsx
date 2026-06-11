import { useState, useEffect, useRef } from "react";
import { startAnalysis, approveAnalysis } from "./api.js";
import IdeaForm    from "./components/IdeaForm.jsx";
import LoadingView from "./components/LoadingView.jsx";
import ReviewPanel from "./components/ReviewPanel.jsx";
import ReportView  from "./components/ReportView.jsx";

function App() {
  // phase: idle - loading - reviewing - generating_report - result - error
  const [phase, setPhase]                   = useState("idle");
  const [idea, setIdea]                     = useState("");
  const [threadId, setThreadId]             = useState(null);
  const [phase1Assessments, setPhase1Assessments] = useState([]);
  const [result, setResult]                 = useState(null);
  const [error, setError]                   = useState("");
  const [elapsed, setElapsed]               = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (phase === "loading" || phase === "generating_report") {
      intervalRef.current = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [phase]);

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = idea.trim();
    if (!trimmed) return;

    setPhase("loading");
    setElapsed(0);
    setError("");

    try {
      const data = await startAnalysis(trimmed);

      if (data.complete) {
        setResult(data);
        setPhase("result");
      } else {
        setThreadId(data.thread_id);
        setPhase1Assessments(data.risk_assessments ?? []);
        setPhase("reviewing");
      }
    } catch (err) {
      setError(err.message);
      setPhase("error");
    }
  }

  async function handleApprove(approved) {
    if (!approved) {
      handleReset();
      return;
    }

    setPhase("generating_report");
    setElapsed(0);

    try {
      const data = await approveAnalysis(threadId, true);
      setResult(data);
      setPhase("result");
    } catch (err) {
      setError(err.message);
      setPhase("error");
    }
  }

  function handleReset() {
    setPhase("idle");
    setResult(null);
    setIdea("");
    setError("");
    setElapsed(0);
    setThreadId(null);
    setPhase1Assessments([]);
  }

  function handleDownloadReport() {
    const blob   = new Blob([result.report], { type: "text/markdown" });
    const url    = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href     = url;
    anchor.download = "fto_report.md";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl bg-slate-800 rounded-xl shadow-2xl p-8 border-t-4 border-blue-500">

        {phase === "idle" && (
          <IdeaForm idea={idea} setIdea={setIdea} onSubmit={handleSubmit} />
        )}

        {(phase === "loading" || phase === "generating_report") && (
          <LoadingView phase={phase} elapsed={elapsed} />
        )}

        {phase === "reviewing" && (
          <ReviewPanel
            idea={idea}
            assessments={phase1Assessments}
            onApprove={handleApprove}
          />
        )}

        {phase === "result" && (
          <ReportView
            result={result}
            onDownload={handleDownloadReport}
            onReset={handleReset}
          />
        )}

        {phase === "error" && (
          <>
            <div className="bg-red-500/10 border border-red-500 rounded-lg p-4">
              <p className="text-red-500 font-bold mb-2">Analysis Failed</p>
              <p className="text-red-400/80">{error}</p>
            </div>
            <button
              type="button"
              onClick={handleReset}
              className="bg-red-600 hover:bg-red-700 text-white rounded-lg px-4 py-2 mt-4"
            >
              Try Again
            </button>
          </>
        )}

      </div>
    </div>
  );
}

export default App;
