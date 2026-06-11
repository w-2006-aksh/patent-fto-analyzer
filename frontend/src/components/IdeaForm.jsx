export default function IdeaForm({ idea, setIdea, onSubmit }) {
  return (
    <>
      <h1 className="text-white font-bold text-3xl mb-2">Patent FTO Analyzer</h1>
      <p className="text-slate-400 text-sm mb-6">
        Check if your invention risks infringing existing patents
      </p>
      <form onSubmit={onSubmit}>
        <label className="block text-slate-300 text-sm font-medium mb-2">
          Invention Idea
        </label>
        <textarea
          rows={4}
          value={idea}
          onChange={(e) => setIdea(e.target.value)}
          placeholder="Describe your invention in detail..."
          className="w-full rounded-lg bg-slate-700 text-white border border-slate-600 p-3 mb-4 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={idea.trim() === ""}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg py-3 font-semibold"
        >
          Run FTO Analysis →
        </button>
      </form>
    </>
  );
}
