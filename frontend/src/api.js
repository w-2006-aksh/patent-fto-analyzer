const API_BASE = import.meta.env.VITE_API_URL || "";

function apiPath(path) {
  if (API_BASE) {
    return `${API_BASE.replace(/\/$/, "")}${path}`;
  }
  return `/api${path}`;
}

async function _post(path, body) {
  const response = await fetch(apiPath(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }
  return response.json();
}

export function startAnalysis(idea) {
  return _post("/start_analysis", { idea });
}

export function approveAnalysis(thread_id, approved) {
  return _post("/approve_analysis", { thread_id, approved });
}
