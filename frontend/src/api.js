async function _post(path, body) {
  const response = await fetch(path, {
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
  return _post("/api/start_analysis", { idea });
}

export function approveAnalysis(thread_id, approved) {
  return _post("/api/approve_analysis", { thread_id, approved });
}
