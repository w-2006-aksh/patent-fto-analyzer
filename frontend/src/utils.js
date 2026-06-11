export function riskBorderClass(level) {
  switch (level?.toUpperCase()) {
    case "HIGH":   return "border-red-500";
    case "MEDIUM": return "border-yellow-500";
    case "LOW":    return "border-green-500";
    default:       return "border-slate-500";
  }
}

export function riskBadgeClass(level) {
  switch (level?.toUpperCase()) {
    case "HIGH":   return "bg-red-500/20 text-red-400";
    case "MEDIUM": return "bg-yellow-500/20 text-yellow-400";
    case "LOW":    return "bg-green-500/20 text-green-400";
    default:       return "bg-slate-500/20 text-slate-400";
  }
}
