export default function StatCard({ icon, value, label, color, delay = 0 }) {
  return (
    <article
      className="stat-card"
      style={{ animationDelay: `${delay}ms` }}
      aria-label={`${label} : ${value}`}
    >
      <div className="stat-icon" style={{ background: `${color}22` }} aria-hidden="true">
        {icon}
      </div>
      <div className="stat-value" style={{ color }}>{value}</div>
      <div className="stat-label">{label}</div>
    </article>
  )
}
