import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ResponsiveContainer,
} from 'recharts'
import StatCard from '../components/StatCard.jsx'

const PIPELINE_STEPS = [
  { color: '#6366F1', label: '1. Données brutes',      desc: 'MovieLens ml-latest-small — 100 836 notes, 9 742 films, 610 utilisateurs' },
  { color: '#06B6D4', label: '2. Filtrage interim',     desc: 'Filtre MIN_RATINGS=5 → 90 274 notes, 3 650 films. Extraction année + genres.' },
  { color: '#10B981', label: '3. Feature engineering',  desc: 'Matrice user-item 610×3650 (sparsité 95,95%). User features, item features, one-hot genres.' },
  { color: '#F59E0B', label: '4. Entraînement + MLflow',desc: '4 modèles. Cross-validation 5-fold. Tracking params/métriques/artéfacts (SQLite).' },
  { color: '#F43F5E', label: '5. Recommandations Top-N',desc: 'SVD prédit les notes pour tous les films non vus. Tri décroissant → Top-10 personnalisé.' },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#0D1229', border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 8, padding: '8px 12px', fontSize: '0.8rem',
    }}>
      <div style={{ color: '#94A3B8', marginBottom: 4 }}>{label}</div>
      <div style={{ color: payload[0].fill, fontWeight: 600 }}>
        RMSE : {payload[0].value}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [stats,  setStats]  = useState(null)
  const [models, setModels] = useState([])
  const [err,    setErr]    = useState(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/models').then(r => r.json()),
    ])
      .then(([s, m]) => { setStats(s); setModels(m) })
      .catch(() => setErr('Impossible de joindre l\'API. Lancez : uvicorn interface.api.main:app --reload --port 8000'))
  }, [])

  const chartData = models.map(m => ({
    name: m.name, rmse: m.rmse, color: m.color,
  }))

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Vue d'ensemble</h1>
        <p className="page-sub">
          MovieLens ml-latest-small · SVD + KNN · MLflow tracking · SEED=42
        </p>
      </header>

      {err && <div className="banner banner-error" role="alert">{err}</div>}

      {/* Stat cards */}
      <section aria-label="Statistiques du dataset">
        <div className="stats-grid">
          <StatCard delay={40}  icon="⭐" value={stats ? stats.n_ratings.toLocaleString('fr-FR') : '—'} label="Notes filtrées (≥ 5 par film)"  color="#6366F1" />
          <StatCard delay={80}  icon="👤" value={stats ? stats.n_users  : '—'}                          label="Utilisateurs actifs"              color="#10B981" />
          <StatCard delay={120} icon="🎬" value={stats ? stats.n_items  : '—'}                          label="Films avec au moins 5 notes"       color="#F59E0B" />
          <StatCard delay={160} icon="◎"  value={stats ? `${stats.density_pct} %` : '—'}                label="Densité de la matrice"             color="#F43F5E" />
        </div>
      </section>

      {/* Chart + Pipeline */}
      <div className="panel-grid">

        {/* RMSE chart */}
        <section className="panel" aria-labelledby="perf-title">
          <h2 className="panel-title" id="perf-title">Performance des modèles — RMSE test</h2>

          {models.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 50, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 1.6]}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: '#94A3B8', fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  width={95}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="rmse" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ color: 'var(--text-3)', fontSize: '0.82rem', padding: '20px 0' }}>
              Chargement des données…
            </div>
          )}

          <p style={{ fontSize: '0.74rem', color: 'var(--text-3)', marginTop: 14, lineHeight: 1.5 }}>
            SVD atteint <strong style={{ color: 'var(--text-2)' }}>RMSE 0,853</strong> — cohérent avec la littérature (0,85–0,93 sur MovieLens ml-latest-small). Plus le RMSE est bas, plus la prédiction est précise.
          </p>
        </section>

        {/* Pipeline */}
        <section className="panel" aria-labelledby="pipeline-title">
          <h2 className="panel-title" id="pipeline-title">Pipeline MLOps</h2>
          <div className="pipeline" role="list">
            {PIPELINE_STEPS.map((step, i) => (
              <div key={i} className="pipe-step" role="listitem">
                <div className="pipe-left">
                  <div className="pipe-dot" style={{ background: step.color }} />
                  {i < PIPELINE_STEPS.length - 1 && <div className="pipe-line" />}
                </div>
                <div>
                  <div className="pipe-label">{step.label}</div>
                  <div className="pipe-desc">{step.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

      </div>
    </div>
  )
}
