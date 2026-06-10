import { useCallback, useEffect, useRef, useState } from 'react'
import FilmCard from '../components/FilmCard.jsx'

// ── Genre chip (multi-select) ─────────────────────────────────────────────────
function GenreChip({ label, selected, onClick }) {
  return (
    <button
      className={`genre-chip-btn${selected ? ' selected' : ''}`}
      onClick={onClick}
      aria-pressed={selected}
    >
      {label}
    </button>
  )
}

// ── Star rating widget ────────────────────────────────────────────────────────
function StarRating({ value, onChange }) {
  const [hover, setHover] = useState(0)
  return (
    <div className="liked-stars" aria-label={`Note : ${value} sur 5`}>
      {[1, 2, 3, 4, 5].map(s => (
        <button
          key={s}
          className={`star-btn${s <= (hover || value) ? ' active' : ' inactive'}`}
          onClick={() => onChange(s)}
          onMouseEnter={() => setHover(s)}
          onMouseLeave={() => setHover(0)}
          aria-label={`${s} étoile${s > 1 ? 's' : ''}`}
        >★</button>
      ))}
    </div>
  )
}

// ── Movie search with dropdown ────────────────────────────────────────────────
function MovieSearch({ onAdd, addedIds }) {
  const [query,   setQuery]   = useState('')
  const [results, setResults] = useState([])
  const [open,    setOpen]    = useState(false)
  const debRef  = useRef(null)
  const wrapRef = useRef(null)

  useEffect(() => {
    clearTimeout(debRef.current)
    if (!query.trim()) { setResults([]); setOpen(false); return }
    debRef.current = setTimeout(async () => {
      try {
        const r = await fetch(`/api/movies/search?q=${encodeURIComponent(query)}&limit=8`)
        const d = await r.json()
        setResults(d)
        setOpen(d.length > 0)
      } catch { setResults([]); setOpen(false) }
    }, 280)
  }, [query])

  useEffect(() => {
    const handler = (e) => { if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div className="search-wrap" ref={wrapRef}>
      <input
        className="input-field"
        type="text"
        placeholder="Rechercher un film…"
        value={query}
        onChange={e => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        aria-label="Rechercher un film par titre"
        aria-autocomplete="list"
        aria-expanded={open}
      />
      {open && (
        <div className="search-dropdown" role="listbox">
          {results.map(m => {
            const already = addedIds.has(m.movie_id)
            return (
              <button
                key={m.movie_id}
                className={`search-item${already ? ' added' : ''}`}
                role="option"
                aria-selected={already}
                onClick={() => {
                  if (!already) {
                    onAdd(m)
                    setQuery('')
                    setOpen(false)
                  }
                }}
              >
                <span className="search-title">{m.title}</span>
                {m.year && <span className="search-year">{m.year}</span>}
                {m.genres_list.length > 0 && (
                  <span className="search-genres">{m.genres_list.slice(0, 2).join(' · ')}</span>
                )}
                {already && <span className="search-added-badge">ajouté</span>}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Méthodes de recommandation profil (statiques — disponibilité vérifiée côté API) ──
const PROFILE_METHODS = [
  { id: 'knn_item',    name: 'KNN Similarité', desc: 'Similarité cosine item-item', color: '#F97316', badge: 'KNN'  },
  { id: 'svd_foldin',  name: 'SVD Fold-In',    desc: 'Espace latent SVD',           color: '#6366F1', badge: 'ML'   },
  { id: 'genre_match', name: 'Genre Match',     desc: 'Overlap de genres pondéré',  color: '#10B981', badge: 'FAST' },
  { id: 'popularity',  name: 'Popularité',      desc: 'Films les plus populaires',  color: '#64748B', badge: null   },
]

function MethodCard({ m, selected, onSelect }) {
  return (
    <button
      className={`method-card${selected ? ' selected' : ''}`}
      onClick={() => onSelect(m.id)}
      aria-pressed={selected}
      title={m.desc}
    >
      <div className="model-dot" style={{ background: m.color }} />
      <span className="model-name-sm">{m.name}</span>
      {m.badge && (
        <span className="method-badge" style={{ background: `${m.color}22`, color: m.color }}>
          {m.badge}
        </span>
      )}
    </button>
  )
}

// ── Profile mode (free user, no dataset ID) ───────────────────────────────────
function ProfileMode() {
  const [genres,      setGenres]      = useState([])
  const [allGenres,   setAllGenres]   = useState([])
  const [liked,       setLiked]       = useState([])   // [{movie_id, title, year, genres_list, rating}]
  const [method,      setMethod]      = useState('knn_item')
  const [loading,     setLoading]     = useState(false)
  const [results,     setResults]     = useState(null)
  const [error,       setError]       = useState(null)

  useEffect(() => {
    fetch('/api/genres').then(r => r.json()).then(setAllGenres).catch(() => {})
  }, [])

  const addedIds = new Set(liked.map(m => m.movie_id))

  const handleAddFilm = useCallback((film) => {
    setLiked(prev => [...prev, { ...film, rating: 4 }])
  }, [])

  const handleRate = useCallback((movie_id, rating) => {
    setLiked(prev => prev.map(m => m.movie_id === movie_id ? { ...m, rating } : m))
  }, [])

  const handleRemove = useCallback((movie_id) => {
    setLiked(prev => prev.filter(m => m.movie_id !== movie_id))
  }, [])

  const toggleGenre = useCallback((g) => {
    setGenres(prev => prev.includes(g) ? prev.filter(x => x !== g) : [...prev, g])
  }, [])

  const handleRecommend = useCallback(async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const resp = await fetch('/api/recommend-profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          liked_movies: liked.map(m => ({ movie_id: m.movie_id, rating: m.rating })),
          genres,
          method,
          n: 10,
        }),
      })
      if (!resp.ok) {
        const d = await resp.json()
        throw new Error(d.detail ?? `Erreur ${resp.status}`)
      }
      setResults(await resp.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [liked, genres])

  const canRecommend = !loading && (liked.length > 0 || genres.length > 0)
  const selectedMethod = PROFILE_METHODS.find(m => m.id === method)

  return (
    <div className="predict-layout">

      {/* ── Sidebar ── */}
      <aside className="predict-sidebar">

        {/* Step 1 — Films */}
        <div>
          <div className="step-label">
            <span className="step-num">1</span>
            Films que vous aimez
          </div>
          <MovieSearch onAdd={handleAddFilm} addedIds={addedIds} />

          {liked.length > 0 ? (
            <div className="liked-list" aria-label="Films notés">
              {liked.map(m => (
                <div key={m.movie_id} className="liked-item">
                  <span className="liked-title" title={m.title}>{m.title}</span>
                  <StarRating value={m.rating} onChange={(r) => handleRate(m.movie_id, r)} />
                  <button className="liked-remove" onClick={() => handleRemove(m.movie_id)} aria-label={`Retirer ${m.title}`}>✕</button>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', marginTop: 8, lineHeight: 1.4 }}>
              Recherchez des films que vous avez déjà vus et aimés. Le système utilisera leurs similarités pour vous recommander.
            </p>
          )}
        </div>

        {/* Step 2 — Genres */}
        <div>
          <div className="step-label">
            <span className="step-num">2</span>
            Genres préférés <span style={{ fontSize: '0.65rem', color: 'var(--text-3)', textTransform: 'none', letterSpacing: 0 }}>(optionnel)</span>
          </div>
          <div className="genre-chips-select" role="group" aria-label="Filtrer par genre">
            {allGenres.map(g => (
              <GenreChip key={g} label={g} selected={genres.includes(g)} onClick={() => toggleGenre(g)} />
            ))}
          </div>
        </div>

        {/* Step 3 — Méthode */}
        <div>
          <div className="step-label">
            <span className="step-num">3</span>
            Méthode de recommandation
          </div>
          <div className="method-cards" role="group" aria-label="Choisir la méthode">
            {PROFILE_METHODS.map(m => (
              <MethodCard key={m.id} m={m} selected={method === m.id} onSelect={setMethod} />
            ))}
          </div>
          {selectedMethod && (
            <p style={{ fontSize: '0.7rem', color: 'var(--text-3)', marginTop: 6, lineHeight: 1.4 }}>
              {selectedMethod.desc}
            </p>
          )}
        </div>

        {/* Step 4 — Recommander */}
        <div>
          <div className="step-label">
            <span className="step-num">4</span>
            Recommander
          </div>
          <button className="btn-predict" onClick={handleRecommend} disabled={!canRecommend} aria-busy={loading}>
            {loading ? (
              <><div className="spinner" aria-hidden="true" /> Calcul en cours…</>
            ) : (
              <><svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><path d="M2.5 1.5l10 5.5-10 5.5V1.5z" fill="currentColor"/></svg> Obtenir mes recommandations</>
            )}
          </button>
          {!canRecommend && !loading && (
            <p style={{ fontSize: '0.7rem', color: 'var(--text-3)', marginTop: 6 }}>
              Ajoutez au moins un film ou sélectionnez un genre.
            </p>
          )}
        </div>

      </aside>

      {/* ── Results ── */}
      <section aria-live="polite" aria-label="Recommandations">
        {error && <div className="banner banner-error" role="alert">{error}</div>}

        {!loading && !results && !error && (
          <div className="empty-state">
            <div className="empty-icon" aria-hidden="true">🎬</div>
            <div className="empty-title">Construisez votre profil</div>
            <div className="empty-desc">
              Ajoutez des films que vous avez vus (et notez-les), filtrez par genre si vous voulez, puis lancez la recommandation. Aucun compte requis.
            </div>
          </div>
        )}

        {loading && (
          <div className="films-grid" aria-label="Chargement">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="skeleton-card" style={{ animationDelay: `${i * 50}ms` }} />
            ))}
          </div>
        )}

        {results && !loading && (
          <>
            <div className="results-header">
              <h2 className="section-title" style={{ margin: 0 }}>
                {results.method === 'popularity' || results.method === 'popularity_fallback'
                  ? 'Films populaires'
                  : 'Recommandations personnalisées'}
                {(() => {
                  const m = PROFILE_METHODS.find(x => results.method.startsWith(x.id))
                  return m
                    ? <span className="badge" style={{ background: `${m.color}22`, color: m.color, border: `1px solid ${m.color}44` }}>{m.name}</span>
                    : <span className="badge badge-indigo">Popularité</span>
                })()}
                {results.genres_filter.length > 0 && (
                  <span className="badge badge-amber">{results.genres_filter.join(', ')}</span>
                )}
              </h2>
              <div className="results-meta">
                {results.n_liked > 0 && <span><strong>{results.n_liked}</strong> film{results.n_liked > 1 ? 's' : ''} de référence</span>}
                <span><strong>{results.recommendations.length}</strong> recommandations</span>
              </div>
            </div>

            {results.method === 'popularity_fallback' && (
              <div className="banner banner-info" style={{ marginBottom: 14 }}>
                Aucun film commun trouvé — affichage des films les plus populaires en fallback.
              </div>
            )}

            <div className="films-grid">
              {results.recommendations.map((film, i) => (
                <ProfileFilmCard key={film.movie_id} film={film} delay={i * 40} />
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  )
}

// ── Card pour les résultats profil (score de similarité) ─────────────────────
const GENRE_CLASS = {
  'Action':'gc-action','Crime':'gc-crime','Comedy':'gc-comedy','Romance':'gc-romance',
  'Sci-Fi':'gc-scifi','Thriller':'gc-thriller','Drama':'gc-drama','War':'gc-war',
  'Adventure':'gc-adventure','Animation':'gc-animation','Mystery':'gc-mystery',
}

function ProfileFilmCard({ film, delay = 0 }) {
  const genres = Array.isArray(film.genres_list) ? film.genres_list.slice(0, 3) : []
  const pct = film.score ?? 0

  return (
    <article className="film-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="film-rank">#{String(film.rank).padStart(2, '0')}</div>
      <div>
        <h3 className="film-title">{film.title}</h3>
        {film.year && <div className="film-year">{film.year}</div>}
      </div>
      {genres.length > 0 && (
        <div className="film-genres">
          {genres.map(g => <span key={g} className={`genre-chip ${GENRE_CLASS[g] ?? 'gc-default'}`}>{g}</span>)}
        </div>
      )}
      <div className="film-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{
            width: 40, height: 40, borderRadius: '50%',
            background: `conic-gradient(var(--cyan) ${pct * 3.6}deg, rgba(255,255,255,0.08) 0deg)`,
            display: 'grid', placeItems: 'center',
            flexShrink: 0,
          }} aria-label={`Score ${pct}%`}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%',
              background: 'var(--bg-2)',
              display: 'grid', placeItems: 'center',
              fontSize: '0.58rem', fontWeight: 700, color: 'var(--cyan)',
            }}>{pct}%</div>
          </div>
        </div>
        <span className="rating-lbl">score</span>
      </div>
    </article>
  )
}

// ── Dataset mode (existing users) ────────────────────────────────────────────
function ModelCardCompact({ model, selected, onSelect }) {
  return (
    <button
      className={`model-card-compact${selected ? ' selected' : ''}${!model.available ? ' disabled' : ''}`}
      onClick={() => model.available && onSelect(model.id)}
      aria-pressed={selected}
      aria-disabled={!model.available}
      title={model.desc}
    >
      <div className="model-dot" style={{ background: model.color }} />
      <span className="model-name-sm">{model.name}</span>
      {model.is_best && <span className="best-chip">BEST</span>}
      <span className="model-rmse-sm">{model.rmse}</span>
    </button>
  )
}

function ModelInfoBanner({ model }) {
  if (!model) return null
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
      background: `${model.color}14`, border: `1px solid ${model.color}30`,
      borderRadius: 8, marginBottom: 16, fontSize: '0.8rem',
    }}>
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: model.color, flexShrink: 0 }} />
      <div>
        <strong style={{ color: 'var(--text-1)' }}>{model.name}</strong>
        <span style={{ color: 'var(--text-3)', marginLeft: 8 }}>{model.desc}</span>
      </div>
      <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, flexShrink: 0 }}>
        <span style={{ color: 'var(--text-3)' }}>RMSE <strong style={{ color: model.color }}>{model.rmse}</strong></span>
        {model.is_best && (
          <span style={{ background: 'rgba(16,185,129,0.15)', color: '#6EE7B7', padding: '2px 8px', borderRadius: 20, fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Meilleur modèle
          </span>
        )}
      </div>
    </div>
  )
}

function DatasetMode() {
  const [models,      setModels]      = useState([])
  const [modelId,     setModelId]     = useState('svd')
  const [userId,      setUserId]      = useState('')
  const [userProfile, setUserProfile] = useState(null)
  const [userError,   setUserError]   = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [results,     setResults]     = useState(null)
  const [error,       setError]       = useState(null)
  const debounceRef = useRef(null)

  useEffect(() => {
    fetch('/api/models').then(r => r.json()).then(setModels).catch(() => {})
  }, [])

  useEffect(() => {
    clearTimeout(debounceRef.current)
    setUserProfile(null); setUserError(null)
    const id = parseInt(userId, 10)
    if (!userId || isNaN(id) || id < 1 || id > 610) return
    debounceRef.current = setTimeout(async () => {
      try {
        const r = await fetch(`/api/users?q=${id}&limit=1`)
        const d = await r.json()
        if (d.users?.length > 0) setUserProfile(d.users[0])
        else setUserError(`Utilisateur ${id} introuvable.`)
      } catch { setUserError('API inaccessible.') }
    }, 400)
  }, [userId])

  const handleRandom = () => setUserId(String(Math.floor(Math.random() * 610) + 1))

  const handlePredict = useCallback(async () => {
    const id = parseInt(userId, 10)
    if (!id || !modelId) return
    setLoading(true); setError(null); setResults(null)
    try {
      const resp = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: id, model_id: modelId, n: 10 }),
      })
      if (!resp.ok) { const d = await resp.json(); throw new Error(d.detail ?? `Erreur ${resp.status}`) }
      setResults(await resp.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [userId, modelId])

  const selectedModel = models.find(m => m.id === modelId)
  const canPredict    = userProfile && modelId && !loading

  return (
    <div className="predict-layout">
      <aside className="predict-sidebar">
        <div>
          <div className="step-label"><span className="step-num">1</span>Utilisateur (1 – 610)</div>
          <div className="user-input-row">
            <input className="input-field" type="number" min={1} max={610} placeholder="ID utilisateur"
              value={userId} onChange={e => setUserId(e.target.value)} aria-label="Identifiant utilisateur" />
            <button className="btn-icon" onClick={handleRandom} title="Aléatoire" aria-label="Choisir un utilisateur aléatoire">
              <svg width="15" height="15" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd"/>
              </svg>
            </button>
          </div>
          {userProfile && (
            <div className="user-profile-card" style={{ marginTop: 8 }}>
              <div className="user-profile-name">Utilisateur #{userProfile.user_id}</div>
              <div className="user-profile-stats">
                <div><div className="user-stat-val">{userProfile.n_ratings}</div><div className="user-stat-lbl">notes</div></div>
                <div><div className="user-stat-val">{userProfile.avg_rating}</div><div className="user-stat-lbl">moy. / 5</div></div>
              </div>
              {userProfile.top_genres.length > 0 && (
                <div className="user-genres">{userProfile.top_genres.map(g => <span key={g} className="user-genre-chip">{g}</span>)}</div>
              )}
            </div>
          )}
          {userError && <div style={{ marginTop: 8, fontSize: '0.75rem', color: '#FDA4AF' }} role="alert">{userError}</div>}
        </div>

        <div>
          <div className="step-label"><span className="step-num">2</span>Modèle</div>
          <div className="model-cards" role="group">
            {models.map(m => (
              <ModelCardCompact key={m.id} model={m} selected={modelId === m.id} onSelect={setModelId} />
            ))}
          </div>
          {selectedModel && (
            <div style={{ marginTop: 8, padding: '8px 10px', background: 'rgba(255,255,255,0.03)', borderRadius: 6, fontSize: '0.72rem', color: 'var(--text-3)', lineHeight: 1.45 }}>
              {selectedModel.desc}
            </div>
          )}
        </div>

        <div>
          <div className="step-label"><span className="step-num">3</span>Lancer</div>
          <button className="btn-predict" onClick={handlePredict} disabled={!canPredict} aria-busy={loading}>
            {loading ? <><div className="spinner" /> Calcul…</> : <><svg width="13" height="13" viewBox="0 0 14 14" fill="none"><path d="M2.5 1.5l10 5.5-10 5.5V1.5z" fill="currentColor"/></svg> Prédire</>}
          </button>
        </div>
      </aside>

      <section aria-live="polite">
        {error && <div className="banner banner-error" role="alert">{error}</div>}
        {!loading && !results && !error && (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <div className="empty-title">Prédiction dataset</div>
            <div className="empty-desc">Sélectionnez un utilisateur du dataset (1–610), choisissez un modèle, et obtenez ses recommandations SVD/KNN réelles.</div>
          </div>
        )}
        {loading && (
          <div className="films-grid">
            {Array.from({ length: 10 }).map((_, i) => <div key={i} className="skeleton-card" style={{ animationDelay: `${i * 50}ms` }} />)}
          </div>
        )}
        {results && !loading && (
          <>
            <div className="results-header">
              <h2 className="section-title" style={{ margin: 0 }}>
                Top 10 — Utilisateur #{results.user_id}
                <span className="badge badge-indigo">{selectedModel?.name ?? results.model}</span>
              </h2>
              <div className="results-meta">
                <span><strong>{results.n_seen}</strong> films notés</span>
                <span><strong>{results.n_unseen}</strong> candidats</span>
              </div>
            </div>
            <ModelInfoBanner model={selectedModel} />
            <div className="films-grid">
              {results.recommendations.map((film, i) => <FilmCard key={film.movie_id} film={film} delay={i * 40} />)}
            </div>
          </>
        )}
      </section>
    </div>
  )
}

// ── Root Predict page ─────────────────────────────────────────────────────────
export default function Predict() {
  const [mode, setMode] = useState('profile')

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Prédiction</h1>
        <p className="page-sub">Recommandations personnalisées — avec ou sans compte dans le dataset.</p>
      </header>

      <div className="mode-tabs" role="tablist">
        <button
          className={`mode-tab${mode === 'profile' ? ' active' : ''}`}
          role="tab"
          aria-selected={mode === 'profile'}
          onClick={() => setMode('profile')}
        >
          🎭 Profil personnalisé
        </button>
        <button
          className={`mode-tab${mode === 'dataset' ? ' active' : ''}`}
          role="tab"
          aria-selected={mode === 'dataset'}
          onClick={() => setMode('dataset')}
        >
          📊 Utilisateur dataset
        </button>
      </div>

      {mode === 'profile' ? <ProfileMode /> : <DatasetMode />}
    </div>
  )
}
