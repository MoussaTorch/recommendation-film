const GENRE_CLASS = {
  'Action': 'gc-action', 'Crime': 'gc-crime', 'Comedy': 'gc-comedy',
  'Romance': 'gc-romance', 'Sci-Fi': 'gc-scifi', 'Thriller': 'gc-thriller',
  'Drama': 'gc-drama', 'War': 'gc-war', 'Adventure': 'gc-adventure',
  'Animation': 'gc-animation', 'Mystery': 'gc-mystery',
}

function Stars({ rating }) {
  const full  = Math.min(5, Math.floor(rating))
  const empty = 5 - full
  return (
    <div className="stars" aria-hidden="true">
      {'★'.repeat(full).split('').map((_, i) => <span key={`f${i}`} className="star-f">★</span>)}
      {'★'.repeat(empty).split('').map((_, i) => <span key={`e${i}`} className="star-e">★</span>)}
    </div>
  )
}

export default function FilmCard({ film, delay = 0 }) {
  const genres = Array.isArray(film.genres_list) ? film.genres_list.slice(0, 3) : []

  return (
    <article
      className="film-card"
      style={{ animationDelay: `${delay}ms` }}
      aria-label={`${film.title} — note prédite ${film.predicted_rating}`}
    >
      <div className="film-rank">#{String(film.rank).padStart(2, '0')}</div>

      <div>
        <h3 className="film-title">{film.title}</h3>
        {film.year && <div className="film-year">{film.year}</div>}
      </div>

      {genres.length > 0 && (
        <div className="film-genres" aria-label={`Genres : ${genres.join(', ')}`}>
          {genres.map(g => (
            <span key={g} className={`genre-chip ${GENRE_CLASS[g] ?? 'gc-default'}`}>{g}</span>
          ))}
        </div>
      )}

      <div className="film-footer">
        <div
          className="rating-row"
          aria-label={`Note prédite : ${film.predicted_rating} sur 5`}
        >
          <Stars rating={film.predicted_rating} />
          <span className="rating-num">{film.predicted_rating.toFixed(1)}</span>
        </div>
        <span className="rating-lbl">prédite</span>
      </div>
    </article>
  )
}
