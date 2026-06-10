export default function Sidebar({ activeTab, setActiveTab, isOpen, onClose }) {
  const navItems = [
    {
      id: 'dashboard',
      label: 'Vue d\'ensemble',
      icon: (
        <svg className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
          <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h3a1 1 0 001-1v-3h2v3a1 1 0 001 1h3a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z"/>
        </svg>
      ),
    },
    {
      id: 'predict',
      label: 'Prédire',
      icon: (
        <svg className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
        </svg>
      ),
    },
  ]

  return (
    <>
      <div className={`overlay${isOpen ? ' open' : ''}`} onClick={onClose} aria-hidden="true" />

      <nav className={`sidebar${isOpen ? ' open' : ''}`} aria-label="Navigation principale">
        <div className="sidebar-logo">
          <div className="logo-icon" aria-hidden="true">🎬</div>
          <div>
            <div className="logo-name">CinéRec</div>
            <div className="logo-sub">MLOps · EPT DIC3</div>
          </div>
        </div>

        <div className="nav">
          <div className="nav-section-label">Navigation</div>
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item${activeTab === item.id ? ' active' : ''}`}
              onClick={() => { setActiveTab(item.id); onClose(); }}
              aria-current={activeTab === item.id ? 'page' : undefined}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            className="mlflow-link"
            aria-label="Ouvrir MLflow UI (nouvelle fenêtre)"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M7 4v3l2 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            MLflow UI
            <svg className="mlflow-link-ext" width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
              <path d="M2 8L8 2M8 2H4M8 2v4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </a>
        </div>
      </nav>
    </>
  )
}
