import { useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Dashboard from './views/Dashboard.jsx'
import Predict from './views/Predict.jsx'

export default function App() {
  const [tab,      setTab]      = useState('dashboard')
  const [sideOpen, setSideOpen] = useState(false)

  return (
    <>
      <a href="#main-content" className="skip-link" style={{
        position: 'absolute', top: '-100%', left: 16,
        background: 'var(--indigo)', color: '#fff',
        padding: '8px 16px', borderRadius: 'var(--radius-sm)',
        fontSize: '0.875rem', zIndex: 9999, textDecoration: 'none',
      }}>
        Aller au contenu principal
      </a>

      <div className="layout">
        <Sidebar
          activeTab={tab}
          setActiveTab={setTab}
          isOpen={sideOpen}
          onClose={() => setSideOpen(false)}
        />

        <button
          className="hamburger"
          onClick={() => setSideOpen(v => !v)}
          aria-expanded={sideOpen}
          aria-label="Ouvrir le menu"
        >
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <rect y="2"  width="18" height="2" rx="1" fill="currentColor"/>
            <rect y="8"  width="18" height="2" rx="1" fill="currentColor"/>
            <rect y="14" width="18" height="2" rx="1" fill="currentColor"/>
          </svg>
        </button>

        <main className="main" id="main-content">
          {tab === 'dashboard' && <Dashboard />}
          {tab === 'predict'   && <Predict />}
        </main>
      </div>
    </>
  )
}
