import { NavLink, Route, Routes } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import DashboardPage from './pages/Dashboard'
import EncountersPage from './pages/Encounters'
import PredictionsPage from './pages/Predictions'
import StaffPage from './pages/Staff'
import { fetchHealth } from './lib/api'
import './App.css'

const routes = [
  { path: '/', label: 'Dashboard' },
  { path: '/encounters', label: 'Encounters' },
  { path: '/predictions', label: 'Predictions' },
  { path: '/staff', label: 'Staff & Resources' },
]

function App() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30000,
  })

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <p className="eyebrow">NU DS5110 | Team 22</p>
          <h1>ED Ops Command</h1>
        </div>
        <nav>
          {routes.map((route) => (
            <NavLink
              key={route.path}
              to={route.path}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
              end={route.path === '/'}
            >
              {route.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <p className={`health-pill ${health?.status === 'healthy' ? 'ok' : 'warn'}`}>
            <span className="dot" /> {health ? health.message : 'Checking API...'}
          </p>
          <p className="subtle">Backend: {import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5001/api'}</p>
        </div>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/encounters" element={<EncountersPage />} />
          <Route path="/predictions" element={<PredictionsPage />} />
          <Route path="/staff" element={<StaffPage />} />
          <Route path="*" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
