import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  fetchOverview,
  fetchEsiStats,
  fetchVitals,
  fetchPayorStats,
  fetchDiagnoses,
} from '../lib/api'
import type { EsiDistributionBreakdown } from '../types/api'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from 'recharts'

const chartPalette = ['#2563eb', '#7c3aed', '#059669', '#ea580c', '#facc15']

const DashboardPage = () => {
  const {
    data: overview,
    isLoading: overviewLoading,
    error: overviewError,
  } = useQuery({ queryKey: ['overview'], queryFn: fetchOverview })
  const { data: esiStats } = useQuery({ queryKey: ['esi-stats'], queryFn: fetchEsiStats })
  const { data: vitals } = useQuery({ queryKey: ['vitals'], queryFn: fetchVitals })
  const { data: payors } = useQuery({ queryKey: ['payors'], queryFn: fetchPayorStats })
  const { data: diagnoses } = useQuery({ queryKey: ['diagnoses'], queryFn: fetchDiagnoses })

  const loading = overviewLoading || !overview
  const error = overviewError

  const esiChartData = useMemo(() => overview?.esi_distribution ?? [], [overview])
  const arrivalData = useMemo(() => overview?.arrival_mode_distribution ?? [], [overview])
  const payorTypeData = useMemo(
    () => payors?.payor_type_distribution.map((entry) => ({ name: entry.type, value: entry.count })) ?? [],
    [payors],
  )

  const lwbsRows = useMemo(() => esiStats?.esi_statistics ?? [], [esiStats])

  if (error) {
    return <div className="panel">Unable to load dashboard: {(error as Error).message}</div>
  }

  if (loading) {
    return <div className="panel">Loading ED metrics...</div>
  }

  return (
    <div className="stack">
      <section className="panel kpi-grid">
        <div>
          <p className="eyebrow">Encounters</p>
          <h2>{overview.totals.encounters.toLocaleString()}</h2>
        </div>
        <div>
          <p className="eyebrow">Unique patients</p>
          <h2>{overview.totals.patients.toLocaleString()}</h2>
        </div>
        <div>
          <p className="eyebrow">Active staff</p>
          <h2>{overview.totals.staff.toLocaleString()}</h2>
        </div>
      </section>

      <div className="grid two">
        <section className="panel">
          <header>
            <h3>ESI distribution</h3>
            <p className="subtle">Volume per triage category</p>
          </header>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={esiChartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="level" tickFormatter={(value) => `ESI ${value}`} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" radius={4} />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel">
          <header>
            <h3>Arrival modes</h3>
            <p className="subtle">Top pathways into the ED</p>
          </header>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={arrivalData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="mode" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#7c3aed" radius={4} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <header>
            <h3>Payor mix</h3>
            <p className="subtle">Type distribution</p>
          </header>
          <div style={{ height: 260 }}>
            <ResponsiveContainer>
              <PieChart>
                <Tooltip />
                <Pie data={payorTypeData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={90}>
                  {payorTypeData.map((entry, idx) => (
                    <Cell key={entry.name} fill={chartPalette[idx % chartPalette.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <ul className="inline legend">
            {payorTypeData.map((entry, idx) => (
              <li key={entry.name}>
                <span className="swatch" style={{ background: chartPalette[idx % chartPalette.length] }} />
                {entry.name} ({entry.value})
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <header>
            <h3>Average vitals at triage</h3>
            <p className="subtle">First measurement captured per encounter</p>
          </header>
          <div className="vitals-grid">
            {Object.entries(vitals?.average_vitals ?? {}).map(([label, value]) => (
              <div key={label}>
                <p className="eyebrow">{label.replace(/_/g, ' ')}</p>
                <strong>{value ?? '—'}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <header>
            <h3>ESI-level safety signals</h3>
            <p className="subtle">Left Without Being Seen (LWBS) rate per level</p>
          </header>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>ESI</th>
                  <th>Total</th>
                  <th>LWBS</th>
                  <th>Rate</th>
                </tr>
              </thead>
              <tbody>
                {lwbsRows.map((row: EsiDistributionBreakdown) => (
                  <tr key={row.esi_level}>
                    <td>Level {row.esi_level}</td>
                    <td>{row.total_count.toLocaleString()}</td>
                    <td>{row.lwbs_count.toLocaleString()}</td>
                    <td>{(row.lwbs_rate * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel">
          <header>
            <h3>Top diagnoses</h3>
            <p className="subtle">Primary codes across encounters</p>
          </header>
          <ol className="diagnosis-list">
            {diagnoses?.top_diagnoses.map((dx) => (
              <li key={dx.code}>
                <span>{dx.code}</span>
                <strong>{dx.count}</strong>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  )
}

export default DashboardPage
