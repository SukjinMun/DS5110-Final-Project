import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  fetchOverview,
  fetchEsiStats,
  fetchVitals,
  fetchPayorStats,
  fetchDiagnoses,
  fetchWaitTimesByEsi,
  fetchLengthOfStay,
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
  Legend,
  Line,
  LineChart,
  ComposedChart,
  RadialBarChart,
  RadialBar,
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
  const { data: waitTimesByEsi } = useQuery({ queryKey: ['wait-times-by-esi'], queryFn: fetchWaitTimesByEsi })
  const { data: lengthOfStay } = useQuery({ queryKey: ['length-of-stay'], queryFn: fetchLengthOfStay })

  const loading = overviewLoading || !overview
  const error = overviewError

  const esiChartData = useMemo(() => overview?.esi_distribution ?? [], [overview])
  const arrivalData = useMemo(() => overview?.arrival_mode_distribution ?? [], [overview])
  const payorTypeData = useMemo(
    () => payors?.payor_type_distribution.map((entry) => ({ name: entry.type, value: entry.count })) ?? [],
    [payors],
  )

  const lwbsRows = useMemo(() => esiStats?.esi_statistics ?? [], [esiStats])
  
  // Prepare LWBS chart data
  const lwbsChartData = useMemo(
    () =>
      lwbsRows.map((row) => ({
        esi_level: `ESI ${row.esi_level}`,
        total: row.total_count,
        lwbs: row.lwbs_count,
        lwbs_rate: (row.lwbs_rate * 100).toFixed(1),
      })),
    [lwbsRows],
  )

  // Prepare wait times chart data
  const waitTimesChartData = useMemo(
    () =>
      waitTimesByEsi?.wait_times_by_esi.map((item) => ({
        esi_level: `ESI ${item.esi_level}`,
        average: item.average_wait_minutes,
        median: item.median_wait_minutes,
        min: item.min_wait_minutes,
        max: item.max_wait_minutes,
        count: item.count,
      })) ?? [],
    [waitTimesByEsi],
  )

  // Prepare LOS chart data
  const losChartData = useMemo(
    () =>
      lengthOfStay?.los_by_esi.map((item) => ({
        esi_level: `ESI ${item.esi_level}`,
        average: item.average_los_minutes,
        median: item.median_los_minutes,
        min: item.min_los_minutes,
        max: item.max_los_minutes,
        count: item.count,
      })) ?? [],
    [lengthOfStay],
  )

  // Prepare top diagnoses chart data for horizontal bar chart (sorted by count, highest first)
  const diagnosesChartData = useMemo(
    () =>
      diagnoses?.top_diagnoses
        .map((dx, index) => ({
          code: dx.code,
          count: dx.count,
          fill: chartPalette[index % chartPalette.length],
        }))
        .sort((a, b) => b.count - a.count) ?? [],
    [diagnoses],
  )

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
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={lwbsChartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="esi_level" />
              <YAxis yAxisId="left" allowDecimals={false} />
              <YAxis yAxisId="right" orientation="right" allowDecimals={false} />
              <Tooltip
                formatter={(value: number, name: string) => {
                  if (name === 'lwbs_rate') return `${value}%`
                  return value.toLocaleString()
                }}
              />
              <Legend />
              <Bar yAxisId="left" dataKey="total" fill="#2563eb" name="Total Encounters" radius={4} />
              <Bar yAxisId="left" dataKey="lwbs" fill="#ea580c" name="LWBS Count" radius={4} />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="lwbs_rate"
                stroke="#facc15"
                strokeWidth={2}
                name="LWBS Rate %"
                dot={{ r: 4 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </section>

        <section className="panel">
          <header>
            <h3>Top diagnoses</h3>
            <p className="subtle">Primary codes across encounters (sorted by count)</p>
          </header>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={diagnosesChartData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <defs>
                {diagnosesChartData.map((entry, index) => (
                  <linearGradient key={`gradient-${index}`} id={`diagnosisGradient-${index}`} x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor={entry.fill} stopOpacity={0.9} />
                    <stop offset="100%" stopColor={entry.fill} stopOpacity={0.6} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e5e7eb" />
              <XAxis type="number" allowDecimals={false} tick={{ fill: '#6b7280', fontSize: 11 }} />
              <YAxis 
                dataKey="code" 
                type="category" 
                width={80}
                tick={{ fill: '#6b7280', fontSize: 11 }}
                axisLine={{ stroke: '#d1d5db' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                }}
                formatter={(value: number) => [`${value.toLocaleString()} encounters`, 'Count']}
                labelFormatter={(label) => `Diagnosis: ${label}`}
              />
              <Bar 
                dataKey="count" 
                radius={[0, 8, 8, 0]}
                animationDuration={800}
              >
                {diagnosesChartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={`url(#diagnosisGradient-${index})`}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <header>
            <h3>Wait times by ESI level</h3>
            <p className="subtle">Average time from arrival to provider</p>
          </header>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={waitTimesChartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="esi_level" />
              <YAxis label={{ value: 'Minutes', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                formatter={(value: number) => `${value.toFixed(1)} min`}
                labelFormatter={(label) => label}
              />
              <Legend />
              <Bar dataKey="average" fill="#2563eb" name="Average" radius={4} />
              <Bar dataKey="median" fill="#7c3aed" name="Median" radius={4} />
              <Line type="monotone" dataKey="min" stroke="#059669" strokeWidth={2} name="Min" dot={{ r: 3 }} />
              <Line type="monotone" dataKey="max" stroke="#ea580c" strokeWidth={2} name="Max" dot={{ r: 3 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </section>

        <section className="panel">
          <header>
            <h3>Length of stay by ESI level</h3>
            <p className="subtle">Time from arrival to departure</p>
          </header>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={losChartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="esi_level" />
              <YAxis label={{ value: 'Minutes', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                formatter={(value: number) => `${value.toFixed(1)} min`}
                labelFormatter={(label) => label}
              />
              <Legend />
              <Bar dataKey="average" fill="#2563eb" name="Average" radius={4} />
              <Bar dataKey="median" fill="#7c3aed" name="Median" radius={4} />
              <Line type="monotone" dataKey="min" stroke="#059669" strokeWidth={2} name="Min" dot={{ r: 3 }} />
              <Line type="monotone" dataKey="max" stroke="#ea580c" strokeWidth={2} name="Max" dot={{ r: 3 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </section>
      </div>

      {lengthOfStay && (
        <section className="panel">
          <header>
            <h3>Overall length of stay statistics</h3>
            <p className="subtle">Total encounters: {lengthOfStay.total_encounters.toLocaleString()}</p>
          </header>
          <div className="kpi-grid">
            <div>
              <p className="eyebrow">Average LOS</p>
              <h2>{(lengthOfStay.overall_statistics.average_los_minutes / 60).toFixed(1)} hours</h2>
              <p className="subtle">{lengthOfStay.overall_statistics.average_los_minutes.toFixed(1)} minutes</p>
            </div>
            <div>
              <p className="eyebrow">Median LOS</p>
              <h2>{(lengthOfStay.overall_statistics.median_los_minutes / 60).toFixed(1)} hours</h2>
              <p className="subtle">{lengthOfStay.overall_statistics.median_los_minutes.toFixed(1)} minutes</p>
            </div>
            <div>
              <p className="eyebrow">Min LOS</p>
              <h2>{(lengthOfStay.overall_statistics.min_los_minutes / 60).toFixed(1)} hours</h2>
              <p className="subtle">{lengthOfStay.overall_statistics.min_los_minutes.toFixed(1)} minutes</p>
            </div>
            <div>
              <p className="eyebrow">Max LOS</p>
              <h2>{(lengthOfStay.overall_statistics.max_los_minutes / 60).toFixed(1)} hours</h2>
              <p className="subtle">{lengthOfStay.overall_statistics.max_los_minutes.toFixed(1)} minutes</p>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}

export default DashboardPage
