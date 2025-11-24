import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchEncounters } from '../lib/api'
import type { Encounter } from '../types/api'

const formatDate = (value: string | null) => {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString()
  } catch (err) {
    return value
  }
}

const EncountersPage = () => {
  const [esi, setEsi] = useState<'all' | number>('all')
  const [limit, setLimit] = useState(25)

  const query = useQuery({
    queryKey: ['encounters', { esi, limit }],
    queryFn: () =>
      fetchEncounters({
        esi_level: esi === 'all' ? undefined : Number(esi),
        limit,
      }),
    refetchInterval: 120000,
  })

  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>Encounter feed</h2>
          <p className="subtle">Filtered from the Flask API</p>
        </div>
        <div className="filters">
          <label>
            ESI level
            <select value={esi} onChange={(event) => setEsi(event.target.value === 'all' ? 'all' : Number(event.target.value))}>
              <option value="all">All</option>
              {[1, 2, 3, 4, 5].map((level) => (
                <option key={level} value={level}>
                  Level {level}
                </option>
              ))}
            </select>
          </label>
          <label>
            Page size
            <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
              {[25, 50, 100].map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => query.refetch()} className="ghost">
            Refresh
          </button>
        </div>
      </header>

      {query.isLoading && <p>Loading encounters...</p>}
      {query.error && <p className="error">Unable to load encounters: {(query.error as Error).message}</p>}

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Patient</th>
              <th>Arrival</th>
              <th>ESI</th>
              <th>Complaint</th>
              <th>Disposition</th>
              <th>LWBS</th>
            </tr>
          </thead>
          <tbody>
            {query.data?.data.map((encounter: Encounter) => (
              <tr key={encounter.encounter_id}>
                <td>#{encounter.encounter_id}</td>
                <td>{encounter.patient_id}</td>
                <td>{formatDate(encounter.arrival_ts)}</td>
                <td>
                  <span className={`pill esi-${encounter.esi_level}`}>ESI {encounter.esi_level}</span>
                </td>
                <td>{encounter.chief_complaint}</td>
                <td>{encounter.disposition_code ?? 'Pending'}</td>
                <td>{encounter.left_without_being_seen ? 'Yes' : 'No'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default EncountersPage
