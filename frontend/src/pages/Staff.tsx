import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchStaff } from '../lib/api'
import type { StaffMember } from '../types/api'

const StaffPage = () => {
  const [activeOnly, setActiveOnly] = useState(true)

  const query = useQuery({
    queryKey: ['staff', activeOnly],
    queryFn: () => fetchStaff(activeOnly),
    staleTime: 60000,
  })

  const roleBreakdown = useMemo(() => {
    if (!query.data) return []
    const summary = query.data.staff.reduce<Record<string, number>>((acc, staff) => {
      acc[staff.role_code] = (acc[staff.role_code] ?? 0) + 1
      return acc
    }, {})
    return Object.entries(summary).map(([role, count]) => ({ role, count }))
  }, [query.data])

  return (
    <div className="stack">
      <section className="panel">
        <header className="panel-header">
          <div>
            <h2>Staff roster</h2>
            <p className="subtle">Data served from /api/staff</p>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={activeOnly} onChange={(event) => setActiveOnly(event.target.checked)} />
            <span>Show active only</span>
          </label>
        </header>

        {query.isLoading && <p>Loading staff...</p>}
        {query.error && <p className="error">Failed to load staff list.</p>}

        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Role</th>
                <th>Department</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {query.data?.staff.map((staff: StaffMember) => (
                <tr key={staff.staff_id}>
                  <td>{staff.staff_id}</td>
                  <td>
                    {staff.first_name} {staff.last_name}
                  </td>
                  <td>{staff.role_code}</td>
                  <td>{staff.department ?? '—'}</td>
                  <td>
                    <span className={staff.is_active ? 'pill ok' : 'pill warn'}>
                      {staff.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <header>
          <h3>Role mix</h3>
          <p className="subtle">Indexed off the filtered roster</p>
        </header>
        <ul className="role-list">
          {roleBreakdown.map((row) => (
            <li key={row.role}>
              <span>{row.role}</span>
              <strong>{row.count}</strong>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}

export default StaffPage
