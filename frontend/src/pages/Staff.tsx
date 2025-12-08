import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchStaff, createStaff, updateStaff, deleteStaff } from '../lib/api'
import type { StaffMember } from '../types/api'

interface StaffFormData {
  first_name: string
  last_name: string
  role_code: string
  department: string
  is_active: boolean
}

const StaffPage = () => {
  const [activeOnly, setActiveOnly] = useState(false)
  const [limit, setLimit] = useState(25)
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<string>('staff_id')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null)
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ['staff', { activeOnly, limit, search, sortBy, sortOrder }],
    queryFn: () =>
      fetchStaff({
        activeOnly,
        limit,
        search: search || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      }),
    staleTime: 60000,
  })

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortOrder('asc')
    }
  }

  const createMutation = useMutation({
    mutationFn: createStaff,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['staff'] })
      setShowCreateForm(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: Partial<StaffMember> }) => updateStaff(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['staff'] })
      setEditingId(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteStaff,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['staff'] })
      setDeleteConfirmId(null)
    },
  })

  const handleCreate = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data: Partial<StaffFormData> = {
      first_name: formData.get('first_name') as string || '',
      last_name: formData.get('last_name') as string || '',
      role_code: formData.get('role_code') as string || '',
      department: formData.get('department') as string || '',
      is_active: formData.get('is_active') === 'on',
    }
    createMutation.mutate(data)
  }

  const handleUpdate = (staff: StaffMember, e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data: Partial<StaffMember> = {
      first_name: formData.get('first_name') as string || staff.first_name,
      last_name: formData.get('last_name') as string || staff.last_name,
      role_code: formData.get('role_code') as string || staff.role_code,
      department: formData.get('department') as string || staff.department || '',
      is_active: formData.get('is_active') === 'on',
    }
    updateMutation.mutate({ id: staff.staff_id, data })
  }

  const handleDelete = (id: number) => {
    deleteMutation.mutate(id)
  }

  const roleBreakdown = useMemo(() => {
    if (!query.data) return []
    const summary = query.data.staff.reduce<Record<string, number>>((acc, staff) => {
      acc[staff.role_code] = (acc[staff.role_code] ?? 0) + 1
      return acc
    }, {})
    return Object.entries(summary).map(([role, count]) => ({ role, count }))
  }, [query.data])

  const StaffForm = ({
    staff,
    onSubmit,
    onCancel,
    submitLabel,
  }: {
    staff?: StaffMember
    onSubmit: (e: React.FormEvent<HTMLFormElement>) => void
    onCancel: () => void
    submitLabel: string
  }) => (
    <form onSubmit={onSubmit} style={{ display: 'grid', gap: '16px', padding: '20px', backgroundColor: '#f9fafb', borderRadius: '8px', marginTop: '16px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          First Name <span style={{ color: 'red' }}>*</span>
          <input
            type="text"
            name="first_name"
            required
            defaultValue={staff?.first_name}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          Last Name <span style={{ color: 'red' }}>*</span>
          <input
            type="text"
            name="last_name"
            required
            defaultValue={staff?.last_name}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Role Code <span style={{ color: 'red' }}>*</span>
          <input
            type="text"
            name="role_code"
            required
            defaultValue={staff?.role_code}
            placeholder="e.g., RN, MD, NP"
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          Department
          <input
            type="text"
            name="department"
            defaultValue={staff?.department || ''}
            placeholder="e.g., Emergency, ICU"
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
      </div>

      <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <input
          type="checkbox"
          name="is_active"
          defaultChecked={staff?.is_active ?? true}
          style={{ width: '18px', height: '18px' }}
        />
        Active Status
      </label>

      <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
        <button type="button" onClick={onCancel} className="ghost">
          Cancel
        </button>
        <button type="submit" disabled={createMutation.isPending || updateMutation.isPending}>
          {createMutation.isPending || updateMutation.isPending ? 'Saving...' : submitLabel}
        </button>
      </div>
    </form>
  )

  return (
    <div className="stack">
      <section className="panel">
        <header className="panel-header">
          <div>
            <h2>Staff roster</h2>
            <p className="subtle">Data served from /api/staff</p>
          </div>
          <div className="filters">
            <label>
              Search
              <input
                type="text"
                placeholder="Search staff..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #d1d5db', minWidth: '200px' }}
              />
            </label>
            <label>
              Sort by
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                <option value="staff_id">ID</option>
                <option value="first_name">First Name</option>
                <option value="last_name">Last Name</option>
                <option value="role_code">Role</option>
                <option value="department">Department</option>
                <option value="is_active">Status</option>
              </select>
            </label>
            <label>
              Order
              <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}>
                <option value="asc">Ascending</option>
                <option value="desc">Descending</option>
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
            <label className="toggle">
              <input type="checkbox" checked={activeOnly} onChange={(event) => setActiveOnly(event.target.checked)} />
              <span>Active only</span>
            </label>
            <button type="button" onClick={() => query.refetch()} className="ghost">
              Refresh
            </button>
            <button type="button" onClick={() => setShowCreateForm(!showCreateForm)} style={{ backgroundColor: '#3b82f6', color: 'white' }}>
              {showCreateForm ? 'Cancel' : '+ New Staff'}
            </button>
          </div>
        </header>

        {showCreateForm && (
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ marginBottom: '12px' }}>Create New Staff Member</h3>
            {createMutation.isError && (
              <p className="error" style={{ marginBottom: '12px' }}>
                Error: {(createMutation.error as Error)?.message || 'Failed to create staff member'}
              </p>
            )}
            <StaffForm
              onSubmit={handleCreate}
              onCancel={() => setShowCreateForm(false)}
              submitLabel="Create Staff Member"
            />
          </div>
        )}

        {query.isLoading && <p>Loading staff...</p>}
        {query.error && <p className="error">Failed to load staff list: {(query.error as Error).message}</p>}
        {updateMutation.isError && (
          <p className="error">Error updating staff: {(updateMutation.error as Error)?.message || 'Failed to update staff member'}</p>
        )}
        {deleteMutation.isError && (
          <p className="error">Error deleting staff: {(deleteMutation.error as Error)?.message || 'Failed to delete staff member'}</p>
        )}

        {query.data && (
          <div style={{ marginBottom: '12px', color: '#6b7280', fontSize: '14px' }}>
            Showing {query.data.staff.length} of {query.data.total} staff members
            {search && ` matching "${search}"`}
          </div>
        )}

        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th 
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  onClick={() => handleSort('staff_id')}
                >
                  ID {sortBy === 'staff_id' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  onClick={() => handleSort('first_name')}
                >
                  First Name {sortBy === 'first_name' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  onClick={() => handleSort('last_name')}
                >
                  Last Name {sortBy === 'last_name' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  onClick={() => handleSort('role_code')}
                >
                  Role {sortBy === 'role_code' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  onClick={() => handleSort('department')}
                >
                  Department {sortBy === 'department' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  onClick={() => handleSort('is_active')}
                >
                  Status {sortBy === 'is_active' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {query.data?.staff.map((staff: StaffMember) => (
                <tr key={staff.staff_id}>
                  {editingId === staff.staff_id ? (
                    <td colSpan={7} style={{ padding: 0 }}>
                      <div style={{ padding: '16px' }}>
                        <h4 style={{ marginBottom: '12px' }}>Edit Staff Member #{staff.staff_id}</h4>
                        <StaffForm
                          staff={staff}
                          onSubmit={(e) => handleUpdate(staff, e)}
                          onCancel={() => setEditingId(null)}
                          submitLabel="Update Staff Member"
                        />
                      </div>
                    </td>
                  ) : (
                    <>
                      <td>{staff.staff_id}</td>
                      <td>{staff.first_name}</td>
                      <td>{staff.last_name}</td>
                      <td>{staff.role_code}</td>
                      <td>{staff.department ?? '—'}</td>
                      <td>
                        <span className={staff.is_active ? 'pill ok' : 'pill warn'}>
                          {staff.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            type="button"
                            onClick={() => setEditingId(staff.staff_id)}
                            className="ghost"
                            style={{ fontSize: '12px', padding: '4px 8px' }}
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            onClick={() => setDeleteConfirmId(staff.staff_id)}
                            className="ghost"
                            style={{ fontSize: '12px', padding: '4px 8px', color: '#ef4444' }}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </>
                  )}
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

      {deleteConfirmId && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setDeleteConfirmId(null)}
        >
          <div
            style={{
              backgroundColor: 'white',
              padding: '24px',
              borderRadius: '8px',
              maxWidth: '400px',
              boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginBottom: '12px' }}>Confirm Delete</h3>
            <p style={{ marginBottom: '20px', color: '#6b7280' }}>
              Are you sure you want to delete staff member #{deleteConfirmId}? This action cannot be undone.
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button type="button" onClick={() => setDeleteConfirmId(null)} className="ghost">
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleDelete(deleteConfirmId)}
                disabled={deleteMutation.isPending}
                style={{ backgroundColor: '#ef4444', color: 'white' }}
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default StaffPage
