import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchEncounters, createEncounter, updateEncounter, deleteEncounter } from '../lib/api'
import type { Encounter } from '../types/api'

const formatDate = (value: string | null) => {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString()
  } catch (err) {
    return value
  }
}

const formatDateForInput = (value: string | null) => {
  if (!value) return ''
  try {
    const date = new Date(value)
    return date.toISOString().slice(0, 16) // Format: YYYY-MM-DDTHH:mm
  } catch (err) {
    return ''
  }
}

interface EncounterFormData {
  patient_id: number
  arrival_ts: string
  triage_start_ts: string
  triage_end_ts: string
  provider_start_ts: string
  dispo_decision_ts: string
  departure_ts: string
  arrival_mode: string
  chief_complaint: string
  esi_level: number
  disposition_code: string
  referral_code: string
  left_without_being_seen: boolean
  notes: string
}

const EncountersPage = () => {
  const [esi, setEsi] = useState<'all' | number>('all')
  const [limit, setLimit] = useState(25)
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<string>('encounter_id')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null)
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ['encounters', { esi, limit, search, sortBy, sortOrder }],
    queryFn: () =>
      fetchEncounters({
        esi_level: esi === 'all' ? undefined : Number(esi),
        limit,
        search: search || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      }),
    refetchInterval: 120000,
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
    mutationFn: createEncounter,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['encounters'] })
      setShowCreateForm(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: Partial<Encounter> }) => updateEncounter(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['encounters'] })
      setEditingId(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteEncounter,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['encounters'] })
      setDeleteConfirmId(null)
    },
  })

  const handleCreate = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data: Partial<EncounterFormData> = {
      patient_id: Number(formData.get('patient_id')),
      arrival_ts: formData.get('arrival_ts') as string || new Date().toISOString(),
      triage_start_ts: formData.get('triage_start_ts') as string || '',
      triage_end_ts: formData.get('triage_end_ts') as string || '',
      provider_start_ts: formData.get('provider_start_ts') as string || '',
      dispo_decision_ts: formData.get('dispo_decision_ts') as string || '',
      departure_ts: formData.get('departure_ts') as string || '',
      arrival_mode: formData.get('arrival_mode') as string || '',
      chief_complaint: formData.get('chief_complaint') as string || '',
      esi_level: Number(formData.get('esi_level')) || 3,
      disposition_code: formData.get('disposition_code') as string || '',
      referral_code: formData.get('referral_code') as string || '',
      left_without_being_seen: formData.get('left_without_being_seen') === 'on',
      notes: formData.get('notes') as string || '',
    }
    createMutation.mutate(data)
  }

  const handleUpdate = (encounter: Encounter, e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const data: Partial<Encounter> = {
      patient_id: Number(formData.get('patient_id')),
      arrival_ts: formData.get('arrival_ts') as string || encounter.arrival_ts,
      triage_start_ts: formData.get('triage_start_ts') as string || encounter.triage_start_ts || null,
      triage_end_ts: formData.get('triage_end_ts') as string || encounter.triage_end_ts || null,
      provider_start_ts: formData.get('provider_start_ts') as string || encounter.provider_start_ts || null,
      dispo_decision_ts: formData.get('dispo_decision_ts') as string || encounter.dispo_decision_ts || null,
      departure_ts: formData.get('departure_ts') as string || encounter.departure_ts || null,
      arrival_mode: formData.get('arrival_mode') as string || encounter.arrival_mode || '',
      chief_complaint: formData.get('chief_complaint') as string || encounter.chief_complaint || '',
      esi_level: Number(formData.get('esi_level')) || encounter.esi_level,
      disposition_code: formData.get('disposition_code') as string || encounter.disposition_code || null,
      referral_code: formData.get('referral_code') as string || encounter.referral_code || null,
      left_without_being_seen: formData.get('left_without_being_seen') === 'on',
      notes: formData.get('notes') as string || encounter.notes || null,
    }
    updateMutation.mutate({ id: encounter.encounter_id, data })
  }

  const handleDelete = (id: number) => {
    deleteMutation.mutate(id)
  }

  const EncounterForm = ({
    encounter,
    onSubmit,
    onCancel,
    submitLabel,
  }: {
    encounter?: Encounter
    onSubmit: (e: React.FormEvent<HTMLFormElement>) => void
    onCancel: () => void
    submitLabel: string
  }) => (
    <form onSubmit={onSubmit} style={{ display: 'grid', gap: '16px', padding: '20px', backgroundColor: '#f9fafb', borderRadius: '8px', marginTop: '16px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Patient ID <span style={{ color: 'red' }}>*</span>
          <input
            type="number"
            name="patient_id"
            required
            defaultValue={encounter?.patient_id}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          ESI Level
          <select
            name="esi_level"
            defaultValue={encounter?.esi_level || 3}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          >
            {[1, 2, 3, 4, 5].map((level) => (
              <option key={level} value={level}>
                Level {level}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label>
        Chief Complaint
        <input
          type="text"
          name="chief_complaint"
          defaultValue={encounter?.chief_complaint || ''}
          style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
        />
      </label>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Arrival Time
          <input
            type="datetime-local"
            name="arrival_ts"
            defaultValue={encounter ? formatDateForInput(encounter.arrival_ts) : formatDateForInput(new Date().toISOString())}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          Arrival Mode
          <input
            type="text"
            name="arrival_mode"
            defaultValue={encounter?.arrival_mode || ''}
            placeholder="e.g., Ambulance, Walk-in"
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Triage Start
          <input
            type="datetime-local"
            name="triage_start_ts"
            defaultValue={encounter ? formatDateForInput(encounter.triage_start_ts) : ''}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          Triage End
          <input
            type="datetime-local"
            name="triage_end_ts"
            defaultValue={encounter ? formatDateForInput(encounter.triage_end_ts) : ''}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Provider Start
          <input
            type="datetime-local"
            name="provider_start_ts"
            defaultValue={encounter ? formatDateForInput(encounter.provider_start_ts) : ''}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          Disposition Decision
          <input
            type="datetime-local"
            name="dispo_decision_ts"
            defaultValue={encounter ? formatDateForInput(encounter.dispo_decision_ts) : ''}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Departure Time
          <input
            type="datetime-local"
            name="departure_ts"
            defaultValue={encounter ? formatDateForInput(encounter.departure_ts) : ''}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label>
          Disposition Code
          <input
            type="text"
            name="disposition_code"
            defaultValue={encounter?.disposition_code || ''}
            placeholder="e.g., DISCH, ADMIT"
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <label>
          Referral Code
          <input
            type="text"
            name="referral_code"
            defaultValue={encounter?.referral_code || ''}
            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db' }}
          />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingTop: '24px' }}>
          <input
            type="checkbox"
            name="left_without_being_seen"
            defaultChecked={encounter?.left_without_being_seen || false}
            style={{ width: '18px', height: '18px' }}
          />
          Left Without Being Seen (LWBS)
        </label>
      </div>

      <label>
        Notes
        <textarea
          name="notes"
          defaultValue={encounter?.notes || ''}
          rows={3}
          style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #d1d5db', fontFamily: 'inherit' }}
        />
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
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>Encounter feed</h2>
          <p className="subtle">Filtered from the Flask API</p>
        </div>
        <div className="filters">
          <label>
            Search
            <input
              type="text"
              placeholder="Search encounters..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #d1d5db', minWidth: '200px' }}
            />
          </label>
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
            Sort by
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="encounter_id">ID</option>
              <option value="patient_id">Patient ID</option>
              <option value="arrival_ts">Arrival Time</option>
              <option value="esi_level">ESI Level</option>
              <option value="chief_complaint">Chief Complaint</option>
              <option value="disposition_code">Disposition</option>
              <option value="departure_ts">Departure Time</option>
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
          <button type="button" onClick={() => query.refetch()} className="ghost">
            Refresh
          </button>
          <button type="button" onClick={() => setShowCreateForm(!showCreateForm)} style={{ backgroundColor: '#3b82f6', color: 'white' }}>
            {showCreateForm ? 'Cancel' : '+ New Encounter'}
          </button>
        </div>
      </header>

      {showCreateForm && (
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ marginBottom: '12px' }}>Create New Encounter</h3>
          {createMutation.isError && (
            <p className="error" style={{ marginBottom: '12px' }}>
              Error: {(createMutation.error as Error)?.message || 'Failed to create encounter'}
            </p>
          )}
          <EncounterForm
            onSubmit={handleCreate}
            onCancel={() => setShowCreateForm(false)}
            submitLabel="Create Encounter"
          />
        </div>
      )}

      {query.isLoading && <p>Loading encounters...</p>}
      {query.error && <p className="error">Unable to load encounters: {(query.error as Error).message}</p>}
      {updateMutation.isError && (
        <p className="error">Error updating encounter: {(updateMutation.error as Error)?.message || 'Failed to update encounter'}</p>
      )}
      {deleteMutation.isError && (
        <p className="error">Error deleting encounter: {(deleteMutation.error as Error)?.message || 'Failed to delete encounter'}</p>
      )}

      {query.data && (
        <div style={{ marginBottom: '12px', color: '#6b7280', fontSize: '14px' }}>
          Showing {query.data.data.length} of {query.data.total} encounters
          {search && ` matching "${search}"`}
        </div>
      )}

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th 
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => handleSort('encounter_id')}
              >
                ID {sortBy === 'encounter_id' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => handleSort('patient_id')}
              >
                Patient {sortBy === 'patient_id' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => handleSort('arrival_ts')}
              >
                Arrival {sortBy === 'arrival_ts' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => handleSort('esi_level')}
              >
                ESI {sortBy === 'esi_level' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => handleSort('chief_complaint')}
              >
                Complaint {sortBy === 'chief_complaint' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => handleSort('disposition_code')}
              >
                Disposition {sortBy === 'disposition_code' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th>LWBS</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {query.data?.data.map((encounter: Encounter) => (
              <tr key={encounter.encounter_id}>
                {editingId === encounter.encounter_id ? (
                  <td colSpan={8} style={{ padding: 0 }}>
                    <div style={{ padding: '16px' }}>
                      <h4 style={{ marginBottom: '12px' }}>Edit Encounter #{encounter.encounter_id}</h4>
                      <EncounterForm
                        encounter={encounter}
                        onSubmit={(e) => handleUpdate(encounter, e)}
                        onCancel={() => setEditingId(null)}
                        submitLabel="Update Encounter"
                      />
                    </div>
                  </td>
                ) : (
                  <>
                    <td>#{encounter.encounter_id}</td>
                    <td>{encounter.patient_id}</td>
                    <td>{formatDate(encounter.arrival_ts)}</td>
                    <td>
                      <span className={`pill esi-${encounter.esi_level}`}>ESI {encounter.esi_level}</span>
                    </td>
                    <td>{encounter.chief_complaint}</td>
                    <td>{encounter.disposition_code ?? 'Pending'}</td>
                    <td>{encounter.left_without_being_seen ? 'Yes' : 'No'}</td>
                    <td>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          type="button"
                          onClick={() => setEditingId(encounter.encounter_id)}
                          className="ghost"
                          style={{ fontSize: '12px', padding: '4px 8px' }}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          onClick={() => setDeleteConfirmId(encounter.encounter_id)}
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
              Are you sure you want to delete encounter #{deleteConfirmId}? This action cannot be undone.
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
    </section>
  )
}

export default EncountersPage
