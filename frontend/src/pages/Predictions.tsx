import { useState } from 'react'
import type { FormEvent } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  fetchPredictionMetadata,
  predictEsi,
  predictWaitTime,
  predictVolume,
} from '../lib/api'
import type {
  EsiModelType,
  EsiPredictionRequest,
  EsiPredictionResponse,
  WaitTimeFeatures,
  WaitTimeResponse,
  VolumeForecastResponse,
} from '../types/api'

const defaultEsiFeatures: EsiPredictionRequest['features'] = {
  patient_age: 42,
  sex_at_birth: 'F',
  arrival_mode: 'Walk-in',
  chief_complaint: 'Abdominal pain',
  heart_rate: 96,
  bp_systolic: 132,
  bp_diastolic: 78,
  respiratory_rate: 18,
  temperature_c: 37.5,
  o2_saturation: 97,
  pain_score: 6,
  arrival_hour: 14,
  arrival_day_of_week: 1,
  is_weekend: 0,
  payor_type: 'private',
}

const defaultWaitFeatures: WaitTimeFeatures = {
  esi_level: 3,
  patient_age: 50,
  sex_at_birth: 'M',
  arrival_mode: 'EMS',
  heart_rate: 105,
  bp_systolic: 140,
  respiratory_rate: 20,
  temperature_c: 37,
  o2_saturation: 95,
  arrival_hour: 19,
  is_weekend: 1,
}

const defaultVolumePayload = {
  hour: 16,
  day_of_week: 3,
  month: 11,
  is_weekend: 0,
}

const PredictionsPage = () => {
  const { data: modelMeta } = useQuery({
    queryKey: ['prediction-metadata'],
    queryFn: fetchPredictionMetadata,
  })

  const [esiPayload, setEsiPayload] = useState<EsiPredictionRequest>({
    model: 'logistic',
    features: defaultEsiFeatures,
  })
  const [waitPayload, setWaitPayload] = useState<WaitTimeFeatures>(defaultWaitFeatures)
  const [volumePayload, setVolumePayload] = useState(defaultVolumePayload)

  const waitNumberFields: Array<{
    key: keyof WaitTimeFeatures
    label: string
    min?: number
    max?: number
    step?: number
  }> = [
    { key: 'esi_level', label: 'ESI level', min: 1, max: 5 },
    { key: 'patient_age', label: 'Patient age', min: 0 },
    { key: 'heart_rate', label: 'Heart rate', min: 0 },
    { key: 'bp_systolic', label: 'BP systolic', min: 0 },
    { key: 'respiratory_rate', label: 'Respiratory rate', min: 0 },
    { key: 'temperature_c', label: 'Temperature °C', step: 0.1 },
    { key: 'o2_saturation', label: 'O₂ saturation', min: 0, max: 100 },
    { key: 'arrival_hour', label: 'Arrival hour', min: 0, max: 23 },
    { key: 'is_weekend', label: 'Weekend flag', min: 0, max: 1 },
  ]

  const esiMutation = useMutation<EsiPredictionResponse, Error, EsiPredictionRequest>({
    mutationFn: predictEsi,
  })
  const waitMutation = useMutation<WaitTimeResponse, Error, WaitTimeFeatures>({
    mutationFn: predictWaitTime,
  })
  const volumeMutation = useMutation<VolumeForecastResponse, Error, typeof volumePayload>({
    mutationFn: predictVolume,
  })

  const submitEsi = (event: FormEvent) => {
    event.preventDefault()
    esiMutation.mutate(esiPayload)
  }

  const submitWait = (event: FormEvent) => {
    event.preventDefault()
    waitMutation.mutate(waitPayload)
  }

  const submitVolume = (event: FormEvent) => {
    event.preventDefault()
    volumeMutation.mutate(volumePayload)
  }

  return (
    <div className="stack">
      <section className="panel">
        <header>
          <h2>Model card overview</h2>
          <p className="subtle">Served from /api/predictions/models/info</p>
        </header>
        {modelMeta ? (
          <div className="grid two">
            <div>
              <h3>Classification models</h3>
              <ul className="model-list">
                {Object.entries(modelMeta.classification_models).map(([key, meta]) => (
                  <li key={key}>
                    <strong>{key}</strong>
                    <p>{meta.description}</p>
                    <p className="subtle">Accuracy: {meta.accuracy ?? 'n/a'}</p>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3>Regression models</h3>
              <ul className="model-list">
                {Object.entries(modelMeta.regression_models).map(([key, meta]) => (
                  <li key={key}>
                    <strong>{key}</strong>
                    <p>{meta.description}</p>
                    <p className="subtle">
                      R² {meta.r2_score ?? '—'} | RMSE {meta.rmse ?? '—'} | MAE {meta.mae ?? '—'}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <p>Loading model metadata...</p>
        )}
      </section>

      <section className="panel">
        <header>
          <h2>ESI triage prediction</h2>
          <p className="subtle">Multinomial logistic baseline</p>
        </header>
        <form className="form-grid" onSubmit={submitEsi}>
          <label>
            Model variant
            <select
              value={esiPayload.model}
              onChange={(event) => setEsiPayload((prev) => ({ ...prev, model: event.target.value as EsiModelType }))}
            >
              <option value="logistic">Logistic regression</option>
              <option value="lda">LDA</option>
              <option value="naive_bayes">Naive Bayes</option>
            </select>
          </label>
          <label>
            Age
            <input
              type="number"
              min={0}
              value={esiPayload.features.patient_age}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, patient_age: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Sex at birth
            <select
              value={esiPayload.features.sex_at_birth}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, sex_at_birth: event.target.value as 'M' | 'F' },
                }))
              }
            >
              <option value="F">Female</option>
              <option value="M">Male</option>
            </select>
          </label>
          <label>
            Arrival mode
            <input
              value={esiPayload.features.arrival_mode}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, arrival_mode: event.target.value },
                }))
              }
            />
          </label>
          <label>
            Chief complaint
            <input
              value={esiPayload.features.chief_complaint}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, chief_complaint: event.target.value },
                }))
              }
            />
          </label>
          <label>
            Heart rate
            <input
              type="number"
              value={esiPayload.features.heart_rate}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, heart_rate: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            BP systolic
            <input
              type="number"
              value={esiPayload.features.bp_systolic}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, bp_systolic: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            BP diastolic
            <input
              type="number"
              value={esiPayload.features.bp_diastolic}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, bp_diastolic: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Respiratory rate
            <input
              type="number"
              value={esiPayload.features.respiratory_rate}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, respiratory_rate: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Temperature °C
            <input
              type="number"
              step="0.1"
              value={esiPayload.features.temperature_c}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, temperature_c: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            O₂ saturation
            <input
              type="number"
              value={esiPayload.features.o2_saturation}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, o2_saturation: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Pain score (0-10)
            <input
              type="number"
              min={0}
              max={10}
              value={esiPayload.features.pain_score}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, pain_score: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Arrival hour
            <input
              type="number"
              min={0}
              max={23}
              value={esiPayload.features.arrival_hour}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, arrival_hour: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Arrival weekday (0-6)
            <input
              type="number"
              min={0}
              max={6}
              value={esiPayload.features.arrival_day_of_week}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, arrival_day_of_week: Number(event.target.value) },
                }))
              }
            />
          </label>
          <label>
            Weekend flag (0/1)
            <input
              type="number"
              min={0}
              max={1}
              value={esiPayload.features.is_weekend}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, is_weekend: Number(event.target.value) as 0 | 1 },
                }))
              }
            />
          </label>
          <label>
            Payor type
            <input
              value={esiPayload.features.payor_type}
              onChange={(event) =>
                setEsiPayload((prev) => ({
                  ...prev,
                  features: { ...prev.features, payor_type: event.target.value },
                }))
              }
            />
          </label>
          <div className="form-actions">
            <button type="submit" disabled={esiMutation.isPending}>
              {esiMutation.isPending ? 'Predicting…' : 'Predict ESI'}
            </button>
          </div>
        </form>
        {esiMutation.data && (
          <div className="result-block">
            <p>
              Predicted level: <strong>ESI {esiMutation.data.predicted_esi_level}</strong>
            </p>
            <p className="subtle">{esiMutation.data.interpretation}</p>
          </div>
        )}
      </section>

      <div className="grid two">
        <section className="panel">
          <header>
            <h3>Wait time forecast</h3>
            <p className="subtle">Linear regression + scaler bundle</p>
          </header>
          <form className="form-grid" onSubmit={submitWait}>
            {waitNumberFields.map((field) => (
              <label key={field.key}>
                {field.label}
                <input
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step ?? 1}
                  value={waitPayload[field.key] as number}
                  onChange={(event) =>
                    setWaitPayload((prev) => ({
                      ...prev,
                      [field.key]: Number(event.target.value),
                    }))
                  }
                />
              </label>
            ))}
            <label>
              Sex at birth
              <select
                value={waitPayload.sex_at_birth}
                onChange={(event) =>
                  setWaitPayload((prev) => ({
                    ...prev,
                    sex_at_birth: event.target.value as 'M' | 'F',
                  }))
                }
              >
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </label>
            <label>
              Arrival mode
              <input
                value={waitPayload.arrival_mode}
                onChange={(event) =>
                  setWaitPayload((prev) => ({
                    ...prev,
                    arrival_mode: event.target.value,
                  }))
                }
              />
            </label>
            <div className="form-actions">
              <button type="submit" disabled={waitMutation.isPending}>
                {waitMutation.isPending ? 'Scoring…' : 'Predict wait time'}
              </button>
            </div>
          </form>
          {waitMutation.data && (
            <div className="result-block">
              <p>
                Expected wait: <strong>{waitMutation.data.predicted_wait_time_formatted}</strong>
              </p>
              <p className="subtle">({waitMutation.data.predicted_wait_time_minutes.toFixed(1)} minutes)</p>
              <p className="subtle">{waitMutation.data.interpretation}</p>
            </div>
          )}
        </section>

        <section className="panel">
          <header>
            <h3>Volume outlook</h3>
            <p className="subtle">Poisson GLM per hour</p>
          </header>
          <form className="form-grid" onSubmit={submitVolume}>
            <label>
              Hour (0-23)
              <input
                type="number"
                min={0}
                max={23}
                value={volumePayload.hour}
                onChange={(event) => setVolumePayload((prev) => ({ ...prev, hour: Number(event.target.value) }))}
              />
            </label>
            <label>
              Day of week (0=Mon)
              <input
                type="number"
                min={0}
                max={6}
                value={volumePayload.day_of_week}
                onChange={(event) => setVolumePayload((prev) => ({ ...prev, day_of_week: Number(event.target.value) }))}
              />
            </label>
            <label>
              Month (1-12)
              <input
                type="number"
                min={1}
                max={12}
                value={volumePayload.month}
                onChange={(event) => setVolumePayload((prev) => ({ ...prev, month: Number(event.target.value) }))}
              />
            </label>
            <label>
              Weekend flag
              <select
                value={volumePayload.is_weekend}
                onChange={(event) => setVolumePayload((prev) => ({ ...prev, is_weekend: Number(event.target.value) }))}
              >
                <option value={0}>Weekday</option>
                <option value={1}>Weekend</option>
              </select>
            </label>
            <div className="form-actions">
              <button type="submit" disabled={volumeMutation.isPending}>
                {volumeMutation.isPending ? 'Estimating…' : 'Forecast volume'}
              </button>
            </div>
          </form>
          {volumeMutation.data && (
            <div className="result-block">
              <p>
                Expected arrivals per hour: <strong>{volumeMutation.data.predicted_volume_per_hour.toFixed(2)}</strong>
              </p>
              <p className="subtle">Next 4h: {volumeMutation.data.predicted_arrivals_4_hours.toFixed(1)} patients</p>
              <p className="subtle">Next 8h: {volumeMutation.data.predicted_arrivals_8_hours.toFixed(1)} patients</p>
              <p className="subtle">{volumeMutation.data.interpretation}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default PredictionsPage
