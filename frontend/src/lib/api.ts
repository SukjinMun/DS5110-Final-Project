import axios from 'axios'
import type {
  DiagnosisResponse,
  EncounterListResponse,
  EsiPredictionRequest,
  EsiPredictionResponse,
  EsiStatisticsResponse,
  HealthStatus,
  OverviewResponse,
  PayorResponse,
  PredictionModelsResponse,
  StaffResponse,
  VitalsResponse,
  VolumeForecastResponse,
  WaitTimeFeatures,
  WaitTimeResponse,
} from '../types/api'

const baseURL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5001/api'

export const apiClient = axios.create({
  baseURL,
  timeout: 15000,
})

const extractData = <T>(promise: Promise<{ data: T }>) => promise.then((res) => res.data)

export const fetchHealth = () => extractData<HealthStatus>(apiClient.get('/health'))
export const fetchOverview = () => extractData<OverviewResponse>(apiClient.get('/statistics/overview'))
export const fetchEsiStats = () => extractData<EsiStatisticsResponse>(apiClient.get('/statistics/esi'))
export const fetchVitals = () => extractData<VitalsResponse>(apiClient.get('/statistics/vitals'))
export const fetchPayorStats = () => extractData<PayorResponse>(apiClient.get('/statistics/payor'))
export const fetchDiagnoses = () => extractData<DiagnosisResponse>(apiClient.get('/statistics/diagnoses'))
export const fetchPredictionMetadata = () =>
  extractData<PredictionModelsResponse>(apiClient.get('/predictions/models/info'))

export const fetchEncounters = (params: { esi_level?: number; limit?: number; offset?: number } = {}) =>
  extractData<EncounterListResponse>(
    apiClient.get('/encounters', {
      params,
    }),
  )

export const fetchStaff = (activeOnly: boolean) =>
  extractData<StaffResponse>(
    apiClient.get('/staff', {
      params: { active_only: activeOnly },
    }),
  )

export const predictEsi = (payload: EsiPredictionRequest) =>
  extractData<EsiPredictionResponse>(apiClient.post('/predictions/esi', payload))

export const predictWaitTime = (features: WaitTimeFeatures) =>
  extractData<WaitTimeResponse>(apiClient.post('/predictions/wait-time', { features }))

export const predictVolume = (params: {
  hour: number
  day_of_week: number
  month: number
  is_weekend: number
}) =>
  extractData<VolumeForecastResponse>(
    apiClient.get('/predictions/volume', {
      params,
    }),
  )
