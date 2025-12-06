export interface Totals {
  encounters: number
  patients: number
  staff: number
}

export interface OverviewResponse {
  totals: Totals
  esi_distribution: Array<{ level: number; count: number }>
  disposition_distribution: Array<{ code: string; count: number }>
  arrival_mode_distribution: Array<{ mode: string; count: number }>
}

export interface EsiDistributionBreakdown {
  esi_level: number
  total_count: number
  lwbs_count: number
  lwbs_rate: number
  dispositions: Array<{ code: string; count: number }>
}

export interface EsiStatisticsResponse {
  esi_statistics: EsiDistributionBreakdown[]
}

export interface VitalsResponse {
  average_vitals: {
    heart_rate: number | null
    systolic_bp: number | null
    diastolic_bp: number | null
    respiratory_rate: number | null
    temperature_c: number | null
    spo2: number | null
    pain_score: number | null
  }
}

export interface PayorResponse {
  payor_name_distribution: Array<{ name: string; count: number }>
  payor_type_distribution: Array<{ type: string; count: number }>
}

export interface DiagnosisResponse {
  total_diagnoses: number
  primary_diagnosis_count: number
  top_diagnoses: Array<{ code: string; count: number }>
}

export interface Encounter {
  encounter_id: number
  patient_id: number
  arrival_ts: string
  triage_start_ts: string | null
  triage_end_ts: string | null
  provider_start_ts: string | null
  dispo_decision_ts: string | null
  departure_ts: string | null
  arrival_mode: string
  chief_complaint: string
  esi_level: number
  disposition_code: string | null
  referral_code: string | null
  left_without_being_seen: boolean
  notes: string | null
}

export interface EncounterListResponse {
  total: number
  limit: number
  offset: number
  sort_by?: string
  sort_order?: string
  search?: string
  data: Encounter[]
}

export interface StaffMember {
  staff_id: number
  first_name: string
  last_name: string
  role_code: string
  department: string | null
  is_active: boolean
}

export interface StaffResponse {
  total: number
  staff: StaffMember[]
}

export interface HealthStatus {
  status: string
  message: string
}

export interface PredictionModelsResponse {
  classification_models: Record<
    string,
    {
      description: string
      accuracy?: string
      best_for?: string
      endpoint: string
    }
  >
  regression_models: Record<
    string,
    {
      description: string
      r2_score?: string
      rmse?: string
      mae?: string
      endpoint: string
    }
  >
  feature_requirements: Record<string, string[]>
}

export type EsiModelType = 'logistic' | 'lda' | 'naive_bayes'

export interface EsiPredictionFeatures {
  patient_age: number
  sex_at_birth: 'M' | 'F'
  arrival_mode: string
  chief_complaint: string
  heart_rate: number
  bp_systolic: number
  bp_diastolic: number
  respiratory_rate: number
  temperature_c: number
  o2_saturation: number
  pain_score: number
  arrival_hour: number
  arrival_day_of_week: number
  is_weekend: 0 | 1
  payor_type: string
}

export interface EsiPredictionRequest {
  model: EsiModelType
  features: EsiPredictionFeatures
}

export interface EsiPredictionResponse {
  predicted_esi_level: number
  model_used: string
  confidence: Record<string, number | null> | null
  interpretation: string
}

export interface WaitTimeFeatures {
  esi_level: number
  patient_age: number
  sex_at_birth: 'M' | 'F'
  arrival_mode: string
  heart_rate: number
  bp_systolic: number
  respiratory_rate: number
  temperature_c: number
  o2_saturation: number
  arrival_hour: number
  is_weekend: 0 | 1
}

export interface WaitTimeResponse {
  predicted_wait_time_minutes: number
  predicted_wait_time_formatted: string
  interpretation: string
}

export interface VolumeForecastResponse {
  predicted_volume_per_hour: number
  predicted_arrivals_4_hours: number
  predicted_arrivals_8_hours: number
  input: {
    hour: number
    day_of_week: number
    day_name: string
    month: number
    is_weekend: boolean
  }
  interpretation: string
}

export interface WaitTimeData {
  encounter_id: number
  esi_level: number
  arrival_ts: string
  provider_start_ts: string
  wait_time_minutes: number
}

export interface WaitTimesResponse {
  total: number
  statistics: {
    average_wait_minutes: number
    min_wait_minutes: number
    max_wait_minutes: number
  }
  wait_times: WaitTimeData[]
}

export interface WaitTimeByEsiItem {
  esi_level: number
  count: number
  average_wait_minutes: number
  min_wait_minutes: number
  max_wait_minutes: number
  median_wait_minutes: number
}

export interface WaitTimesByEsiResponse {
  wait_times_by_esi: WaitTimeByEsiItem[]
}

export interface LengthOfStayResponse {
  total_encounters: number
  overall_statistics: {
    average_los_minutes: number
    min_los_minutes: number
    max_los_minutes: number
    median_los_minutes: number
  }
  los_by_esi: Array<{
    esi_level: number
    count: number
    average_los_minutes: number
    min_los_minutes: number
    max_los_minutes: number
    median_los_minutes: number
  }>
  los_by_disposition: Array<{
    disposition_code: string
    count: number
    average_los_minutes: number
    min_los_minutes: number
    max_los_minutes: number
  }>
}
