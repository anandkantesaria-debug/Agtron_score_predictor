-- =============================================================================
-- Supabase Telemetry & OOD Monitoring Schema for Agtron Score Predictor
-- Paste and run this entire script in the Supabase Dashboard -> SQL Editor
-- =============================================================================

create table if not exists public.prediction_logs (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    timestamp timestamptz not null default now(),
    endpoint text not null,
    model_version text not null,
    consumed double precision not null,
    temp double precision not null,
    quench double precision not null,
    month text not null,
    blend text not null,
    target_agtron double precision,
    predicted_agtron double precision not null,
    recommended_temp double precision,
    recommended_quench double precision,
    expected_agtron double precision,
    latency_ms double precision not null,
    inference_latency_ms double precision,
    compute_latency_ms double precision,
    is_ood boolean not null default false,
    ood_features jsonb not null default '[]'::jsonb,
    ood_warnings jsonb not null default '[]'::jsonb
);

-- Indexes for fast time-series drift queries and OOD filtering
create index if not exists idx_prediction_logs_timestamp
    on public.prediction_logs (timestamp desc);

create index if not exists idx_prediction_logs_is_ood
    on public.prediction_logs (is_ood)
    where is_ood = true;

create index if not exists idx_prediction_logs_blend
    on public.prediction_logs (blend);

-- Enable Row Level Security (RLS)
alter table public.prediction_logs enable row level security;

-- Allow the FastAPI microservice to insert telemetry rows using either
-- the anon/publishable key or the service_role key, while blocking public reads.
drop policy if exists "Allow insert for API telemetry" on public.prediction_logs;
create policy "Allow insert for API telemetry"
    on public.prediction_logs
    for insert
    to anon, authenticated, service_role
    with check (true);

-- Daily drift & latency monitoring view (visible in Supabase Table Editor -> Views)
create or replace view public.daily_ood_drift_summary as
select
    date_trunc('day', timestamp) as day,
    endpoint,
    model_version,
    count(*) as total_requests,
    sum(case when is_ood then 1 else 0 end) as ood_requests,
    round(
        100.0 * sum(case when is_ood then 1 else 0 end)::numeric / nullif(count(*), 0),
        2
    ) as ood_rate_pct,
    round(avg(latency_ms)::numeric, 2) as avg_latency_ms,
    round(
        percentile_cont(0.95) within group (order by latency_ms)::numeric,
        2
    ) as p95_latency_ms
from public.prediction_logs
group by 1, 2, 3
order by 1 desc, 2 asc;

