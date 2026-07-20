// =============================================================================
// src/admin.rs  — RouteFuel v0.5
//
// Admin Dashboard API
//
// Endpoints (all require X-API-Key with admin scope):
//   GET /admin/overview          — total spend, savings, request count
//   GET /admin/cache             — cache hit rate, top cached prompts
//   GET /admin/models/expensive  — top 5 most expensive models by total cost
//   GET /admin/models/usage      — all models ranked by request count
//   GET /admin/clients           — per-client spend breakdown
//   GET /admin/timeline          — hourly request/cost timeline (last 24h)
// =============================================================================

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use sqlx::Row;
use std::sync::Arc;
use tracing::instrument;

// =============================================================================
// Shared DB state for admin handlers
// =============================================================================

#[derive(Clone)]
pub struct AdminState {
    pub pool: Arc<PgPool>,
}

// =============================================================================
// Query params
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct DateRangeQuery {
    /// ISO date string e.g. "2026-05-01"
    #[serde(default = "thirty_days_ago")]
    pub from: String,
    /// ISO date string e.g. "2026-05-07"
    #[serde(default = "today")]
    pub to: String,
}

fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

fn thirty_days_ago() -> String {
    (chrono::Utc::now() - chrono::Duration::days(30))
        .format("%Y-%m-%d")
        .to_string()
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    pub period_from:           String,
    pub period_to:             String,
    pub total_requests:        i64,
    pub successful_requests:   i64,
    pub failed_requests:       i64,
    pub total_spend_usd:       f64,
    pub total_saved_usd:       f64,
    pub total_saved_pct:       f64,
    pub avg_latency_ms:        f64,
    pub avg_routing_ms:        f64,
    pub cache_hits:            i64,
    pub cache_hit_rate_pct:    f64,
}

#[derive(Debug, Serialize)]
pub struct CacheStatsResponse {
    pub total_entries:         i64,
    pub total_hits:            i64,
    pub hit_rate_pct:          f64,
    pub estimated_saved_usd:   f64,
    pub top_cached:            Vec<TopCachedEntry>,
}

#[derive(Debug, Serialize)]
pub struct TopCachedEntry {
    pub prompt_preview: String,
    pub model_used:     String,
    pub hit_count:      i64,
}

#[derive(Debug, Serialize)]
pub struct ModelCostEntry {
    pub rank:             i32,
    pub model_api_id:     String,
    pub provider:         String,
    pub total_requests:   i64,
    pub total_spend_usd:  f64,
    pub avg_cost_usd:     f64,
    pub total_tokens_in:  i64,
    pub total_tokens_out: i64,
}

#[derive(Debug, Serialize)]
pub struct ClientSpendEntry {
    pub client_id:         String,
    pub total_requests:    i64,
    pub total_spend_usd:   f64,
    pub total_saved_usd:   f64,
    pub avg_latency_ms:    f64,
}

#[derive(Debug, Serialize)]
pub struct TimelineEntry {
    pub hour:            String,
    pub request_count:   i64,
    pub spend_usd:       f64,
    pub saved_usd:       f64,
    pub avg_latency_ms:  f64,
    pub cache_hits:      i64,
}

// =============================================================================
// GET /admin/overview
// =============================================================================

#[instrument(skip(state))]
pub async fn overview_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let row = sqlx::query(
        r#"
        SELECT
            COUNT(*)                                          AS total_requests,
            COUNT(*) FILTER (WHERE status = 'success')       AS successful_requests,
            COUNT(*) FILTER (WHERE status = 'failed')        AS failed_requests,
            COALESCE(SUM(cost_cents)  / 100.0, 0)           AS total_spend_usd,
            COALESCE(SUM(cost_saved_cents) / 100.0, 0)      AS total_saved_usd,
            COALESCE(AVG(latency_ms), 0)                     AS avg_latency_ms,
            COALESCE(AVG(routing_decision_ms), 0)            AS avg_routing_ms,
            COUNT(*) FILTER (WHERE from_cache = TRUE)        AS cache_hits
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_one(state.pool.as_ref())
    .await;

    match row {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(r) => {
            let total:   i64 = r.get("total_requests");
            let hits:    i64 = r.get("cache_hits");
            let spend:   f64 = r.get("total_spend_usd");
            let saved:   f64 = r.get("total_saved_usd");
            let hit_pct: f64 = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };
            let saved_pct: f64 = if (spend + saved) > 0.0 {
                saved / (spend + saved) * 100.0
            } else { 0.0 };

            Json(OverviewResponse {
                period_from:         q.from,
                period_to:           q.to,
                total_requests:      total,
                successful_requests: r.get("successful_requests"),
                failed_requests:     r.get("failed_requests"),
                total_spend_usd:     spend,
                total_saved_usd:     saved,
                total_saved_pct:     saved_pct,
                avg_latency_ms:      r.get("avg_latency_ms"),
                avg_routing_ms:      r.get("avg_routing_ms"),
                cache_hits:          hits,
                cache_hit_rate_pct:  hit_pct,
            })
            .into_response()
        }
    }
}

// =============================================================================
// GET /admin/cache
// =============================================================================

#[instrument(skip(state))]
pub async fn cache_stats_handler(
    State(state): State<AdminState>,
) -> impl IntoResponse {
    // Total entries + hits
    let summary = sqlx::query(
        r#"
        SELECT
            COUNT(*)           AS total_entries,
            SUM(hit_count)     AS total_hits
        FROM semantic_cache
        "#,
    )
    .fetch_one(state.pool.as_ref())
    .await;

    // Top 10 most-hit cache entries
    let top_rows = sqlx::query(
        r#"
        SELECT prompt_preview, model_used, hit_count
        FROM semantic_cache
        ORDER BY hit_count DESC
        LIMIT 10
        "#,
    )
    .fetch_all(state.pool.as_ref())
    .await
    .unwrap_or_default();

    // Requests served from cache
    let cache_hit_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM request_logs WHERE from_cache = TRUE",
    )
    .fetch_one(state.pool.as_ref())
    .await
    .unwrap_or(0);

    let total_req: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM request_logs",
    )
    .fetch_one(state.pool.as_ref())
    .await
    .unwrap_or(0);

    // Estimated savings: avg cost per non-cached request × cache hits
    let avg_cost: f64 = sqlx::query_scalar(
        r#"
        SELECT COALESCE(AVG(cost_cents) / 100.0, 0)
        FROM request_logs
        WHERE from_cache = FALSE AND status = 'success'
        "#,
    )
    .fetch_one(state.pool.as_ref())
    .await
    .unwrap_or(0.0);

    let hit_rate = if total_req > 0 {
        cache_hit_count as f64 / total_req as f64 * 100.0
    } else { 0.0 };

    let top_cached: Vec<TopCachedEntry> = top_rows.iter().map(|r| TopCachedEntry {
        prompt_preview: r.get("prompt_preview"),
        model_used:     r.get("model_used"),
        hit_count:      r.get::<i64, _>("hit_count"),
    }).collect();

    let (total_entries, total_hits) = match summary {
        Ok(r) => (
            r.get::<i64, _>("total_entries"),
            r.get::<i64, _>("total_hits"),
        ),
        Err(_) => (0, 0),
    };

    Json(CacheStatsResponse {
        total_entries,
        total_hits,
        hit_rate_pct:        hit_rate,
        estimated_saved_usd: avg_cost * cache_hit_count as f64,
        top_cached,
    })
    .into_response()
}

// =============================================================================
// GET /admin/models/expensive  — Top 5 most expensive models by total cost
// =============================================================================

#[instrument(skip(state))]
pub async fn top_expensive_models_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            ROW_NUMBER() OVER (ORDER BY SUM(cost_cents) DESC)::INT AS rank,
            model_api_id,
            provider,
            COUNT(*)                                     AS total_requests,
            SUM(cost_cents)       / 100.0                AS total_spend_usd,
            AVG(cost_cents)       / 100.0                AS avg_cost_usd,
            COALESCE(SUM(tokens_in),  0)                 AS total_tokens_in,
            COALESCE(SUM(tokens_out), 0)                 AS total_tokens_out
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
          AND status = 'success'
        GROUP BY model_api_id, provider
        ORDER BY SUM(cost_cents) DESC
        LIMIT 5
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;

    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let models: Vec<ModelCostEntry> = rows.iter().map(|r| ModelCostEntry {
                rank:             r.get("rank"),
                model_api_id:     r.get("model_api_id"),
                provider:         r.get("provider"),
                total_requests:   r.get("total_requests"),
                total_spend_usd:  r.get("total_spend_usd"),
                avg_cost_usd:     r.get("avg_cost_usd"),
                total_tokens_in:  r.get("total_tokens_in"),
                total_tokens_out: r.get("total_tokens_out"),
            }).collect();
            Json(models).into_response()
        }
    }
}

// =============================================================================
// GET /admin/models/usage  — All models ranked by request count
// =============================================================================

#[instrument(skip(state))]
pub async fn model_usage_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC)::INT AS rank,
            model_api_id,
            provider,
            COUNT(*)                                     AS total_requests,
            SUM(cost_cents)       / 100.0                AS total_spend_usd,
            AVG(cost_cents)       / 100.0                AS avg_cost_usd,
            COALESCE(SUM(tokens_in),  0)                 AS total_tokens_in,
            COALESCE(SUM(tokens_out), 0)                 AS total_tokens_out
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
          AND status = 'success'
        GROUP BY model_api_id, provider
        ORDER BY COUNT(*) DESC
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;

    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let models: Vec<ModelCostEntry> = rows.iter().map(|r| ModelCostEntry {
                rank:             r.get("rank"),
                model_api_id:     r.get("model_api_id"),
                provider:         r.get("provider"),
                total_requests:   r.get("total_requests"),
                total_spend_usd:  r.get("total_spend_usd"),
                avg_cost_usd:     r.get("avg_cost_usd"),
                total_tokens_in:  r.get("total_tokens_in"),
                total_tokens_out: r.get("total_tokens_out"),
            }).collect();
            Json(models).into_response()
        }
    }
}

// =============================================================================
// GET /admin/clients  — Per-client spend breakdown
// =============================================================================

#[instrument(skip(state))]
pub async fn client_spend_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            COALESCE(client_id, 'anonymous')             AS client_id,
            COUNT(*)                                     AS total_requests,
            SUM(cost_cents)       / 100.0                AS total_spend_usd,
            SUM(cost_saved_cents) / 100.0                AS total_saved_usd,
            AVG(latency_ms)                              AS avg_latency_ms
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
        GROUP BY client_id
        ORDER BY SUM(cost_cents) DESC
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;

    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let clients: Vec<ClientSpendEntry> = rows.iter().map(|r| ClientSpendEntry {
                client_id:       r.get("client_id"),
                total_requests:  r.get("total_requests"),
                total_spend_usd: r.get("total_spend_usd"),
                total_saved_usd: r.get("total_saved_usd"),
                avg_latency_ms:  r.get("avg_latency_ms"),
            }).collect();
            Json(clients).into_response()
        }
    }
}

// =============================================================================
// GET /admin/timeline  — Hourly breakdown for last 24 hours
// =============================================================================

#[instrument(skip(state))]
pub async fn timeline_handler(
    State(state): State<AdminState>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            TO_CHAR(DATE_TRUNC('hour', created_at), 'YYYY-MM-DD HH24:00') AS hour,
            COUNT(*)                                     AS request_count,
            COALESCE(SUM(cost_cents)       / 100.0, 0)  AS spend_usd,
            COALESCE(SUM(cost_saved_cents) / 100.0, 0)  AS saved_usd,
            COALESCE(AVG(latency_ms),       0)           AS avg_latency_ms,
            COUNT(*) FILTER (WHERE from_cache = TRUE)    AS cache_hits
        FROM request_logs
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY DATE_TRUNC('hour', created_at) ASC
        "#,
    )
    .fetch_all(state.pool.as_ref())
    .await;

    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let timeline: Vec<TimelineEntry> = rows.iter().map(|r| TimelineEntry {
                hour:          r.get("hour"),
                request_count: r.get("request_count"),
                spend_usd:     r.get("spend_usd"),
                saved_usd:     r.get("saved_usd"),
                avg_latency_ms:r.get("avg_latency_ms"),
                cache_hits:    r.get("cache_hits"),
            }).collect();
            Json(timeline).into_response()
        }
    }
}
