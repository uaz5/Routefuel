// ============================================================================
// src/cost_tracker.rs
// PostgreSQL audit trail and cost tracking
// Non-blocking async writes via tokio::spawn
// ============================================================================

use crate::connectors::Provider;
use crate::tokens::TokenCostBreakdown;
use chrono::{DateTime, Utc};
use sqlx::postgres::PgPool;
use sqlx::{Error as SqlxError, Row};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, error, instrument, warn};

#[derive(Error, Debug)]
pub enum CostTrackerError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] SqlxError),

    #[error("Invalid provider")]
    InvalidProvider,

    #[error("Invalid model")]
    InvalidModel,

    #[error("Query failed: {0}")]
    QueryFailed(String),
}

// ============================================================================
// DATABASE SCHEMA (migrations should exist in migrations/)
// ============================================================================

// CREATE TABLE request_logs (
//     id BIGSERIAL PRIMARY KEY,
//     request_id VARCHAR(36) NOT NULL UNIQUE,
//     provider VARCHAR(50) NOT NULL,
//     model_name VARCHAR(100) NOT NULL,
//     tokens_in INTEGER NOT NULL,
//     tokens_out INTEGER NOT NULL,
//     cost_cents DECIMAL(10, 6) NOT NULL,
//     cost_saved_cents DECIMAL(10, 6) NOT NULL,
//     latency_ms INTEGER NOT NULL,
//     routing_decision_ms INTEGER NOT NULL,
//     client_id VARCHAR(100),
//     user_tier VARCHAR(20),
//     priority VARCHAR(20),
//     status VARCHAR(20) NOT NULL DEFAULT 'success',
//     error_message TEXT,
//     created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
//     INDEX idx_request_id (request_id),
//     INDEX idx_provider (provider),
//     INDEX idx_created_at (created_at),
//     INDEX idx_client_id (client_id)
// );

// ============================================================================
// RECORD TYPES
// ============================================================================

#[derive(Debug, Clone)]
pub struct RequestLogRecord {
    pub id: i64,
    pub request_id: String,
    pub provider: String,
    pub model_name: String,
    pub tokens_in: i32,
    pub tokens_out: i32,
    pub cost_cents: f64,
    pub cost_saved_cents: f64,
    pub latency_ms: i32,
    pub routing_decision_ms: i32,
    pub client_id: Option<String>,
    pub user_tier: Option<String>,
    pub priority: Option<String>,
    pub status: String,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DayAuditReport {
    pub date: String,
    pub total_requests: i64,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub total_tokens_in: i64,
    pub total_tokens_out: i64,
    pub total_cost_cents: f64,
    pub total_cost_saved_cents: f64,
    pub avg_latency_ms: f64,
    pub providers: Vec<ProviderBreakdown>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProviderBreakdown {
    pub provider: String,
    pub request_count: i64,
    pub total_cost_cents: f64,
    pub total_saved_cents: f64,
    pub avg_latency_ms: f64,
}

// ============================================================================
// COST TRACKER
// ============================================================================

pub struct CostTracker {
    pool: Arc<PgPool>,
}

impl CostTracker {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }

    /// Record a request asynchronously to the database
    ///
    /// This spawns a background task via tokio::spawn so it doesn't block
    /// the response. If the write fails, it logs an error but doesn't fail the request.
    ///
    /// This ensures zero latency impact on the response path.
    #[instrument(skip(self), fields(request_id = %request_id))]
    pub fn record_request(
        &self,
        request_id: String,
        provider: Provider,
        model_name: String,
        token_cost: &TokenCostBreakdown,
        baseline_cost_cents: f64,
        latency_ms: u64,
        routing_decision_ms: u64,
        client_id: Option<String>,
        user_tier: Option<String>,
        priority: Option<String>,
    ) {
        let pool = Arc::clone(&self.pool);
        let provider_str = provider.to_string();

        // Clone data for the async task
        let token_cost = token_cost.clone();

        // Spawn async task - fire and forget
        tokio::spawn(async move {
            if let Err(e) = Self::write_request_log(
                pool,
                request_id.clone(),
                provider_str,
                model_name,
                token_cost,
                baseline_cost_cents,
                latency_ms,
                routing_decision_ms,
                client_id,
                user_tier,
                priority,
            )
            .await
            {
                error!(
                    request_id = %request_id,
                    error = %e,
                    "Failed to write request log to database"
                );
            }
        });

        debug!(
            request_id = %request_id,
            cost_cents = token_cost.total_cost_cents,
            "Queued request log for database write"
        );
    }

    /// Record a failed request asynchronously
    #[instrument(skip(self), fields(request_id = %request_id))]
    pub fn record_error(
        &self,
        request_id: String,
        provider: Provider,
        model_name: String,
        error_message: String,
        latency_ms: u64,
        routing_decision_ms: u64,
        client_id: Option<String>,
    ) {
        let pool = Arc::clone(&self.pool);
        let provider_str = provider.to_string();

        tokio::spawn(async move {
            if let Err(e) = sqlx::query(
                r#"
                INSERT INTO request_logs (
                    request_id, provider, model_name,
                    tokens_in, tokens_out,
                    cost_cents, cost_saved_cents,
                    latency_ms, routing_decision_ms,
                    client_id,
                    status, error_message
                )
                VALUES ($1, $2, $3, 0, 0, 0, 0, $4, $5, $6, 'failed', $7)
                "#,
            )
            .bind(&request_id)
            .bind(&provider_str)
            .bind(&model_name)
            .bind(latency_ms as i32)
            .bind(routing_decision_ms as i32)
            .bind(&client_id)
            .bind(&error_message)
            .execute(pool.as_ref())
            .await
            {
                error!(
                    request_id = %request_id,
                    error = %e,
                    "Failed to write error log"
                );
            }
        });

        debug!(
            request_id = %request_id,
            error = %error_message,
            "Queued error log for database write"
        );
    }

    /// Internal: Write request log to database
    async fn write_request_log(
        pool: Arc<PgPool>,
        request_id: String,
        provider: String,
        model_name: String,
        token_cost: TokenCostBreakdown,
        baseline_cost_cents: f64,
        latency_ms: u64,
        routing_decision_ms: u64,
        client_id: Option<String>,
        user_tier: Option<String>,
        priority: Option<String>,
    ) -> Result<(), CostTrackerError> {
        let cost_saved = baseline_cost_cents - token_cost.total_cost_cents;

        sqlx::query(
            r#"
            INSERT INTO request_logs (
                request_id, provider, model_name,
                tokens_in, tokens_out,
                cost_cents, cost_saved_cents,
                latency_ms, routing_decision_ms,
                client_id, user_tier, priority,
                status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'success')
            "#,
        )
        .bind(&request_id)
        .bind(&provider)
        .bind(&model_name)
        .bind(token_cost.input_tokens as i32)
        .bind(token_cost.output_tokens as i32)
        .bind(token_cost.total_cost_cents)
        .bind(cost_saved)
        .bind(latency_ms as i32)
        .bind(routing_decision_ms as i32)
        .bind(&client_id)
        .bind(&user_tier)
        .bind(&priority)
        .execute(pool.as_ref())
        .await?;

        debug!(
            request_id = %request_id,
            "Request log written to database"
        );

        Ok(())
    }

    /// Get daily audit report
    #[instrument(skip(self))]
    pub async fn get_daily_report(&self, date: &str) -> Result<DayAuditReport, CostTrackerError> {
        // Get summary stats
        let row = sqlx::query(
            r#"
            SELECT
                DATE(created_at)::TEXT as date,
                COUNT(*) as total_requests,
                COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_requests,
                COUNT(CASE WHEN status != 'success' THEN 1 END) as failed_requests,
                COALESCE(SUM(tokens_in), 0) as total_tokens_in,
                COALESCE(SUM(tokens_out), 0) as total_tokens_out,
                COALESCE(SUM(cost_cents), 0) as total_cost_cents,
                COALESCE(SUM(cost_saved_cents), 0) as total_cost_saved_cents,
                COALESCE(AVG(latency_ms), 0) as avg_latency_ms
            FROM request_logs
            WHERE DATE(created_at) = $1::DATE AND status = 'success'
            GROUP BY DATE(created_at)
            "#,
        )
        .bind(date)
        .fetch_optional(self.pool.as_ref())
        .await?;

        let (
            total_requests,
            successful_requests,
            failed_requests,
            total_tokens_in,
            total_tokens_out,
            total_cost_cents,
            total_cost_saved_cents,
            avg_latency_ms,
        ) = match row {
            Some(r) => (
                r.get::<i64, _>("total_requests"),
                r.get::<i64, _>("successful_requests"),
                r.get::<i64, _>("failed_requests"),
                r.get::<i64, _>("total_tokens_in"),
                r.get::<i64, _>("total_tokens_out"),
                r.get::<f64, _>("total_cost_cents"),
                r.get::<f64, _>("total_cost_saved_cents"),
                r.get::<f64, _>("avg_latency_ms"),
            ),
            None => (0, 0, 0, 0, 0, 0.0, 0.0, 0.0),
        };

        // Get provider breakdown
        let providers_rows = sqlx::query(
            r#"
            SELECT
                provider,
                COUNT(*) as request_count,
                COALESCE(SUM(cost_cents), 0) as total_cost_cents,
                COALESCE(SUM(cost_saved_cents), 0) as total_saved_cents,
                COALESCE(AVG(latency_ms), 0) as avg_latency_ms
            FROM request_logs
            WHERE DATE(created_at) = $1::DATE AND status = 'success'
            GROUP BY provider
            ORDER BY request_count DESC
            "#,
        )
        .bind(date)
        .fetch_all(self.pool.as_ref())
        .await?;

        let providers: Vec<ProviderBreakdown> = providers_rows
            .iter()
            .map(|r| ProviderBreakdown {
                provider: r.get("provider"),
                request_count: r.get("request_count"),
                total_cost_cents: r.get("total_cost_cents"),
                total_saved_cents: r.get("total_saved_cents"),
                avg_latency_ms: r.get("avg_latency_ms"),
            })
            .collect();

        Ok(DayAuditReport {
            date: date.to_string(),
            total_requests,
            successful_requests,
            failed_requests,
            total_tokens_in,
            total_tokens_out,
            total_cost_cents,
            total_cost_saved_cents,
            avg_latency_ms,
            providers,
        })
    }

    /// Get total savings for a date range
    #[instrument(skip(self))]
    pub async fn get_total_savings(
        &self,
        start_date: &str,
        end_date: &str,
    ) -> Result<(f64, i64), CostTrackerError> {
        let row = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(cost_saved_cents), 0) as total_savings,
                COUNT(*) as request_count
            FROM request_logs
            WHERE status = 'success'
                AND created_at >= $1::TIMESTAMP
                AND created_at <= $2::TIMESTAMP
            "#,
        )
        .bind(format!("{} 00:00:00", start_date))
        .bind(format!("{} 23:59:59", end_date))
        .fetch_one(self.pool.as_ref())
        .await?;

        let total_savings: f64 = row.get("total_savings");
        let request_count: i64 = row.get("request_count");

        Ok((total_savings, request_count))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_serialization() {
        let record = RequestLogRecord {
            id: 1,
            request_id: "req-123".to_string(),
            provider: "OpenAI".to_string(),
            model_name: "gpt-4o".to_string(),
            tokens_in: 100,
            tokens_out: 50,
            cost_cents: 0.075,
            cost_saved_cents: 2.425,
            latency_ms: 245,
            routing_decision_ms: 8,
            client_id: Some("client-1".to_string()),
            user_tier: Some("pro".to_string()),
            priority: Some("cost".to_string()),
            status: "success".to_string(),
            error_message: None,
            created_at: Utc::now(),
        };

        assert_eq!(record.provider, "OpenAI");
        assert_eq!(record.tokens_in + record.tokens_out, 150);
    }
}
