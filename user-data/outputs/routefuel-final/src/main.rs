// ============================================================================
// src/main.rs
// RouteFuel production server with live connectors and cost auditing
// ============================================================================

mod circuit_breaker;
mod connectors;
mod cost_tracker;
mod rate_limiter;
mod route_engine;
mod tokens;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use crate::circuit_breaker::CircuitBreaker;
use crate::connectors::{
    AnthropicConnector, ChatCompletionRequest, ChatCompletionResponse, Connector,
    ConnectorManager, OpenAIConnector, Provider,
};
use crate::cost_tracker::CostTracker;
use crate::route_engine::RouteEngine;
use crate::tokens::TokenCostBreakdown;

// ============================================================================
// APPLICATION STATE
// ============================================================================

#[derive(Clone)]
struct AppState {
    route_engine: Arc<RouteEngine>,
    connector_manager: Arc<ConnectorManager>,
    cost_tracker: Arc<CostTracker>,
    circuit_breaker: Arc<CircuitBreaker>,
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(serde::Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

enum ApiError {
    BadRequest(String),
    RateLimited,
    CircuitOpen,
    ProviderError(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, error_type) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg, "invalid_request_error"),
            ApiError::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded".to_string(),
                "rate_limit_error",
            ),
            ApiError::CircuitOpen => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Service temporarily unavailable".to_string(),
                "server_error",
            ),
            ApiError::ProviderError(msg) => (StatusCode::BAD_GATEWAY, msg, "provider_error"),
            ApiError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                msg,
                "internal_error",
            ),
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
            },
        });

        (status, body).into_response()
    }
}

// ============================================================================
// CHAT COMPLETIONS HANDLER
// ============================================================================

#[instrument(skip(state, request), fields(request_id))]
async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let request_id = Uuid::new_v4().to_string();
    tracing::Span::current().record("request_id", &request_id);

    let start = Instant::now();
    info!(
        model = %request.model,
        message_count = request.messages.len(),
        "Received chat completion request"
    );

    // ========================================================================
    // STEP 1: COUNT INPUT TOKENS PRECISELY
    // ========================================================================

    let input_tokens = tokens::count_request_tokens(&request.messages, &request.model)
        .map_err(|e| {
            error!("Token counting failed: {}", e);
            ApiError::InternalError(format!("Token counting failed: {}", e))
        })?;

    let estimated_output = tokens::estimate_output_tokens(request.max_tokens, &request.model);

    debug!(
        input_tokens = input_tokens,
        estimated_output_tokens = estimated_output,
        "Counted request tokens"
    );

    // ========================================================================
    // STEP 2: ROUTING DECISION (<10ms target)
    // ========================================================================

    let routing_start = Instant::now();

    let selected_provider = state
        .route_engine
        .select_provider(&request.model)
        .map_err(|e| {
            error!("Routing failed: {}", e);
            ApiError::ProviderError("No available providers".to_string())
        })?;

    let routing_decision_ms = routing_start.elapsed().as_millis() as u64;

    debug!(
        selected_provider = ?selected_provider,
        routing_decision_ms = routing_decision_ms,
        "Routing decision completed"
    );

    if routing_decision_ms > 10 {
        error!(
            "Routing decision exceeded 10ms target: {}ms",
            routing_decision_ms
        );
    }

    // ========================================================================
    // STEP 3: CALL SELECTED PROVIDER
    // ========================================================================

    let connector_result = state
        .connector_manager
        .call(selected_provider, &request)
        .await
        .map_err(|e| {
            error!("Connector error: {}", e);

            // Record failure in cost tracker
            state.cost_tracker.record_error(
                request_id.clone(),
                selected_provider,
                request.model.clone(),
                e.to_string(),
                start.elapsed().as_millis() as u64,
                routing_decision_ms,
                None,
            );

            match e {
                connectors::ConnectorError::CircuitOpen => ApiError::CircuitOpen,
                connectors::ConnectorError::RateLimited => ApiError::RateLimited,
                connectors::ConnectorError::ServerError { status } => {
                    ApiError::ProviderError(format!("Provider returned error status {}", status))
                }
                _ => ApiError::ProviderError(e.to_string()),
            }
        })?;

    let response = connector_result.response.clone();
    let latency_ms = connector_result.latency_ms;
    let output_tokens = connector_result.output_tokens;

    // ========================================================================
    // STEP 4: VERIFY OUTPUT TOKENS
    // ========================================================================

    let response_text = response
        .choices
        .first()
        .map(|c| c.message.content.as_str())
        .unwrap_or("");

    if let Ok((counted, matches)) = tokens::verify_output_tokens(response_text, output_tokens) {
        debug!(
            reported = output_tokens,
            counted = counted,
            matches = matches,
            "Verified output tokens"
        );
    }

    // ========================================================================
    // STEP 5: CALCULATE COST WITH PRECISE TOKENS
    // ========================================================================

    let (cost_per_1m_input, cost_per_1m_output) = state
        .route_engine
        .get_pricing(&request.model) // <-- No comma, just the model!
        .map_err(|e| {
            error!("Pricing lookup failed: {}", e);
            ApiError::InternalError("Pricing lookup failed".to_string())
        })?;

    let token_cost =
        TokenCostBreakdown::new(input_tokens, output_tokens, cost_per_1m_input, cost_per_1m_output);

    // Calculate baseline cost (GPT-4o pricing: 250 input, 1000 output)
    let baseline_cost =
        TokenCostBreakdown::new(input_tokens, output_tokens, 250.0, 1000.0);

    let cost_saved = baseline_cost.total_cost_cents - token_cost.total_cost_cents;
    let savings_pct = (cost_saved / baseline_cost.total_cost_cents) * 100.0;

    debug!(
        cost_cents = token_cost.total_cost_cents,
        baseline_cents = baseline_cost.total_cost_cents,
        cost_saved_cents = cost_saved,
        savings_pct = savings_pct,
        "Calculated costs"
    );

    // ========================================================================
    // STEP 6: RECORD TO POSTGRES (non-blocking via tokio::spawn)
    // ========================================================================

    state.cost_tracker.record_request(
        request_id.clone(),
        selected_provider,
        response.model.clone(),
        &token_cost,
        baseline_cost.total_cost_cents,
        latency_ms,
        routing_decision_ms,
        None, // TODO: Extract from auth header
        None, // TODO: Extract from headers
        None, // TODO: Extract from headers
    );

    // ========================================================================
    // STEP 7: RETURN RESPONSE
    // ========================================================================

    let total_latency = start.elapsed().as_millis() as u64;

    info!(
        request_id = %request_id,
        provider = ?selected_provider,
        latency_ms = total_latency,
        cost_cents = token_cost.total_cost_cents,
        saved_cents = cost_saved,
        "Request completed successfully"
    );

    Ok(Json(response))
}

// ============================================================================
// HEALTH CHECK
// ============================================================================

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

// ============================================================================
// AUDIT REPORT
// ============================================================================

async fn audit_daily_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let date = params
        .get("date")
        .ok_or_else(|| ApiError::BadRequest("date parameter required".to_string()))?;

    let report = state
        .cost_tracker
        .get_daily_report(date)
        .await
        .map_err(|e| {
            error!("Failed to generate report: {}", e);
            ApiError::InternalError("Report generation failed".to_string())
        })?;

    Ok(Json(serde_json::to_value(report).unwrap()))
}

// ============================================================================
// MAIN SERVER
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(true)
        .with_level(true)
        .with_thread_ids(true)
        .init();

    info!("Starting RouteFuel v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    dotenv::dotenv().ok();

    let openai_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable not set");
    let deepseek_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    let gemini_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL environment variable not set");

    info!("Configuration loaded from environment");

    // Initialize database
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await?;

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await?;

    info!("Database migrations completed");

    // Initialize components
    let circuit_breaker = Arc::new(CircuitBreaker::new());
    let route_engine = Arc::new(RouteEngine::new());
    let cost_tracker = Arc::new(CostTracker::new(pool));

    // Initialize connector manager with key strings & circuit breaker
    let connector_manager = Arc::new(ConnectorManager::new(
        openai_key,
        anthropic_key,
        deepseek_key,
        gemini_key,
        Arc::clone(&circuit_breaker),
    ));

    info!("Components initialized");

    // Application state
    let state = AppState {
        route_engine,
        connector_manager,
        cost_tracker,
        circuit_breaker,
    };

    // Build router
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/health", axum::routing::get(health_handler))
        .route("/audit/daily", axum::routing::get(audit_daily_handler))
        .with_state(state)
        .layer(DefaultBodyLimit::max(1024 * 1024 * 10)) // 10MB limit
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(60)));

    // Start server
    let addr = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let bind_addr = format!("{}:{}", addr, port);

    let listener = TcpListener::bind(&bind_addr).await?;

    info!("Server listening on http://{}", bind_addr);
    info!("OpenAI-compatible endpoint: POST /v1/chat/completions");

    axum::serve(listener, app).await?;

    Ok(())
}