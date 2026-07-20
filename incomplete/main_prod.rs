use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    response::{IntoResponse, Json},
    routing::{post, get},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;
use sha2::{Sha256, Digest};

use routefuel::{
    AsyncRequestHandler, ConnectorManager, CostTracker, ModelConfig, ModelProvider,
    RequestPriority, RouteEngine, RoutingRequest, RoutingResponse, UserTier,
    CircuitBreaker, CircuitBreakerConfig,
    RateLimiter, ClientConfig,
    TelemetryRecorder, TelemetryData,
};

// ============================================================================
// APPLICATION STATE
// ============================================================================

#[derive(Clone)]
struct AppState {
    handler: Arc<AsyncRequestHandler>,
    tracker: Arc<CostTracker>,
    telemetry: Arc<TelemetryRecorder>,
    rate_limiter: Arc<RateLimiter>,
}

// ============================================================================
// API REQUEST/RESPONSE TYPES
// ============================================================================

#[derive(Debug, Deserialize)]
struct RouteApiRequest {
    prompt: String,
    #[serde(default)]
    requirements: RouteRequirements,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    user_tier: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RouteRequirements {
    #[serde(default = "default_priority")]
    priority: String,
}

fn default_priority() -> String {
    "balanced".to_string()
}

#[derive(Debug, Serialize)]
struct RouteApiResponse {
    request_id: String,
    model_used: String,
    response: String,
    cost_analysis: CostAnalysisResponse,
    metadata: MetadataResponse,
}

#[derive(Debug, Serialize)]
struct CostAnalysisResponse {
    selected_model_cost_usd: f64,
    baseline_cost_usd: f64,
    savings_usd: f64,
    savings_percentage: f64,
    tokens_used: TokensUsed,
}

#[derive(Debug, Serialize)]
struct TokensUsed {
    input: u32,
    output: u32,
    total: u32,
}

#[derive(Debug, Serialize)]
struct MetadataResponse {
    routing_latency_ms: u32,
    total_processing_time_ms: u32,
    failover_attempted: bool,
    models_tried: Vec<String>,
    confidence: f64,
    reason: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    request_id: Option<String>,
    error_code: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_secs: u64,
    circuit_breakers: CircuitBreakerStatus,
}

#[derive(Debug, Serialize)]
struct CircuitBreakerStatus {
    openai: String,
    anthropic: String,
    deepseek: String,
}

#[derive(Debug, Serialize)]
struct ClientStatsResponse {
    client_id: String,
    requests_this_minute: u32,
    requests_per_minute_limit: u32,
    requests_this_hour: u32,
    requests_per_hour_limit: u32,
    concurrent_requests: u32,
    max_concurrent: u32,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn hash_api_key(api_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

// ============================================================================
// MAIN APPLICATION ENTRY POINT
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,routefuel=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let start_time = std::time::Instant::now();

    tracing::info!("🚀 Starting RouteFuel Server - Enterprise Edition");

    // ========================================================================
    // LOAD API KEYS FROM ENVIRONMENT
    // ========================================================================

    let openai_key = std::env::var("OPENAI_API_KEY").ok();
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let deepseek_key = std::env::var("DEEPSEEK_API_KEY").ok();

    if openai_key.is_none() && anthropic_key.is_none() && deepseek_key.is_none() {
        tracing::warn!("⚠️  No API keys found in environment. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY");
        tracing::warn!("⚠️  Server will start but LLM calls will fail");
    } else {
        if openai_key.is_some() {
            tracing::info!("✓ OpenAI API key loaded");
        }
        if anthropic_key.is_some() {
            tracing::info!("✓ Anthropic API key loaded");
        }
        if deepseek_key.is_some() {
            tracing::info!("✓ DeepSeek API key loaded");
        }
    }

    // ========================================================================
    // INITIALIZE ROUTING ENGINE
    // ========================================================================

    let engine = Arc::new(RouteEngine::new("gpt-4o".to_string()));

    // Register models
    let models = vec![
        ModelConfig {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            provider: ModelProvider::OpenAI,
            cost_per_1m_input_tokens: 250.0,  // $2.50 per 1M input tokens in cents
            cost_per_1m_output_tokens: 1000.0, // $10.00 per 1M output tokens in cents
            latency_ms: 200,
            throughput: 5.0,
            context_window: 128000,
            enabled: openai_key.is_some(),
            quality_score: 1.0,
        },
        ModelConfig {
            id: "gpt-4-turbo".to_string(),
            name: "GPT-4 Turbo".to_string(),
            provider: ModelProvider::OpenAI,
            cost_per_1m_input_tokens: 1000.0, // $10 per 1M
            cost_per_1m_output_tokens: 3000.0, // $30 per 1M
            latency_ms: 180,
            throughput: 6.0,
            context_window: 128000,
            enabled: openai_key.is_some(),
            quality_score: 0.98,
        },
        ModelConfig {
            id: "claude-3-5-sonnet-20241022".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            provider: ModelProvider::Anthropic,
            cost_per_1m_input_tokens: 300.0,  // $3 per 1M
            cost_per_1m_output_tokens: 1500.0, // $15 per 1M
            latency_ms: 220,
            throughput: 4.5,
            context_window: 200000,
            enabled: anthropic_key.is_some(),
            quality_score: 1.0,
        },
        ModelConfig {
            id: "deepseek-chat".to_string(),
            name: "DeepSeek V3".to_string(),
            provider: ModelProvider::DeepSeek,
            cost_per_1m_input_tokens: 14.0,   // $0.14 per 1M in cents
            cost_per_1m_output_tokens: 28.0,  // $0.28 per 1M in cents
            latency_ms: 150,
            throughput: 8.0,
            context_window: 64000,
            enabled: deepseek_key.is_some(),
            quality_score: 0.95,
        },
    ];

    engine.register_models(models.clone()).await?;
    tracing::info!("✓ Registered {} models", models.len());

    // ========================================================================
    // INITIALIZE CIRCUIT BREAKER
    // ========================================================================

    let circuit_breaker_config = CircuitBreakerConfig {
        failure_threshold: 3,
        reset_timeout: Duration::from_secs(60),
        failure_rate_threshold: 0.5,
        minimum_requests: 10,
    };

    let circuit_breaker = Arc::new(CircuitBreaker::new(circuit_breaker_config));
    tracing::info!("✓ Circuit breaker initialized (60s reset, 3 failures threshold)");

    // ========================================================================
    // INITIALIZE CONNECTORS
    // ========================================================================

    let connectors = Arc::new(ConnectorManager::new(openai_key, anthropic_key, deepseek_key));

    // ========================================================================
    // INITIALIZE RATE LIMITER AND REGISTER CLIENTS
    // ========================================================================

    let rate_limiter = Arc::new(RateLimiter::new());

    // Load client configs from environment or use defaults
    let clients = vec![
        ClientConfig::new(
            "pilot-client-1".to_string(),
            hash_api_key("pk_test_pilot_1"),
            "Pilot Client 1".to_string(),
            "pro".to_string(),
        ),
        ClientConfig::new(
            "pilot-client-2".to_string(),
            hash_api_key("pk_test_pilot_2"),
            "Pilot Client 2".to_string(),
            "free".to_string(),
        ),
    ];

    for client in clients {
        rate_limiter.register_client(client).await?;
    }

    tracing::info!("✓ Rate limiter initialized with pilot clients");

    // ========================================================================
    // INITIALIZE TELEMETRY
    // ========================================================================

    let telemetry_dir = std::env::var("TELEMETRY_DIR").unwrap_or_else(|_| "./telemetry".to_string());
    let telemetry = Arc::new(TelemetryRecorder::new(&telemetry_dir, 100)?);
    tracing::info!("✓ Telemetry recorder initialized (directory: {})", telemetry_dir);

    // ========================================================================
    // INITIALIZE ASYNC HANDLER AND COST TRACKER
    // ========================================================================

    let max_concurrent = std::env::var("MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let handler = Arc::new(AsyncRequestHandler::new(
        engine.clone(),
        connectors,
        circuit_breaker.clone(),
        max_concurrent,
    ));

    let tracker = Arc::new(CostTracker::new(10000));

    tracing::info!("✓ Max concurrent requests: {}", max_concurrent);

    // ========================================================================
    // BUILD AXUM APPLICATION
    // ========================================================================

    let app_state = AppState {
        handler,
        tracker,
        telemetry,
        rate_limiter,
    };

    let app = Router::new()
        .route("/v1/route", post(route_handler))
        .route("/health", get(health_handler))
        .route("/v1/health", get(health_handler))
        .route("/v1/stats/:client_id", get(stats_handler))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(Duration::from_secs(60)))
        .with_state(app_state);

    // ========================================================================
    // START SERVER
    // ========================================================================

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("🎯 Server listening on http://{}", addr);
    tracing::info!("📡 Primary endpoint: POST /v1/route");
    tracing::info!("💚 Health check: GET /health");
    tracing::info!("📊 Client stats: GET /v1/stats/{{client_id}}");
    tracing::info!("✅ Ready to accept requests!");

    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// ROUTE HANDLER WITH FULL ENTERPRISE FEATURES
// ============================================================================

async fn route_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<RouteApiRequest>,
) -> impl IntoResponse {
    let request_id = Uuid::new_v4().to_string();

    // ========================================================================
    // STEP 1: AUTHENTICATE CLIENT VIA API KEY
    // ========================================================================

    let api_key = match extract_api_key(&headers) {
        Some(key) => key,
        None => {
            tracing::warn!("Request {} missing X-API-Key header", request_id);
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Missing X-API-Key header".to_string(),
                    request_id: Some(request_id),
                    error_code: "MISSING_API_KEY".to_string(),
                }),
            )
                .into_response();
        }
    };

    let api_key_hash = hash_api_key(&api_key);

    let client_id = match state.rate_limiter.authenticate(&api_key_hash).await {
        Ok(id) => id,
        Err(_) => {
            tracing::warn!("Request {} invalid API key", request_id);
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Invalid API key".to_string(),
                    request_id: Some(request_id),
                    error_code: "INVALID_API_KEY".to_string(),
                }),
            )
                .into_response();
        }
    };

    tracing::info!(
        "Routing request {} from client {} with priority: {}",
        request_id,
        client_id,
        payload.requirements.priority
    );

    // ========================================================================
    // STEP 2: CHECK RATE LIMITS
    // ========================================================================

    if let Err(rate_limit_err) = state.rate_limiter.check_rate_limit(&client_id).await {
        tracing::warn!(
            "Request {} rate limited: {}",
            request_id,
            rate_limit_err
        );

        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: rate_limit_err.to_string(),
                request_id: Some(request_id),
                error_code: "RATE_LIMIT_EXCEEDED".to_string(),
            }),
        )
            .into_response();
    }

    // ========================================================================
    // STEP 3: BUILD ROUTING REQUEST
    // ========================================================================

    let priority = match payload.requirements.priority.to_lowercase().as_str() {
        "cost" => RequestPriority::Cost,
        "quality" => RequestPriority::Quality,
        "balanced" | _ => RequestPriority::Balanced,
    };

    let user_tier = match payload.user_tier.as_deref() {
        Some("free") => UserTier::Free,
        Some("enterprise") => UserTier::Enterprise,
        Some("pro") | _ => UserTier::Pro,
    };

    let estimated_tokens = (payload.prompt.len() / 4) as u32;

    let routing_request = RoutingRequest {
        request_id: request_id.clone(),
        prompt: payload.prompt.clone(),
        estimated_tokens: Some(estimated_tokens),
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        user_tier,
        priority,
        shadow_mode: false,
        timeout_ms: 30000, // 30 second timeout
    };

    // ========================================================================
    // STEP 4: EXECUTE ROUTING WITH CIRCUIT BREAKER
    // ========================================================================

    let response: RoutingResponse = state.handler.handle_request(routing_request).await;

    // ========================================================================
    // STEP 5: RELEASE RATE LIMIT PERMIT
    // ========================================================================

    state.rate_limiter.release_request(&client_id).await;

    // ========================================================================
    // STEP 6: RECORD TELEMETRY
    // ========================================================================

    if response.success {
        if let Some(llm_resp) = &response.llm_response {
            let mut telemetry_data = TelemetryData::new(
                request_id.clone(),
                response.decision.selected_model_id.clone(),
                response.decision.metadata.cost_analysis.selected_model_cost,
                response.decision.metadata.cost_analysis.baseline_cost,
                client_id.clone(),
            );

            telemetry_data.latency_ms = llm_resp.latency_ms;
            telemetry_data.priority = format!("{:?}", priority);
            telemetry_data.user_tier = format!("{:?}", user_tier);
            telemetry_data.failover_triggered = response.failover_attempted;
            telemetry_data.provider = format!("{:?}", llm_resp.provider);
            telemetry_data.success = true;
            telemetry_data.tokens_used = llm_resp.tokens_used.total_tokens;

            let _ = state.telemetry.record(telemetry_data).await;

            // Also record in cost tracker
            let _ = state
                .tracker
                .record_decision(
                    request_id.clone(),
                    response.decision.selected_model_id.clone(),
                    response.decision.metadata.cost_analysis.selected_model_cost,
                    response.decision.metadata.cost_analysis.baseline_cost,
                    llm_resp.tokens_used.total_tokens,
                    format!("{:?}", user_tier),
                )
                .await;
        }
    } else {
        // Record failure in telemetry
        let mut telemetry_data = TelemetryData::new(
            request_id.clone(),
            "FAILED".to_string(),
            0.0,
            0.0,
            client_id.clone(),
        );
        telemetry_data.success = false;
        telemetry_data.error = response.error.clone();

        let _ = state.telemetry.record(telemetry_data).await;
    }

    // ========================================================================
    // STEP 7: BUILD AND RETURN RESPONSE
    // ========================================================================

    if response.success {
        let llm_response = response.llm_response.as_ref().unwrap();

        let api_response = RouteApiResponse {
            request_id: response.request_id,
            model_used: response.decision.selected_model_id.clone(),
            response: llm_response.text.clone(),
            cost_analysis: CostAnalysisResponse {
                selected_model_cost_usd: response.decision.metadata.cost_analysis.selected_model_cost
                    / 100.0,
                baseline_cost_usd: response.decision.metadata.cost_analysis.baseline_cost / 100.0,
                savings_usd: response.decision.metadata.cost_analysis.potential_savings / 100.0,
                savings_percentage: response.decision.metadata.cost_analysis.savings_percentage,
                tokens_used: TokensUsed {
                    input: llm_response.tokens_used.input_tokens,
                    output: llm_response.tokens_used.output_tokens,
                    total: llm_response.tokens_used.total_tokens,
                },
            },
            metadata: MetadataResponse {
                routing_latency_ms: response.decision.metadata.routing_latency_ms,
                total_processing_time_ms: response.processing_time_ms,
                failover_attempted: response.failover_attempted,
                models_tried: response.models_tried,
                confidence: response.decision.confidence,
                reason: response.decision.reason,
            },
        };

        (StatusCode::OK, Json(api_response)).into_response()
    } else {
        let error_response = ErrorResponse {
            error: response.error.unwrap_or_else(|| "Unknown error".to_string()),
            request_id: Some(response.request_id),
            error_code: if response.failover_attempted {
                "FAILOVER_EXHAUSTED".to_string()
            } else {
                "LLM_ERROR".to_string()
            },
        };

        (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
    }
}

// ============================================================================
// HEALTH CHECK HANDLER
// ============================================================================

static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let start = START_TIME.get_or_init(std::time::Instant::now);
    let uptime = start.elapsed().as_secs();

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        circuit_breakers: CircuitBreakerStatus {
            openai: format!("{:?}", state.handler),
            anthropic: format!("{:?}", state.handler),
            deepseek: format!("{:?}", state.handler),
        },
    };

    (StatusCode::OK, Json(response))
}

// ============================================================================
// CLIENT STATS HANDLER
// ============================================================================

async fn stats_handler(
    State(state): State<AppState>,
    axum::extract::Path(client_id): axum::extract::Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Verify client is authenticated
    let api_key = match extract_api_key(&headers) {
        Some(key) => key,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Missing X-API-Key header".to_string(),
                    request_id: None,
                    error_code: "MISSING_API_KEY".to_string(),
                }),
            )
                .into_response();
        }
    };

    let api_key_hash = hash_api_key(&api_key);
    let authenticated_client = match state.rate_limiter.authenticate(&api_key_hash).await {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Invalid API key".to_string(),
                    request_id: None,
                    error_code: "INVALID_API_KEY".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Ensure client can only view their own stats
    if authenticated_client != client_id {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Cannot access other client's stats".to_string(),
                request_id: None,
                error_code: "FORBIDDEN".to_string(),
            }),
        )
            .into_response();
    }

    match state.rate_limiter.get_client_stats(&client_id).await {
        Ok(stats) => {
            let response = ClientStatsResponse {
                client_id: stats.client_id,
                requests_this_minute: stats.requests_this_minute,
                requests_per_minute_limit: stats.requests_per_minute_limit,
                requests_this_hour: stats.requests_this_hour,
                requests_per_hour_limit: stats.requests_per_hour_limit,
                concurrent_requests: stats.concurrent_requests,
                max_concurrent: stats.max_concurrent,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
                request_id: None,
                error_code: "CLIENT_NOT_FOUND".to_string(),
            }),
        )
            .into_response(),
    }
}
