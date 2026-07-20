use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use serde::{Deserialize, Serialize};
use crate::route_engine::{
    RouteEngine, RoutingContext, RoutingDecision, RouteError, RouteResult, UserTier, RequestPriority,
};
use crate::connectors::{ConnectorManager, LlmRequest, LlmResponse, ConnectorError};
use crate::circuit_breaker::CircuitBreaker;

// ============================================================================
// REQUEST/RESPONSE TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRequest {
    /// Unique request ID for tracking
    pub request_id: String,

    /// The user's prompt
    pub prompt: String,

    /// Estimated token count (if known)
    pub estimated_tokens: Option<u32>,

    /// Max tokens for the response
    pub max_tokens: Option<u32>,

    /// Temperature for generation
    pub temperature: Option<f32>,

    /// User tier
    pub user_tier: UserTier,

    /// Request priority
    pub priority: RequestPriority,

    /// Enable shadow mode for this request
    pub shadow_mode: bool,

    /// Timeout in milliseconds
    pub timeout_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingResponse {
    pub request_id: String,
    pub decision: RoutingDecision,
    pub llm_response: Option<LlmResponse>,
    pub processing_time_ms: u32,
    pub success: bool,
    pub error: Option<String>,
    pub failover_attempted: bool,
    pub models_tried: Vec<String>,
}

impl RoutingResponse {
    pub fn from_error(request_id: String, error: RouteError, elapsed_ms: u32) -> Self {
        Self {
            request_id,
            decision: RoutingDecision {
                selected_model_id: String::new(),
                confidence: 0.0,
                reason: String::new(),
                shadow_model_id: None,
                metadata: crate::route_engine::RoutingMetadata {
                    scores: Default::default(),
                    cost_analysis: crate::route_engine::CostAnalysis {
                        selected_model_cost: 0.0,
                        baseline_cost: 0.0,
                        potential_savings: 0.0,
                        savings_percentage: 0.0,
                        cost_breakdown: crate::route_engine::CostBreakdown {
                            input_token_cost: 0.0,
                            output_token_cost: 0.0,
                            total_cost: 0.0,
                        },
                    },
                    routing_latency_ms: elapsed_ms,
                },
            },
            llm_response: None,
            processing_time_ms: elapsed_ms,
            success: false,
            error: Some(error.to_string()),
            failover_attempted: false,
            models_tried: vec![],
        }
    }
}

// ============================================================================
// CONCURRENCY MANAGER
// ============================================================================

/// Manages concurrent routing requests with semaphore-based backpressure
pub struct ConcurrencyManager {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl ConcurrencyManager {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("semaphore poisoned")
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

// ============================================================================
// ASYNC REQUEST HANDLER
// ============================================================================

pub struct AsyncRequestHandler {
    engine: Arc<RouteEngine>,
    connectors: Arc<ConnectorManager>,
    circuit_breaker: Arc<CircuitBreaker>,
    concurrency_mgr: ConcurrencyManager,
    request_timeout_ms: u32,
    enable_failover: bool,
}

impl AsyncRequestHandler {
    pub fn new(
        engine: Arc<RouteEngine>,
        connectors: Arc<ConnectorManager>,
        circuit_breaker: Arc<CircuitBreaker>,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            engine,
            connectors,
            circuit_breaker,
            concurrency_mgr: ConcurrencyManager::new(max_concurrent_requests),
            request_timeout_ms: 5000, // 5 second default timeout
            enable_failover: true,
        }
    }

    pub fn with_timeout(mut self, timeout_ms: u32) -> Self {
        self.request_timeout_ms = timeout_ms;
        self
    }

    pub fn disable_failover(mut self) -> Self {
        self.enable_failover = false;
        self
    }

    /// Process a single routing request with circuit breaker and automatic failover
    pub async fn handle_request(&self, req: RoutingRequest) -> RoutingResponse {
        let start = Instant::now();
        let request_id = req.request_id.clone();

        // Acquire concurrency permit (applies backpressure)
        let _permit = self.concurrency_mgr.acquire().await;

        // STEP 1: Route the request (<10ms overhead for routing decision)
        let routing_start = Instant::now();
        let context = RoutingContext {
            prompt: req.prompt.clone(),
            estimated_tokens: req.estimated_tokens.unwrap_or(250),
            user_tier: req.user_tier,
            priority: req.priority,
            shadow_mode: req.shadow_mode,
        };

        let routing_decision = match tokio::time::timeout(
            std::time::Duration::from_millis(10), // <10ms for routing
            self.engine.route(context),
        )
        .await
        {
            Ok(Ok(decision)) => decision,
            Ok(Err(e)) => {
                let elapsed_ms = start.elapsed().as_millis() as u32;
                return RoutingResponse::from_error(request_id, e, elapsed_ms);
            }
            Err(_) => {
                let elapsed_ms = start.elapsed().as_millis() as u32;
                return RoutingResponse::from_error(
                    request_id,
                    RouteError::Unknown("Routing timeout".to_string()),
                    elapsed_ms,
                );
            }
        };

        let routing_latency = routing_start.elapsed().as_millis() as u32;
        tracing::info!(
            "Routing decision made in {}ms: {}",
            routing_latency,
            routing_decision.selected_model_id
        );

        // STEP 2: Get model config for the selected model
        let model_config = match self.engine.get_model(&routing_decision.selected_model_id).await {
            Ok(config) => config,
            Err(e) => {
                let elapsed_ms = start.elapsed().as_millis() as u32;
                return RoutingResponse::from_error(request_id, e, elapsed_ms);
            }
        };

        // STEP 3: Build LLM request
        let llm_request = LlmRequest {
            prompt: req.prompt.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            model: model_config.id.clone(),
        };

        // STEP 4: Check circuit breaker before calling primary model
        let mut models_tried = vec![model_config.id.clone()];
        let llm_response = if let Err(_) = self.circuit_breaker.is_available(&model_config.provider).await {
            tracing::warn!(
                "Circuit breaker open for {:?}, skipping to failover",
                model_config.provider
            );
            Err(ConnectorError::Unknown(
                "Circuit breaker open".to_string(),
            ))
        } else {
            // Attempt to call the primary model
            let call_start = Instant::now();
            let result = self
                .connectors
                .complete(&model_config.provider, &llm_request)
                .await;

            match &result {
                Ok(response) => {
                    // Record success in circuit breaker
                    self.circuit_breaker
                        .record_success(&model_config.provider, response.latency_ms)
                        .await;
                }
                Err(_) => {
                    // Record failure in circuit breaker
                    self.circuit_breaker
                        .record_failure(&model_config.provider)
                        .await;
                }
            }

            result
        };

        // STEP 5: Handle failover if primary fails (<15ms for failover decision)
        let failover_start = Instant::now();
        let (final_response, failover_attempted) = match llm_response {
            Ok(response) => (Ok(response), false),
            Err(primary_error) => {
                tracing::warn!(
                    "Primary model {} failed: {}",
                    model_config.id,
                    primary_error
                );

                if !self.enable_failover {
                    (Err(primary_error), false)
                } else {
                    // Try failover to next best model (<15ms)
                    let failover_result = tokio::time::timeout(
                        std::time::Duration::from_millis(15),
                        self.try_failover(&routing_decision, &req, &mut models_tried),
                    )
                    .await;

                    match failover_result {
                        Ok(Ok(response)) => (Ok(response), true),
                        Ok(Err(e)) => (Err(e), true),
                        Err(_) => (
                            Err(ConnectorError::Unknown("Failover timeout".to_string())),
                            true,
                        ),
                    }
                }
            }
        };

        let failover_latency = failover_start.elapsed().as_millis() as u32;
        if failover_attempted {
            tracing::info!("Failover completed in {}ms", failover_latency);
        }

        let elapsed_ms = start.elapsed().as_millis() as u32;

        // STEP 6: Build response
        match final_response {
            Ok(llm_resp) => RoutingResponse {
                request_id,
                decision: routing_decision,
                llm_response: Some(llm_resp),
                processing_time_ms: elapsed_ms,
                success: true,
                error: None,
                failover_attempted,
                models_tried,
            },
            Err(e) => {
                let mut error_response = RoutingResponse::from_error(
                    request_id,
                    RouteError::Unknown(format!("LLM call failed: {}", e)),
                    elapsed_ms,
                );
                error_response.failover_attempted = failover_attempted;
                error_response.models_tried = models_tried;
                error_response
            }
        }
    }

    /// Try failover to the next best model based on scores
    async fn try_failover(
        &self,
        original_decision: &RoutingDecision,
        req: &RoutingRequest,
        models_tried: &mut Vec<String>,
    ) -> Result<LlmResponse, ConnectorError> {
        // Get sorted list of models by score (excluding already tried)
        let mut candidates: Vec<_> = original_decision
            .metadata
            .scores
            .iter()
            .filter(|(model_id, _)| !models_tried.contains(model_id))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Try up to 2 failover attempts
        for (model_id, _score) in candidates.iter().take(2) {
            tracing::info!("Attempting failover to model: {}", model_id);
            models_tried.push((*model_id).clone());

            let model_config = match self.engine.get_model(model_id).await {
                Ok(config) => config,
                Err(_) => continue,
            };

            // Check if provider is available
            if !self.connectors.is_provider_available(&model_config.provider) {
                tracing::warn!("Provider {:?} not available, skipping", model_config.provider);
                continue;
            }

            let llm_request = LlmRequest {
                prompt: req.prompt.clone(),
                max_tokens: req.max_tokens,
                temperature: req.temperature,
                model: model_config.id.clone(),
            };

            match self
                .connectors
                .complete(&model_config.provider, &llm_request)
                .await
            {
                Ok(response) => {
                    tracing::info!("Failover successful with model: {}", model_id);
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!("Failover attempt with {} failed: {}", model_id, e);
                    continue;
                }
            }
        }

        Err(ConnectorError::Unknown(
            "All failover attempts exhausted".to_string(),
        ))
    }

    /// Process multiple requests concurrently
    pub async fn handle_batch(
        &self,
        requests: Vec<RoutingRequest>,
    ) -> Vec<RoutingResponse> {
        let futures: Vec<_> = requests
            .into_iter()
            .map(|req| self.handle_request(req))
            .collect();

        futures::future::join_all(futures).await
    }

    /// Stream-based request handler for high-throughput scenarios
    pub fn create_stream_handler(
        self: Arc<Self>,
        buffer_size: usize,
    ) -> (
        mpsc::Sender<RoutingRequest>,
        JoinHandle<Vec<RoutingResponse>>,
    ) {
        let (tx, mut rx) = mpsc::channel(buffer_size);
        let handler = self.clone();

        let join_handle = tokio::spawn(async move {
            let mut responses = Vec::new();

            while let Some(request) = rx.recv().await {
                let response = handler.handle_request(request).await;
                responses.push(response);
            }

            responses
        });

        (tx, join_handle)
    }

    /// Get concurrency manager stats
    pub fn stats(&self) -> ConcurrencyStats {
        ConcurrencyStats {
            available_permits: self.concurrency_mgr.available_permits(),
            max_concurrent: self.concurrency_mgr.max_concurrent(),
            utilization_percent: {
                let used = self.concurrency_mgr.max_concurrent()
                    - self.concurrency_mgr.available_permits();
                (used as f64 / self.concurrency_mgr.max_concurrent() as f64) * 100.0
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencyStats {
    pub available_permits: usize,
    pub max_concurrent: usize,
    pub utilization_percent: f64,
}

// ============================================================================
// BATCH PROCESSOR WITH SHADOW MODE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModeResult {
    pub primary_model_id: String,
    pub shadow_model_id: String,
    pub primary_cost: f64,
    pub shadow_cost: f64,
    pub cost_difference: f64,
}

pub struct BatchProcessor {
    handler: Arc<AsyncRequestHandler>,
    shadow_mode_enabled: bool,
}

impl BatchProcessor {
    pub fn new(handler: Arc<AsyncRequestHandler>) -> Self {
        Self {
            handler,
            shadow_mode_enabled: false,
        }
    }

    pub fn enable_shadow_mode(mut self) -> Self {
        self.shadow_mode_enabled = true;
        self
    }

    /// Process requests and collect shadow mode results if enabled
    pub async fn process_batch_with_analytics(
        &self,
        requests: Vec<RoutingRequest>,
    ) -> BatchAnalytics {
        let responses = self.handler.handle_batch(requests).await;

        let total_requests = responses.len();
        let successful = responses.iter().filter(|r| r.success).count();
        let failed = total_requests - successful;

        let total_processing_time: u32 = responses.iter().map(|r| r.processing_time_ms).sum();
        let avg_processing_time = if successful > 0 {
            total_processing_time / successful as u32
        } else {
            0
        };

        let total_savings: f64 = responses
            .iter()
            .filter(|r| r.success)
            .map(|r| r.decision.metadata.cost_analysis.potential_savings)
            .sum();

        let shadow_results = if self.shadow_mode_enabled {
            responses
                .iter()
                .filter_map(|r| {
                    r.decision.shadow_model_id.as_ref().map(|shadow_id| {
                        ShadowModeResult {
                            primary_model_id: r.decision.selected_model_id.clone(),
                            shadow_model_id: shadow_id.clone(),
                            primary_cost: r.decision.metadata.cost_analysis.selected_model_cost,
                            shadow_cost: r.decision.metadata.cost_analysis.baseline_cost,
                            cost_difference: (r.decision.metadata.cost_analysis.baseline_cost
                                - r.decision.metadata.cost_analysis.selected_model_cost)
                                .abs(),
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        BatchAnalytics {
            total_requests,
            successful,
            failed,
            avg_processing_time_ms: avg_processing_time,
            total_potential_savings: total_savings,
            responses,
            shadow_results,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchAnalytics {
    pub total_requests: usize,
    pub successful: usize,
    pub failed: usize,
    pub avg_processing_time_ms: u32,
    pub total_potential_savings: f64,
    pub responses: Vec<RoutingResponse>,
    pub shadow_results: Vec<ShadowModeResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_engine::ModelConfig;
    use crate::route_engine::ModelProvider;

    async fn setup_handler() -> (Arc<AsyncRequestHandler>, Arc<RouteEngine>) {
        use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};

        let engine = Arc::new(RouteEngine::new("gpt-4o".to_string()));

        let gpt4 = ModelConfig {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            provider: ModelProvider::OpenAI,
            cost_per_1m_input_tokens: 3.0,
            cost_per_1m_output_tokens: 6.0,
            latency_ms: 200,
            throughput: 5.0,
            context_window: 128000,
            enabled: true,
            quality_score: 1.0,
        };

        engine.register_model(gpt4).await.unwrap();

        let connectors = Arc::new(ConnectorManager::new(None, None, None));
        let circuit_breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default()));
        let handler = Arc::new(AsyncRequestHandler::new(
            engine.clone(),
            connectors,
            circuit_breaker,
            10,
        ));
        (handler, engine)
    }

    #[tokio::test]
    async fn test_single_request_handling() {
        let (handler, _) = setup_handler().await;

        let request = RoutingRequest {
            request_id: "test-1".to_string(),
            prompt: "Test prompt".to_string(),
            estimated_tokens: Some(100),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            user_tier: UserTier::Pro,
            priority: RequestPriority::Balanced,
            shadow_mode: false,
            timeout_ms: 5000,
        };

        let response = handler.handle_request(request).await;
        assert_eq!(response.request_id, "test-1");
        // Note: Will fail without real API keys, but routing decision will succeed
    }

    #[tokio::test]
    async fn test_concurrency_backpressure() {
        let (handler, _) = setup_handler().await;

        let initial_permits = handler.concurrency_mgr.available_permits();
        assert_eq!(initial_permits, 10);

        let request = RoutingRequest {
            request_id: "test-1".to_string(),
            prompt: "Test".to_string(),
            estimated_tokens: Some(100),
            max_tokens: None,
            temperature: None,
            user_tier: UserTier::Free,
            priority: RequestPriority::Cost,
            shadow_mode: false,
            timeout_ms: 1000,
        };

        let _response = handler.handle_request(request).await;
        // Permits should be returned after completion
        assert_eq!(handler.concurrency_mgr.available_permits(), initial_permits);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let (handler, _) = setup_handler().await;

        let requests = vec![
            RoutingRequest {
                request_id: "batch-1".to_string(),
                prompt: "Request 1".to_string(),
                estimated_tokens: Some(100),
                max_tokens: None,
                temperature: None,
                user_tier: UserTier::Pro,
                priority: RequestPriority::Cost,
                shadow_mode: false,
                timeout_ms: 5000,
            },
            RoutingRequest {
                request_id: "batch-2".to_string(),
                prompt: "Request 2".to_string(),
                estimated_tokens: Some(200),
                max_tokens: None,
                temperature: None,
                user_tier: UserTier::Free,
                priority: RequestPriority::Balanced,
                shadow_mode: false,
                timeout_ms: 5000,
            },
        ];

        let responses = handler.handle_batch(requests).await;
        assert_eq!(responses.len(), 2);
    }
}
