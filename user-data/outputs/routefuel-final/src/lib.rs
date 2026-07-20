// ============================================================================
// src/lib.rs
// Public module exports and re-exports
// ============================================================================

pub mod circuit_breaker;
pub mod connectors;
pub mod cost_tracker;
pub mod rate_limiter;
pub mod route_engine;
pub mod tokens;

// Re-export commonly used types
pub use circuit_breaker::CircuitBreaker;
pub use connectors::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ConnectorManager, Provider};
pub use cost_tracker::CostTracker;
pub use rate_limiter::RateLimiter;
pub use route_engine::RouteEngine;
pub use tokens::{TokenCostBreakdown, count_request_tokens, estimate_output_tokens};
