// ============================================================================
// src/connectors.rs
//
// Why each provider has its own connector:
//   OpenAI   → standard {model, messages, …} JSON, Bearer auth
//   DeepSeek → IDENTICAL schema to OpenAI, different base URL + Bearer auth
//              (literally just change the URL — cheapest win available)
//   Anthropic→ different JSON shape + x-api-key header, needs conversion
//   Gemini   → completely different REST + auth model, stub below
// ============================================================================

use crate::circuit_breaker::CircuitBreaker;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, error, instrument, warn};

// ============================================================================
// PROVIDER ENUM
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    DeepSeek,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Anthropic => write!(f, "anthropic"),
            Provider::OpenAI    => write!(f, "openai"),
            Provider::Gemini    => write!(f, "gemini"),
            Provider::DeepSeek  => write!(f, "deepseek"),
        }
    }
}

// ============================================================================
// ERROR
// ============================================================================

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Provider returned 5xx ({status})")]
    ServerError { status: u16 },
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Rate limited")]
    RateLimited,
    #[error("Timeout")]
    Timeout,
    #[error("Bad response: {0}")]
    BadResponse(String),
    #[error("Serialization: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Circuit breaker open")]
    CircuitOpen,
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl ConnectorError {
    pub fn trips_circuit(&self) -> bool {
        matches!(self, Self::ServerError { .. } | Self::Timeout | Self::Http(_))
    }
}

// ============================================================================
// OPENAI-COMPATIBLE TYPES
// These are used by OpenAI, DeepSeek (identical schema), and our API surface.
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// CONNECTOR TRAIT
// ============================================================================

#[derive(Debug, Clone)]
pub struct ConnectorResult {
    pub provider: Provider,
    pub model_id: String,
    pub response: ChatCompletionResponse,
    pub latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[async_trait]
pub trait Connector: Send + Sync {
    async fn complete(&self, req: &ChatCompletionRequest) -> Result<ConnectorResult, ConnectorError>;
    fn provider(&self) -> Provider;
}

// ============================================================================
// OPENAI CONNECTOR
// POST https://api.openai.com/v1/chat/completions
// ============================================================================

pub struct OpenAIConnector {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl OpenAIConnector {
    pub fn new(api_key: String, circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1/chat/completions".into(),
            client: build_client(),
            circuit_breaker,
        }
    }
}

#[async_trait]
impl Connector for OpenAIConnector {
    #[instrument(skip(self, req), fields(model = %req.model))]
    async fn complete(&self, req: &ChatCompletionRequest) -> Result<ConnectorResult, ConnectorError> {
        openai_compatible_call(
            &self.client,
            &self.base_url,
            &self.api_key,
            req,
            Provider::OpenAI,
            &self.circuit_breaker,
        ).await
    }

    fn provider(&self) -> Provider { Provider::OpenAI }
}

// ============================================================================
// DEEPSEEK CONNECTOR
// POST https://api.deepseek.com/v1/chat/completions
//
// DeepSeek uses the EXACT same JSON schema as OpenAI.
// We just swap the base URL and use a DeepSeek API key.
// This is the fastest new provider to add — literally 3 changed lines.
// ============================================================================

pub struct DeepSeekConnector {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl DeepSeekConnector {
    pub fn new(api_key: String, circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self {
            api_key,
            base_url: "https://api.deepseek.com/v1/chat/completions".into(),
            client: build_client(),
            circuit_breaker,
        }
    }
}

#[async_trait]
impl Connector for DeepSeekConnector {
    #[instrument(skip(self, req), fields(model = %req.model))]
    async fn complete(&self, req: &ChatCompletionRequest) -> Result<ConnectorResult, ConnectorError> {
        openai_compatible_call(
            &self.client,
            &self.base_url,
            &self.api_key,
            req,
            Provider::DeepSeek,
            &self.circuit_breaker,
        ).await
    }

    fn provider(&self) -> Provider { Provider::DeepSeek }
}

// ============================================================================
// ANTHROPIC CONNECTOR
// POST https://api.anthropic.com/v1/messages
// Different JSON — needs conversion to/from OpenAI format.
// ============================================================================

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VER: &str = "2023-06-01";

#[derive(Debug, Serialize)]
struct AnthropicReq<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResp {
    id: String,
    model: String,
    content: Vec<AnthropicBlock>,
    stop_reason: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

pub struct AnthropicConnector {
    api_key: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl AnthropicConnector {
    pub fn new(api_key: String, circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { api_key, client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for AnthropicConnector {
    #[instrument(skip(self, req), fields(model = %req.model))]
    async fn complete(&self, req: &ChatCompletionRequest) -> Result<ConnectorResult, ConnectorError> {
        let start = Instant::now();

        if self.circuit_breaker.is_open(Provider::Anthropic) {
            return Err(ConnectorError::CircuitOpen);
        }

        let body = AnthropicReq {
            model: &req.model,
            messages: &req.messages,
            max_tokens: req.max_tokens.unwrap_or(1024),
            temperature: req.temperature,
        };

        let http_resp = self.client
            .post(ANTHROPIC_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VER)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() { self.circuit_breaker.record_failure(Provider::Anthropic); }
                if e.is_timeout() { ConnectorError::Timeout } else { ConnectorError::Http(e) }
            })?;

        let status = http_resp.status().as_u16();
        let text   = http_resp.text().await?;

        match status {
            200..=299 => {
                let ar: AnthropicResp = serde_json::from_str(&text)?;
                let content = ar.content.iter()
                    .find(|b| b.kind == "text")
                    .and_then(|b| b.text.as_deref())
                    .unwrap_or("")
                    .to_owned();

                let response = ChatCompletionResponse {
                    id: ar.id,
                    object: "chat.completion".into(),
                    created: unix_now(),
                    model: ar.model.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage { role: "assistant".into(), content },
                        finish_reason: ar.stop_reason,
                    }],
                    usage: Usage {
                        prompt_tokens: ar.usage.input_tokens,
                        completion_tokens: ar.usage.output_tokens,
                        total_tokens: ar.usage.input_tokens + ar.usage.output_tokens,
                    },
                };

                self.circuit_breaker.record_success(Provider::Anthropic);
                Ok(ConnectorResult {
                    provider: Provider::Anthropic,
                    model_id: ar.model,
                    input_tokens: response.usage.prompt_tokens,
                    output_tokens: response.usage.completion_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                    response,
                })
            }
            401 => Err(ConnectorError::Unauthorized),
            429 => Err(ConnectorError::RateLimited),
            500..=599 => {
                self.circuit_breaker.record_failure(Provider::Anthropic);
                Err(ConnectorError::ServerError { status })
            }
            _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
        }
    }

    fn provider(&self) -> Provider { Provider::Anthropic }
}

// ============================================================================
// GEMINI CONNECTOR  (stub — wire up when you have a Gemini API key)
//
// Gemini API is fundamentally different:
//   URL:  POST https://generativelanguage.googleapis.com/v1beta/
//              models/{modelId}:generateContent?key={API_KEY}
//   Body: { "contents": [{ "role": "user", "parts": [{"text": "…"}] }] }
//   Resp: { "candidates": [{ "content": { "parts": [{"text": "…"}] } }] }
//
// To fully implement this:
//  1. Translate ChatMessage[] → Gemini contents[]
//  2. Map "system" role to systemInstruction field
//  3. Parse candidates[0].content.parts[0].text
//  4. Map usageMetadata.{promptTokenCount, candidatesTokenCount}
// ============================================================================

pub struct GeminiConnector {
    api_key: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl GeminiConnector {
    pub fn new(api_key: String, circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { api_key, client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for GeminiConnector {
    async fn complete(&self, _req: &ChatCompletionRequest) -> Result<ConnectorResult, ConnectorError> {
        // TODO: implement Gemini API call
        // See the comment block above for the mapping you need.
        Err(ConnectorError::NotImplemented(
            "GeminiConnector: translate ChatMessage[] → Gemini contents[] format, then call \
             https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent".into()
        ))
    }

    fn provider(&self) -> Provider { Provider::Gemini }
}

// ============================================================================
// CONNECTOR MANAGER
// ============================================================================

pub struct ConnectorManager {
    openai:    OpenAIConnector,
    anthropic: AnthropicConnector,
    deepseek:  DeepSeekConnector,
    gemini:    GeminiConnector,
}

impl ConnectorManager {
    pub fn new(
        openai_key:    String,
        anthropic_key: String,
        deepseek_key:  String,
        gemini_key:    String,
        cb: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            openai:    OpenAIConnector::new(openai_key, Arc::clone(&cb)),
            anthropic: AnthropicConnector::new(anthropic_key, Arc::clone(&cb)),
            deepseek:  DeepSeekConnector::new(deepseek_key, Arc::clone(&cb)),
            gemini:    GeminiConnector::new(gemini_key, cb),
        }
    }

    pub async fn call(
        &self,
        provider: Provider,
        req: &ChatCompletionRequest,
    ) -> Result<ConnectorResult, ConnectorError> {
        match provider {
            Provider::OpenAI    => self.openai.complete(req).await,
            Provider::Anthropic => self.anthropic.complete(req).await,
            Provider::DeepSeek  => self.deepseek.complete(req).await,
            Provider::Gemini    => self.gemini.complete(req).await,
        }
    }
}

// ============================================================================
// SHARED HELPERS
// ============================================================================

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Failed to build HTTP client")
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Shared call logic for OpenAI-compatible endpoints (OpenAI + DeepSeek).
async fn openai_compatible_call(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
) -> Result<ConnectorResult, ConnectorError> {
    let start = Instant::now();

    if cb.is_open(provider) {
        return Err(ConnectorError::CircuitOpen);
    }

    let http_resp = client
        .post(url)
        .bearer_auth(api_key)
        .json(req)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() { cb.record_failure(provider); ConnectorError::Timeout }
            else { ConnectorError::Http(e) }
        })?;

    let status = http_resp.status().as_u16();
    let text   = http_resp.text().await?;

    match status {
        200..=299 => {
            let resp: ChatCompletionResponse = serde_json::from_str(&text)?;
            cb.record_success(provider);
            Ok(ConnectorResult {
                provider,
                model_id: resp.model.clone(),
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                response: resp,
            })
        }
        401 => Err(ConnectorError::Unauthorized),
        429 => Err(ConnectorError::RateLimited),
        500..=599 => {
            cb.record_failure(provider);
            Err(ConnectorError::ServerError { status })
        }
        _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
    }
}
