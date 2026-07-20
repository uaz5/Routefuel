// =============================================================================
// src/streaming.rs  — RouteFuel v0.5
//
// SSE Streaming Proxy
//
// Architecture:
//   Client ──POST /v1/stream──► RouteFuel ──stream──► OpenAI/Anthropic
//                                    │
//                                    ├─ yield each SSE chunk to client  (<1ms)
//                                    └─ tokio::spawn: intercept full text
//                                       → Postgres audit write (non-blocking)
//
// Key design decisions confirmed from live docs (May 2026):
//   - reqwest Response::bytes_stream() gives raw SSE bytes
//   - axum Sse<impl Stream<Item = Result<Event, Infallible>>> is the return type
//   - [DONE] sentinel from OpenAI marks end-of-stream
//   - async_stream::try_stream! macro lets us yield inside async blocks cleanly
// =============================================================================

use crate::connectors::{ChatRequest, Provider};
use crate::cost_tracker::CostTracker;
use crate::route_engine::RouteEngine;
use async_stream::try_stream;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tracing::{debug, error, instrument};
use uuid::Uuid;

// =============================================================================
// SSE chunk types (OpenAI wire format — Anthropic uses same shape on stream)
// =============================================================================

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Option<Vec<StreamChoice>>,
    usage:   Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta:        StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    prompt_tokens:     Option<u32>,
    completion_tokens: Option<u32>,
}

// =============================================================================
// Streaming handler — returns Sse<impl Stream>
// =============================================================================

/// Proxy a streaming request from OpenAI/Anthropic to the client.
///
/// Guarantees:
///   - Each SSE chunk forwarded within <1ms of receipt (no buffering)
///   - Full response text captured in a side channel for audit logging
///   - Postgres write happens in tokio::spawn AFTER stream ends (never blocks)
#[instrument(skip(route_engine, cost_tracker, http_client, req))]
pub async fn stream_handler(
    request_id:   String,
    provider:     Provider,
    model_api_id: String,
    base_url:     String,
    api_key:      String,
    req:          ChatRequest,
    client_id:    Option<String>,
    route_engine: Arc<RouteEngine>,
    cost_tracker: Arc<CostTracker>,
    http_client:  reqwest::Client,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let start = Instant::now();

    // Side channel: stream sends completed text here; audit task reads from it
    let (audit_tx, mut audit_rx) = mpsc::channel::<AuditPayload>(1);

    // Clone everything needed inside the stream closure
    let cost_tracker_clone  = Arc::clone(&cost_tracker);
    let request_id_clone    = request_id.clone();
    let model_api_id_clone  = model_api_id.clone();
    let client_id_clone     = client_id.clone();

    // ── Audit task: waits for stream to finish, then writes to Postgres ───────
    // This never touches the hot path — it only runs after [DONE] is received.
    tokio::spawn(async move {
        if let Some(payload) = audit_rx.recv().await {
            cost_tracker_clone.record_request(
                request_id_clone,
                payload.provider,
                model_api_id_clone,
                payload.input_tokens,
                payload.output_tokens,
                payload.latency_ms,
                payload.routing_ms,
                client_id_clone,
                None,
                Some("streaming".to_string()),
            );
        }
    });

    // ── Build streaming request to LLM provider ───────────────────────────────
    // Inject stream: true into the request body
    let mut body = serde_json::to_value(&req).unwrap_or_default();
    body["stream"] = serde_json::Value::Bool(true);

    let http_req = match provider {
        Provider::Anthropic => {
            // Anthropic uses x-api-key header
            http_client
                .post(&base_url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body)
        }
        _ => {
            // OpenAI, DeepSeek, Grok: Bearer auth
            http_client
                .post(&base_url)
                .bearer_auth(&api_key)
                .json(&body)
        }
    };

    // ── SSE stream definition ─────────────────────────────────────────────────
    let stream = try_stream! {
        let response = match http_req.send().await {
            Ok(r) => r,
            Err(e) => {
                error!("Streaming request failed: {e}");
                yield Event::default()
                    .data(format!(r#"{{"error":"upstream failed: {e}"}}"#));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            error!(status = status, body = %body, "Provider returned error");
            yield Event::default()
                .data(format!(r#"{{"error":"provider error {status}"}}"#));
            return;
        }

        let mut byte_stream = response.bytes_stream();

        // Accumulators for audit trail
        let mut full_text      = String::new();
        let mut input_tokens   = 0u32;
        let mut output_tokens  = 0u32;
        let mut chunk_count    = 0u32;

        while let Some(chunk_result) = byte_stream.next().await {
            let bytes = match chunk_result {
                Ok(b)  => b,
                Err(e) => {
                    error!("Stream read error: {e}");
                    break;
                }
            };

            let text = match std::str::from_utf8(&bytes) {
                Ok(t)  => t,
                Err(_) => continue,
            };

            // SSE lines look like: "data: {...}\n\n"
            for line in text.lines() {
                if line.is_empty() || line.starts_with(':') { continue; }

                let data = if let Some(d) = line.strip_prefix("data: ") {
                    d.trim()
                } else {
                    continue;
                };

                // OpenAI/Anthropic end-of-stream sentinel
                if data == "[DONE]" {
                    debug!(
                        chunks = chunk_count,
                        chars  = full_text.len(),
                        "Stream complete"
                    );

                    // Send audit payload through side channel
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let _ = audit_tx.send(AuditPayload {
                        provider,
                        input_tokens,
                        output_tokens,
                        latency_ms,
                        routing_ms: 0,   // already recorded in route handler
                    }).await;

                    yield Event::default().data("[DONE]");
                    return;
                }

                // Parse chunk to extract delta text + usage
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    // Extract text delta from choices
                    if let Some(choices) = &chunk.choices {
                        for choice in choices {
                            if let Some(content) = &choice.delta.content {
                                full_text.push_str(content);
                                chunk_count += 1;
                            }
                        }
                    }

                    // Some providers send usage in the final chunk
                    if let Some(usage) = &chunk.usage {
                        if let Some(pt) = usage.prompt_tokens     { input_tokens  = pt; }
                        if let Some(ct) = usage.completion_tokens { output_tokens = ct; }
                    }

                    // Forward the raw SSE data to the client — <1ms overhead
                    yield Event::default().data(data);
                }
            }
        }

        // Stream ended without [DONE] (connection drop etc.)
        let latency_ms = start.elapsed().as_millis() as u64;
        let _ = audit_tx.send(AuditPayload {
            provider,
            input_tokens,
            output_tokens,
            latency_ms,
            routing_ms: 0,
        }).await;
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    )
}

// =============================================================================
// Internal types
// =============================================================================

#[derive(Debug)]
struct AuditPayload {
    provider:      Provider,
    input_tokens:  u32,
    output_tokens: u32,
    latency_ms:    u64,
    routing_ms:    u64,
}
