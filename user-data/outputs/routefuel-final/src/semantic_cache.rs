// =============================================================================
// src/semantic_cache.rs  — RouteFuel v0.5
//
// Semantic Cache using pgvector
//
// Flow:
//   1. Incoming prompt → SHA-256 exact hash check (free, <0.1ms)
//   2. Cache miss → embed prompt via text-embedding-3-small (~50ms, $0.00002)
//   3. pgvector cosine similarity search — hit if similarity >= 0.96
//   4. Cache hit  → return cached response instantly (saves LLM call cost)
//   5. Cache miss → call LLM normally → store result async (non-blocking)
//
// pgvector 0.4 with sqlx feature flag (confirmed May 2026)
// Embedding model: text-embedding-3-small (1536 dims, cheapest OpenAI embedding)
// =============================================================================

use anyhow::Result;
use pgvector::Vector;
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, instrument};

const SIMILARITY_THRESHOLD: f64 = 0.96;
const EMBEDDING_MODEL: &str     = "text-embedding-3-small";
const EMBEDDING_DIMS:  usize    = 1536;
const OPENAI_EMBED_URL: &str    = "https://api.openai.com/v1/embeddings";

// =============================================================================
// Types
// =============================================================================

#[derive(Debug, Clone)]
pub struct CacheHit {
    pub cached_response: String,   // raw JSON of the original ChatResponse
    pub model_used:      String,
    pub similarity:      f64,
}

// =============================================================================
// SemanticCache
// =============================================================================

pub struct SemanticCache {
    pool:           Arc<PgPool>,
    openai_key:     String,
    http_client:    reqwest::Client,
    enabled:        bool,
}

impl SemanticCache {
    pub fn new(pool: Arc<PgPool>, openai_key: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to build embedding HTTP client");

        Self { pool, openai_key, http_client, enabled: true }
    }

    pub fn disable(&mut self) { self.enabled = false; }

    // =========================================================================
    // Lookup — check cache before calling LLM
    // =========================================================================

    #[instrument(skip(self, prompt))]
    pub async fn lookup(&self, prompt: &str) -> Option<CacheHit> {
        if !self.enabled { return None; }

        let start = Instant::now();

        // ── Step 1: exact hash check (free, <0.1ms) ──────────────────────────
        let hash = sha256_hex(prompt);

        let exact: Option<(String, String)> = sqlx::query_as(
            "SELECT cached_response, model_used
             FROM semantic_cache
             WHERE prompt_hash = $1
             LIMIT 1",
        )
        .bind(&hash)
        .fetch_optional(self.pool.as_ref())
        .await
        .ok()
        .flatten();

        if let Some((response, model)) = exact {
            debug!(latency_us = start.elapsed().as_micros(), "Exact cache hit");
            // Bump hit count async
            let pool = Arc::clone(&self.pool);
            let hash_clone = hash.clone();
            tokio::spawn(async move {
                let _ = sqlx::query(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1,
                     last_hit_at = NOW() WHERE prompt_hash = $1"
                )
                .bind(&hash_clone)
                .execute(pool.as_ref())
                .await;
            });
            return Some(CacheHit {
                cached_response: response,
                model_used: model,
                similarity: 1.0,
            });
        }

        // ── Step 2: embed prompt ──────────────────────────────────────────────
        let embedding = match self.embed(prompt).await {
            Ok(v)  => v,
            Err(e) => {
                debug!("Embedding failed (cache bypassed): {e}");
                return None;
            }
        };

        let vector = Vector::from(embedding);

        // ── Step 3: pgvector cosine similarity search ─────────────────────────
        // 1 - cosine_distance = cosine_similarity
        // We want similarity >= 0.96, so distance <= 0.04
        let row: Option<(String, String, f64)> = sqlx::query_as(
            "SELECT cached_response, model_used,
                    1 - (embedding <=> $1::vector) AS similarity
             FROM semantic_cache
             WHERE 1 - (embedding <=> $1::vector) >= $2
             ORDER BY embedding <=> $1::vector
             LIMIT 1",
        )
        .bind(&vector)
        .bind(SIMILARITY_THRESHOLD)
        .fetch_optional(self.pool.as_ref())
        .await
        .ok()
        .flatten();

        if let Some((response, model, similarity)) = row {
            debug!(
                similarity = similarity,
                latency_ms = start.elapsed().as_millis(),
                "Semantic cache hit"
            );

            // Bump hit count async
            let pool = Arc::clone(&self.pool);
            tokio::spawn(async move {
                let _ = sqlx::query(
                    "UPDATE semantic_cache
                     SET hit_count = hit_count + 1, last_hit_at = NOW()
                     WHERE cached_response = $1
                     LIMIT 1"
                )
                .bind(&response)
                .execute(pool.as_ref())
                .await;
            });

            return Some(CacheHit { cached_response: response, model_used: model, similarity });
        }

        debug!(latency_ms = start.elapsed().as_millis(), "Cache miss");
        None
    }

    // =========================================================================
    // Store — cache a response after a successful LLM call (fire-and-forget)
    // =========================================================================

    pub fn store(
        &self,
        prompt:          String,
        cached_response: String,
        model_used:      String,
    ) {
        if !self.enabled { return; }

        let pool       = Arc::clone(&self.pool);
        let openai_key = self.openai_key.clone();
        let client     = self.http_client.clone();

        tokio::spawn(async move {
            let hash = sha256_hex(&prompt);

            // Generate embedding
            let embedding = match embed_via_api(&client, &openai_key, &prompt).await {
                Ok(e)  => e,
                Err(e) => {
                    debug!("Cache store embedding failed: {e}");
                    return;
                }
            };

            let vector = Vector::from(embedding);

            // Upsert into cache table
            if let Err(e) = sqlx::query(
                "INSERT INTO semantic_cache
                    (prompt_hash, prompt_preview, embedding, cached_response, model_used)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (prompt_hash) DO NOTHING",
            )
            .bind(&hash)
            .bind(&prompt[..prompt.len().min(200)])  // preview only
            .bind(&vector)
            .bind(&cached_response)
            .bind(&model_used)
            .execute(pool.as_ref())
            .await
            {
                debug!("Cache store failed: {e}");
            } else {
                debug!("Cached response for prompt (hash: {})", &hash[..8]);
            }
        });
    }

    // =========================================================================
    // Embed — convert text to vector via OpenAI embeddings API
    // =========================================================================

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        embed_via_api(&self.http_client, &self.openai_key, text).await
    }
}

// =============================================================================
// Standalone embed function (used in both lookup and store paths)
// =============================================================================

async fn embed_via_api(
    client:     &reqwest::Client,
    openai_key: &str,
    text:       &str,
) -> Result<Vec<f32>> {
    #[derive(serde::Serialize)]
    struct EmbedReq<'a> {
        model:           &'a str,
        input:           &'a str,
        encoding_format: &'a str,
    }

    #[derive(serde::Deserialize)]
    struct EmbedResp {
        data: Vec<EmbedData>,
    }

    #[derive(serde::Deserialize)]
    struct EmbedData {
        embedding: Vec<f32>,
    }

    let body = EmbedReq {
        model:           EMBEDDING_MODEL,
        input:           text,
        encoding_format: "float",
    };

    let resp = client
        .post(OPENAI_EMBED_URL)
        .bearer_auth(openai_key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<EmbedResp>()
        .await?;

    Ok(resp.data.into_iter().next()
        .ok_or_else(|| anyhow::anyhow!("Empty embedding response"))?
        .embedding)
}

fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    format!("{:x}", h.finalize())
}
