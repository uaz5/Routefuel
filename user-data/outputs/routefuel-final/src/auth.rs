// =============================================================================
// src/auth.rs  — RouteFuel v0.4
//
// How it works:
//   1. Client sends:   X-API-Key: rf_live_abc123yoursecretkey
//   2. Middleware SHA-256 hashes the raw key
//   3. Compares hash against the in-memory store (no plaintext ever stored)
//   4. If valid → injects X-Routefuel-Client-Id header for downstream handlers
//   5. If invalid → returns 401 immediately, request never reaches the handler
//
// Key store is loaded once at startup from the ROUTEFUEL_API_KEYS env var.
// Format:  sha256hex:ClientName,sha256hex:ClientName,...
//
// To generate a key hash on the command line:
//   echo -n "rf_live_yoursecretkey" | sha256sum | awk '{print $1}'
// =============================================================================

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, warn};

// =============================================================================
// ApiKeyStore  — loaded once, shared via Arc, read-only after init
// =============================================================================

pub struct ApiKeyStore {
    /// sha256_hex → client display name
    keys: HashMap<String, String>,
}

impl ApiKeyStore {
    /// Parse the ROUTEFUEL_API_KEYS env var.
    /// Expected format:  "sha256hex:ClientA,sha256hex:ClientB"
    pub fn from_env_string(raw: &str) -> Self {
        let mut keys = HashMap::new();

        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() { continue; }

            match entry.split_once(':') {
                Some((hash, name)) => {
                    let hash = hash.trim().to_lowercase();
                    let name = name.trim().to_string();
                    if hash.len() == 64 {
                        keys.insert(hash, name);
                    } else {
                        tracing::error!(
                            "Invalid API key hash '{}' — expected 64-char SHA-256 hex string",
                            hash
                        );
                    }
                }
                None => {
                    tracing::error!(
                        "Malformed ROUTEFUEL_API_KEYS entry '{}' — expected 'sha256hex:ClientName'",
                        entry
                    );
                }
            }
        }

        tracing::info!("Loaded {} API key(s) into auth store", keys.len());
        Self { keys }
    }

    /// Validate a raw API key and return the associated client name if valid.
    pub fn validate(&self, raw_key: &str) -> Option<&str> {
        let hash = sha256_hex(raw_key);
        self.keys.get(&hash).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

// =============================================================================
// Middleware
// =============================================================================

#[derive(Serialize)]
struct AuthError {
    error: AuthErrorDetail,
}

#[derive(Serialize)]
struct AuthErrorDetail {
    message: &'static str,
    code:    &'static str,
}

fn unauthorized(message: &'static str) -> Response {
    let body = Json(AuthError {
        error: AuthErrorDetail {
            message,
            code: "unauthorized",
        },
    });
    (StatusCode::UNAUTHORIZED, body).into_response()
}

/// Axum middleware — validates X-API-Key, injects client ID, passes through.
pub async fn api_key_middleware(
    State(store): State<Arc<ApiKeyStore>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // Extract raw key from header
    let raw_key = match request.headers().get("x-api-key") {
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                warn!("X-API-Key header contains non-UTF-8 bytes");
                return unauthorized("X-API-Key header must be a valid UTF-8 string");
            }
        },
        None => {
            debug!("Request missing X-API-Key header");
            return unauthorized("Missing X-API-Key header");
        }
    };

    // Validate against the store
    match store.validate(&raw_key) {
        Some(client_name) => {
            debug!(client = client_name, "API key validated");

            // Inject client ID so handlers can use it without re-hashing
            request.headers_mut().insert(
                "x-routefuel-client-id",
                client_name
                    .parse()
                    .unwrap_or_else(|_| "unknown".parse().unwrap()),
            );

            next.run(request).await
        }
        None => {
            warn!(
                key_prefix = &raw_key[..raw_key.len().min(8)],
                "Invalid API key"
            );
            unauthorized("Invalid API key")
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(raw_key: &str, name: &str) -> ApiKeyStore {
        let hash = sha256_hex(raw_key);
        let env_str = format!("{}:{}", hash, name);
        ApiKeyStore::from_env_string(&env_str)
    }

    #[test]
    fn valid_key_returns_client_name() {
        let store = store_with("rf_live_supersecret", "AcmeCorp");
        assert_eq!(store.validate("rf_live_supersecret"), Some("AcmeCorp"));
    }

    #[test]
    fn wrong_key_returns_none() {
        let store = store_with("rf_live_supersecret", "AcmeCorp");
        assert!(store.validate("rf_live_wrongkey").is_none());
    }

    #[test]
    fn empty_env_string_produces_empty_store() {
        let store = ApiKeyStore::from_env_string("");
        assert!(store.is_empty());
    }

    #[test]
    fn multiple_keys_parsed_correctly() {
        let k1 = sha256_hex("key_one");
        let k2 = sha256_hex("key_two");
        let env_str = format!("{k1}:ClientA,{k2}:ClientB");
        let store = ApiKeyStore::from_env_string(&env_str);
        assert_eq!(store.len(), 2);
        assert_eq!(store.validate("key_one"), Some("ClientA"));
        assert_eq!(store.validate("key_two"), Some("ClientB"));
    }

    #[test]
    fn short_hash_rejected() {
        // Only 10 chars — not a valid SHA-256 hex
        let store = ApiKeyStore::from_env_string("abc123:Client");
        assert!(store.is_empty());
    }

    #[test]
    fn key_not_stored_in_plaintext() {
        let store = store_with("rf_live_plaintextkey", "TestClient");
        // The raw key must not appear anywhere in the keys map
        for k in store.keys.keys() {
            assert_ne!(k, "rf_live_plaintextkey");
        }
    }
}
