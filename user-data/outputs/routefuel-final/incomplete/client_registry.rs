// =============================================================================
// src/client_registry.rs  — RouteFuel v0.5
//
// Per-client tier assignment.
//
// Loaded at startup from either:
//   A) Environment variable ROUTEFUEL_CLIENT_TIERS (fast, no DB needed)
//      Format:  "sha256hex:free,sha256hex:pro,sha256hex:enterprise"
//      Example: "abc123:pro,def456:enterprise"
//
//   B) Postgres table `client_tiers` (for runtime changes without redeploy)
//
// The RateLimiter is populated once at startup. Tier changes need a restart
// unless you use the Postgres path and call reload() periodically.
// =============================================================================

use crate::rate_limiter::{RateLimiter, TierConfig};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use tracing::{info, warn, error};

// =============================================================================
// Tier parsing
// =============================================================================

pub fn parse_tier(s: &str) -> TierConfig {
    match s.trim().to_lowercase().as_str() {
        "free"       => TierConfig::FREE,
        "pro"        => TierConfig::PRO,
        "enterprise" => TierConfig::ENTERPRISE,
        other => {
            warn!("Unknown tier '{}' — defaulting to Pro", other);
            TierConfig::PRO
        }
    }
}

// =============================================================================
// Load from environment variable
//
// ROUTEFUEL_CLIENT_TIERS format:
//   "raw_key_1:pro,raw_key_2:enterprise,raw_key_3:free"
//
// Keys are stored as SHA-256 hashes in ApiKeyStore.
// Here we accept the raw key so the same secret works for both auth and tier.
// =============================================================================

pub fn load_tiers_from_env(
    raw: &str,
    rate_limiter: &Arc<RateLimiter>,
) -> usize {
    let mut count = 0;

    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() { continue; }

        match entry.split_once(':') {
            Some((key, tier_str)) => {
                let client_id = sha256_hex(key.trim());
                let tier      = parse_tier(tier_str);

                rate_limiter.register(&client_id, tier);

                info!(
                    client_id = &client_id[..8],
                    tier = tier_str.trim(),
                    "Registered client tier from env"
                );
                count += 1;
            }
            None => {
                error!(
                    "Bad ROUTEFUEL_CLIENT_TIERS entry '{}' — format: raw_key:tier",
                    entry
                );
            }
        }
    }

    count
}

// =============================================================================
// Load from Postgres (optional — only if DATABASE_URL is set)
//
// CREATE TABLE IF NOT EXISTS client_tiers (
//     client_id   VARCHAR(64)  PRIMARY KEY,   -- SHA-256 hash of raw API key
//     tier        VARCHAR(20)  NOT NULL DEFAULT 'pro',
//     notes       TEXT,
//     created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
//     updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
// );
// =============================================================================

pub async fn load_tiers_from_db(
    pool:         &PgPool,
    rate_limiter: &Arc<RateLimiter>,
) -> Result<usize, sqlx::Error> {
    // Check if the table exists before querying
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'client_tiers'
         )"
    )
    .fetch_one(pool)
    .await?;

    if !table_exists {
        info!("client_tiers table not found — skipping DB tier load");
        return Ok(0);
    }

    let rows = sqlx::query_as::<_, (String, String)>(
        "SELECT client_id, tier FROM client_tiers ORDER BY updated_at DESC"
    )
    .fetch_all(pool)
    .await?;

    let count = rows.len();

    for (client_id, tier_str) in rows {
        let tier = parse_tier(&tier_str);
        rate_limiter.register(&client_id, tier);
        info!(
            client_id = &client_id[..client_id.len().min(8)],
            tier = %tier_str,
            "Registered client tier from DB"
        );
    }

    info!("Loaded {} client tiers from database", count);
    Ok(count)
}

// =============================================================================
// Combined loader — tries env first, then DB
// =============================================================================

pub async fn load_all_tiers(
    pool:              &PgPool,
    rate_limiter:      &Arc<RateLimiter>,
    env_tiers_raw:     &str,
    fallback_tier:     TierConfig,
) {
    let mut total = 0;

    // 1. Load from env var
    if !env_tiers_raw.is_empty() {
        let n = load_tiers_from_env(env_tiers_raw, rate_limiter);
        total += n;
        info!("Loaded {} client tiers from ROUTEFUEL_CLIENT_TIERS", n);
    }

    // 2. Load from DB (merges with env, DB takes precedence for same client)
    match load_tiers_from_db(pool, rate_limiter).await {
        Ok(n) => {
            total += n;
            info!("Loaded {} client tiers from database", n);
        }
        Err(e) => {
            warn!("Could not load tiers from DB ({}), using env only", e);
        }
    }

    if total == 0 {
        warn!(
            "No client tiers configured — all clients will use {:?} tier. \
             Set ROUTEFUEL_CLIENT_TIERS or add rows to client_tiers table.",
            match fallback_tier.capacity as u32 {
                c if c >= 2000 => "Enterprise",
                c if c >= 200  => "Pro",
                _              => "Free",
            }
        );
        // Register a catch-all "default" client that auto-registers
        // unknown clients through the rate_limiter's auto-register path
        rate_limiter.register("default", fallback_tier);
    }

    info!("Client tier registry ready ({} entries)", total);
}

// =============================================================================
// Migration SQL (run this manually or add to sqlx migrations)
// =============================================================================

pub const CLIENT_TIERS_MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS client_tiers (
    client_id   VARCHAR(64)  PRIMARY KEY,
    tier        VARCHAR(20)  NOT NULL DEFAULT 'pro'
                             CHECK (tier IN ('free', 'pro', 'enterprise')),
    notes       TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_client_tiers_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER client_tiers_updated_at
    BEFORE UPDATE ON client_tiers
    FOR EACH ROW EXECUTE FUNCTION update_client_tiers_updated_at();

COMMENT ON TABLE client_tiers IS
    'Per-client rate limit tiers. client_id is SHA-256 hex of the raw API key.';
"#;

// =============================================================================
// Helper
// =============================================================================

fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    format!("{:x}", h.finalize())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_tiers() {
        assert_eq!(parse_tier("free").capacity       as u32, TierConfig::FREE.capacity as u32);
        assert_eq!(parse_tier("pro").capacity        as u32, TierConfig::PRO.capacity  as u32);
        assert_eq!(parse_tier("enterprise").capacity as u32, TierConfig::ENTERPRISE.capacity as u32);
    }

    #[test]
    fn parse_unknown_defaults_to_pro() {
        let t = parse_tier("gold");
        assert_eq!(t.capacity as u32, TierConfig::PRO.capacity as u32);
    }

    #[test]
    fn load_from_env_registers_clients() {
        let rl  = Arc::new(RateLimiter::new());
        let raw = "rf_live_key1:pro,rf_live_key2:free,rf_live_key3:enterprise";
        let n   = load_tiers_from_env(raw, &rl);
        assert_eq!(n, 3);

        // Verify each was registered
        let id1 = {
            let mut h = sha2::Sha256::new();
            h.update(b"rf_live_key1");
            format!("{:x}", h.finalize())
        };
        let status = rl.status(&id1);
        assert!(status.is_some());
    }

    #[test]
    fn empty_env_string_registers_nothing() {
        let rl = Arc::new(RateLimiter::new());
        let n  = load_tiers_from_env("", &rl);
        assert_eq!(n, 0);
    }
}
