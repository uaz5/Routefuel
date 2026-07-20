// ============================================================================
// src/rate_limiter.rs
// Tier-based rate limiting for clients
// ============================================================================

use governor::{Quota, RateLimiter as GovernorLimiter};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::num::NonZeroU32;
use thiserror::Error;
use tracing::{warn, debug};

#[derive(Error, Debug)]
pub enum RateLimitError {
    #[error("Rate limit exceeded")]
    LimitExceeded,

    #[error("Unknown tier: {0}")]
    UnknownTier(String),
}

#[derive(Debug, Clone, Copy)]
pub enum UserTier {
    Free,
    Pro,
    Enterprise,
}

impl UserTier {
    /// Get requests per second limit for this tier
    pub fn rps_limit(&self) -> NonZeroU32 {
        match self {
            UserTier::Free => NonZeroU32::new(10).unwrap(),       // 10 req/s
            UserTier::Pro => NonZeroU32::new(100).unwrap(),       // 100 req/s
            UserTier::Enterprise => NonZeroU32::new(1000).unwrap(), // 1000 req/s
        }
    }
}

pub struct RateLimiter {
    limiters: RwLock<HashMap<String, GovernorLimiter>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limiters: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a client can make a request
    pub fn check_limit(&self, client_id: &str, tier: UserTier) -> Result<(), RateLimitError> {
        let mut limiters = self.limiters.write();

        let limiter = limiters
            .entry(client_id.to_string())
            .or_insert_with(|| GovernorLimiter::direct(Quota::per_second(tier.rps_limit())));

        if limiter.check().is_err() {
            warn!("Rate limit exceeded for client: {}", client_id);
            return Err(RateLimitError::LimitExceeded);
        }

        debug!("Rate limit check passed for client: {}", client_id);
        Ok(())
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_limits() {
        assert_eq!(UserTier::Free.rps_limit().get(), 10);
        assert_eq!(UserTier::Pro.rps_limit().get(), 100);
        assert_eq!(UserTier::Enterprise.rps_limit().get(), 1000);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new();
        let result = limiter.check_limit("client-1", UserTier::Free);
        assert!(result.is_ok());
    }
}
