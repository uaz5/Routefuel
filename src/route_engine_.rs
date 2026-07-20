// ============================================================================
// src/route_engine.rs  — accurate as of May 6, 2026
//
// Sources:
//   Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
//   OpenAI:    https://platform.openai.com/docs/models
//   Google:    https://ai.google.dev/gemini-api/docs/models
//   DeepSeek:  https://platform.deepseek.com/api-docs
// ============================================================================

use crate::connectors::Provider;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use tracing::{debug, info, instrument};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The exact string you pass as the model field in API requests
    pub api_id: String,
    pub display_name: String,
    pub provider: Provider,
    /// Cost in CENTS per 1 million input tokens
    pub cost_per_1m_input: f64,
    /// Cost in CENTS per 1 million output tokens
    pub cost_per_1m_output: f64,
    /// Typical median latency ms (real-world, not marketing)
    pub latency_ms: u64,
    /// Subjective 0.0–1.0 quality score used in routing math
    pub quality_score: f32,
    /// Maximum input context tokens
    pub context_window: u32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum RoutingPriority {
    Cost,
    Balanced,
    Quality,
    Speed,
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub model: ModelConfig,
    pub score: f64,
    pub reason: String,
}

/// Tasks your meeting-assistant product exposes to callers.
/// Callers send `"task": "summarise"` — RouteFuel picks the model.
#[derive(Debug, Clone, Copy)]
pub enum MeetingTask {
    Summarise,
    AnswerQuestion,
    ExtractActionItems,
    DraftResponse,
    Classify,
}

impl std::str::FromStr for MeetingTask {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "summarise" | "summarize"    => Ok(Self::Summarise),
            "answer_question" | "qa"     => Ok(Self::AnswerQuestion),
            "extract_action_items"       => Ok(Self::ExtractActionItems),
            "draft_response" | "draft"   => Ok(Self::DraftResponse),
            "classify"                   => Ok(Self::Classify),
            _ => Err(anyhow!("Unknown task '{}'. Valid: summarise, answer_question, extract_action_items, draft_response, classify", s)),
        }
    }
}

pub struct RouteEngine {
    models: RwLock<Vec<ModelConfig>>,
}

impl RouteEngine {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(Self::build_registry()),
        }
    }

    /// Master registry — every real model available as of May 2026.
    fn build_registry() -> Vec<ModelConfig> {
        vec![

            // ============================================================
            // ANTHROPIC  (connector: AnthropicConnector)
            // POST https://api.anthropic.com/v1/messages
            // Headers: x-api-key, anthropic-version: 2023-06-01
            // ============================================================

            ModelConfig {
                api_id:            "claude-opus-4-7".into(),
                display_name:      "Claude Opus 4.7".into(),
                provider:          Provider::Anthropic,
                cost_per_1m_input: 500.0,   // $5.00 / 1M
                cost_per_1m_output:2500.0,  // $25.00 / 1M
                latency_ms:        280,
                quality_score:     1.00,
                context_window:    1_000_000,
                enabled:           true,
                // ⚠ New tokenizer — up to 35% more tokens than 4.6 for
                //   the same text. Benchmark before migrating from 4.6.
            },

            ModelConfig {
                api_id:            "claude-opus-4-6".into(),
                display_name:      "Claude Opus 4.6".into(),
                provider:          Provider::Anthropic,
                cost_per_1m_input: 500.0,
                cost_per_1m_output:2500.0,
                latency_ms:        260,
                quality_score:     0.97,
                context_window:    1_000_000,
                enabled:           true,
            },

            ModelConfig {
                api_id:            "claude-sonnet-4-6".into(),
                display_name:      "Claude Sonnet 4.6".into(),
                provider:          Provider::Anthropic,
                cost_per_1m_input: 300.0,   // $3.00 / 1M
                cost_per_1m_output:1500.0,  // $15.00 / 1M
                latency_ms:        170,
                quality_score:     0.92,
                context_window:    1_000_000,
                enabled:           true,
            },

            ModelConfig {
                api_id:            "claude-haiku-4-5-20251001".into(),
                display_name:      "Claude Haiku 4.5".into(),
                provider:          Provider::Anthropic,
                cost_per_1m_input: 100.0,   // $1.00 / 1M
                cost_per_1m_output:500.0,   // $5.00 / 1M
                latency_ms:        90,
                quality_score:     0.78,
                context_window:    200_000,
                enabled:           true,
            },

            // ============================================================
            // OPENAI  (connector: OpenAIConnector)
            // POST https://api.openai.com/v1/chat/completions
            // Header: Authorization: Bearer <key>
            // ============================================================

            ModelConfig {
                api_id:            "gpt-4o".into(),
                display_name:      "GPT-4o".into(),
                provider:          Provider::OpenAI,
                cost_per_1m_input: 250.0,   // $2.50 / 1M
                cost_per_1m_output:1000.0,  // $10.00 / 1M
                latency_ms:        200,
                quality_score:     0.94,
                context_window:    128_000,
                enabled:           true,
            },

            ModelConfig {
                api_id:            "gpt-4o-mini".into(),
                display_name:      "GPT-4o Mini".into(),
                provider:          Provider::OpenAI,
                cost_per_1m_input: 15.0,    // $0.15 / 1M
                cost_per_1m_output:60.0,    // $0.60 / 1M
                latency_ms:        120,
                quality_score:     0.75,
                context_window:    128_000,
                enabled:           true,
            },

            // ============================================================
            // GOOGLE GEMINI  (connector: GeminiConnector — TODO)
            // POST https://generativelanguage.googleapis.com/v1beta/
            //       models/{model_id}:generateContent?key={API_KEY}
            // Different JSON schema — needs its own connector.
            // Enabled: false until GeminiConnector is implemented.
            // ============================================================

            ModelConfig {
                api_id:            "gemini-3.1-pro".into(),
                display_name:      "Gemini 3.1 Pro".into(),
                provider:          Provider::Gemini,
                cost_per_1m_input: 200.0,   // $2.00 / 1M (≤200K ctx)
                cost_per_1m_output:1200.0,  // $12.00 / 1M
                latency_ms:        220,
                quality_score:     0.95,
                context_window:    1_000_000,
                enabled:           false,   // flip to true after adding GeminiConnector
            },

            ModelConfig {
                api_id:            "gemini-3-flash".into(),
                display_name:      "Gemini 3 Flash".into(),
                provider:          Provider::Gemini,
                cost_per_1m_input: 50.0,    // $0.50 / 1M
                cost_per_1m_output:300.0,   // $3.00 / 1M
                latency_ms:        110,
                quality_score:     0.85,
                context_window:    1_000_000,
                enabled:           false,
            },

            ModelConfig {
                api_id:            "gemini-3.1-flash-lite".into(),
                display_name:      "Gemini 3.1 Flash-Lite".into(),
                provider:          Provider::Gemini,
                cost_per_1m_input: 10.0,    // $0.10 / 1M — very cheap
                cost_per_1m_output:40.0,    // $0.40 / 1M
                latency_ms:        75,
                quality_score:     0.70,
                context_window:    1_000_000,
                enabled:           false,
            },

            // ============================================================
            // DEEPSEEK V4  (connector: re-use OpenAIConnector, different base URL)
            // POST https://api.deepseek.com/v1/chat/completions
            // Header: Authorization: Bearer <key>
            // ✅ OpenAI-compatible JSON — cheapest change to add
            // ============================================================

            ModelConfig {
                api_id:            "deepseek-v4-flash".into(),
                display_name:      "DeepSeek V4 Flash".into(),
                provider:          Provider::DeepSeek,
                cost_per_1m_input: 14.0,    // $0.14 / 1M — cheapest quality model
                cost_per_1m_output:28.0,    // $0.28 / 1M
                latency_ms:        140,
                quality_score:     0.82,
                context_window:    1_000_000,
                enabled:           false,   // flip to true after adding DEEPSEEK_API_KEY
            },

            ModelConfig {
                api_id:            "deepseek-v4-pro".into(),
                display_name:      "DeepSeek V4 Pro".into(),
                provider:          Provider::DeepSeek,
                cost_per_1m_input: 43.0,    // $0.43 / 1M
                cost_per_1m_output:87.0,    // $0.87 / 1M
                latency_ms:        185,
                quality_score:     0.90,
                context_window:    1_000_000,
                enabled:           false,
            },
        ]
    }

    // ===================================================================
    // SCORE-BASED ROUTING
    // ===================================================================

    #[instrument(skip(self))]
    pub fn select(
        &self,
        input_tokens: u32,
        max_output_tokens: u32,
        priority: RoutingPriority,
    ) -> Result<RoutingDecision> {
        let models = self.models.read();

        // Weight tuples: (cost, latency, quality, context_headroom)
        let (wc, wl, wq, wx) = match priority {
            RoutingPriority::Cost     => (0.60, 0.15, 0.20, 0.05),
            RoutingPriority::Balanced => (0.35, 0.25, 0.30, 0.10),
            RoutingPriority::Quality  => (0.10, 0.15, 0.65, 0.10),
            RoutingPriority::Speed    => (0.10, 0.70, 0.15, 0.05),
        };

        let mut best: Option<(ModelConfig, f64)> = None;

        for m in models.iter().filter(|m| m.enabled) {
            if input_tokens >= m.context_window {
                debug!(model = %m.api_id, "Skipped — context overflow");
                continue;
            }

            let cost = (input_tokens as f64 / 1_000_000.0) * m.cost_per_1m_input
                     + (max_output_tokens as f64 / 1_000_000.0) * m.cost_per_1m_output;

            let s_cost    = 1.0 / (1.0 + cost / 10.0);
            let s_latency = 1.0 / (1.0 + m.latency_ms as f64 / 200.0);
            let s_quality = m.quality_score as f64;
            let s_context = 1.0 - (input_tokens as f64 / m.context_window as f64);

            let score = wc * s_cost + wl * s_latency + wq * s_quality + wx * s_context;

            debug!(model = %m.api_id, score = format!("{:.4}", score));

            if best.as_ref().map_or(true, |(_, b)| score > *b) {
                best = Some((m.clone(), score));
            }
        }

        let (model, score) = best.ok_or_else(|| {
            anyhow!("No eligible model — check that at least one model is enabled and your input fits its context window")
        })?;

        let reason = format!(
            "{} (score={:.4}, priority={:?})",
            model.display_name, score, priority
        );
        info!("{}", reason);
        Ok(RoutingDecision { model, score, reason })
    }

    // ===================================================================
    // TASK-BASED ROUTING  (for meeting assistant)
    // ===================================================================

    pub fn select_for_task(
        &self,
        task: MeetingTask,
        input_tokens: u32,
    ) -> Result<RoutingDecision> {
        let (priority, preferred) = match task {
            MeetingTask::Summarise          => (RoutingPriority::Balanced, "claude-sonnet-4-6"),
            MeetingTask::AnswerQuestion      => (RoutingPriority::Speed,    "gemini-3-flash"),
            MeetingTask::ExtractActionItems  => (RoutingPriority::Cost,     "deepseek-v4-flash"),
            MeetingTask::DraftResponse       => (RoutingPriority::Quality,  "claude-opus-4-6"),
            MeetingTask::Classify            => (RoutingPriority::Cost,     "gemini-3.1-flash-lite"),
        };

        // Try the preferred model first (best UX for each task)
        if let Ok(m) = self.find(preferred) {
            if m.enabled && input_tokens < m.context_window {
                let reason = format!("{} chosen as task-optimal for {:?}", m.display_name, task);
                info!("{}", reason);
                return Ok(RoutingDecision { model: m, score: 1.0, reason });
            }
        }

        // Preferred not available → fall back to score-based
        self.select(input_tokens, 1024, priority)
    }

    // ===================================================================
    // HELPERS
    // ===================================================================

    pub fn find(&self, api_id: &str) -> Result<ModelConfig> {
        self.models.read()
            .iter()
            .find(|m| m.api_id == api_id)
            .cloned()
            .ok_or_else(|| anyhow!("Unknown model id: {}", api_id))
    }

    pub fn get_pricing(&self, api_id: &str) -> Result<(f64, f64)> {
        let m = self.find(api_id)?;
        Ok((m.cost_per_1m_input, m.cost_per_1m_output))
    }

    pub fn list_enabled(&self) -> Vec<ModelConfig> {
        self.models.read().iter().filter(|m| m.enabled).cloned().collect()
    }
}

impl Default for RouteEngine {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_picks_a_model() {
        let e = RouteEngine::new();
        e.select(5_000, 1_000, RoutingPriority::Balanced).unwrap();
    }

    #[test]
    fn cost_does_not_pick_opus() {
        let e = RouteEngine::new();
        let d = e.select(5_000, 1_000, RoutingPriority::Cost).unwrap();
        assert_ne!(d.model.api_id, "claude-opus-4-7");
    }

    #[test]
    fn overflow_model_excluded() {
        // 200K tokens — must skip gpt-4o (128K window)
        let e = RouteEngine::new();
        let d = e.select(200_000, 1_000, RoutingPriority::Balanced).unwrap();
        assert!(d.model.context_window >= 200_001);
    }

    #[test]
    fn task_routing_extract_picks_deepseek() {
        let e = RouteEngine::new();
        // DeepSeek V4 Flash is disabled by default — enable it
        {
            let mut models = e.models.write();
            models.iter_mut()
                .find(|m| m.api_id == "deepseek-v4-flash")
                .map(|m| m.enabled = true);
        }
        let d = e.select_for_task(MeetingTask::ExtractActionItems, 5_000).unwrap();
        assert_eq!(d.model.api_id, "deepseek-v4-flash");
    }

    #[test]
    fn find_opus_47_pricing() {
        let e = RouteEngine::new();
        let (input, output) = e.get_pricing("claude-opus-4-7").unwrap();
        assert_eq!(input, 500.0);
        assert_eq!(output, 2500.0);
    }
}
