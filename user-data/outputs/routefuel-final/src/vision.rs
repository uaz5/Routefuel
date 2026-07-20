// =============================================================================
// src/vision.rs  — RouteFuel v0.5
//
// Vision / Multimodal Support
//
// Extends the ChatMessage schema to carry:
//   - text content (existing)
//   - image_url    (URL pointing to an image)
//   - base64       (inline image data)
//
// All three OpenAI-compatible vision models supported:
//   - gpt-5.5, gpt-5.4  (OpenAI)
//   - claude-opus-4-7, claude-sonnet-4-6 (Anthropic — different wire format)
//   - gemini-3.1-pro, gemini-3-flash (Gemini — different wire format)
//
// The router detects image content and filters to vision-capable models only.
// =============================================================================

use serde::{Deserialize, Serialize};

// =============================================================================
// Extended message content types
// =============================================================================

/// A chat message that can carry either text or image content.
/// Replaces the plain `ChatMessage { role, content: String }` for multimodal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalMessage {
    pub role:    String,
    pub content: MessageContent,
}

/// Content can be a plain string (text-only) or a list of content parts
/// (text + one or more images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text-only message — same as before
    Text(String),
    /// Multimodal: list of text and/or image parts
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract all text segments as a single string (for token counting / caching)
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match &p.kind {
                    PartKind::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Returns true if this message contains at least one image
    pub fn has_image(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| !matches!(p.kind, PartKind::Text { .. })),
        }
    }
}

/// A single content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(flatten)]
    pub kind: PartKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PartKind {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    ImageBase64 {
        image_data: ImageBase64,
    },
}

/// Reference to an image by URL (must be publicly accessible or data URI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url:    String,
    /// "low" | "high" | "auto" — controls detail level and cost
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Inline base64-encoded image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBase64 {
    /// MIME type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    pub media_type: String,
    /// Raw base64 string (no data URI prefix)
    pub data:       String,
}

// =============================================================================
// Vision-capable model registry
// =============================================================================

/// Every model that can process image inputs as of May 2026.
/// The router filters to these models when the request contains images.
pub const VISION_CAPABLE_MODELS: &[(&str, &str)] = &[
    // OpenAI
    ("gpt-5.5",              "openai"),
    ("gpt-5.4",              "openai"),
    ("gpt-5.1",              "openai"),
    ("gpt-5",                "openai"),
    ("gpt-5.4-mini",         "openai"),
    // Anthropic
    ("claude-opus-4-7",      "anthropic"),
    ("claude-opus-4-6",      "anthropic"),
    ("claude-sonnet-4-6",    "anthropic"),
    // Gemini — all models support vision natively
    ("gemini-3.1-pro",       "gemini"),
    ("gemini-3-flash",       "gemini"),
    ("gemini-3.1-flash-lite","gemini"),
];

pub fn is_vision_capable(api_id: &str) -> bool {
    VISION_CAPABLE_MODELS.iter().any(|(id, _)| *id == api_id)
}

// =============================================================================
// Format conversion: OpenAI ↔ Anthropic ↔ Gemini
// =============================================================================

/// Convert a multimodal message into the Anthropic wire format.
/// Anthropic uses: { role, content: [ { type, text|source } ] }
pub fn to_anthropic_content(msg: &MultimodalMessage) -> serde_json::Value {
    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": msg.role,
            "content": t
        }),
        MessageContent::Parts(parts) => {
            let content: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url.url
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_data.media_type,
                            "data": image_data.data
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": msg.role, "content": content })
        }
    }
}

/// Convert a multimodal message into the Gemini wire format.
/// Gemini uses: { role, parts: [ { text } | { inlineData } ] }
pub fn to_gemini_content(msg: &MultimodalMessage) -> serde_json::Value {
    let role = if msg.role == "assistant" { "model" } else { "user" };

    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": role,
            "parts": [{ "text": t }]
        }),
        MessageContent::Parts(parts) => {
            let gemini_parts: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({ "text": text }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        // Gemini accepts file URIs or GCS URLs
                        "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": image_url.url
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "inlineData": {
                            "mimeType": image_data.media_type,
                            "data":     image_data.data
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": role, "parts": gemini_parts })
        }
    }
}

// =============================================================================
// Vision-aware routing helper
// =============================================================================

use crate::route_engine::{ModelConfig, RouteEngine, RoutingPriority};

/// Select the best vision-capable model.
/// Only considers models that support image input.
/// Falls back to quality priority if no preference given.
pub fn select_vision_model(
    engine:       &RouteEngine,
    input_tokens: u32,
    priority:     RoutingPriority,
) -> anyhow::Result<crate::route_engine::RoutingDecision> {
    // Ask the engine for its best pick, then verify it handles vision
    let decision = engine.select(input_tokens, 1024, priority)?;

    if is_vision_capable(&decision.model.api_id) {
        return Ok(decision);
    }

    // Fallback: try quality priority (flagship models are all vision-capable)
    let fallback = engine.select(input_tokens, 1024, RoutingPriority::Quality)?;
    if is_vision_capable(&fallback.model.api_id) {
        return Ok(fallback);
    }

    anyhow::bail!("No vision-capable model available in registry")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_message_has_no_image() {
        let msg = MultimodalMessage {
            role:    "user".into(),
            content: MessageContent::Text("hello".into()),
        };
        assert!(!msg.content.has_image());
    }

    #[test]
    fn parts_message_detects_image() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "text".into(),
                    kind: PartKind::Text { text: "What is in this image?".into() },
                },
                ContentPart {
                    part_type: "image_url".into(),
                    kind: PartKind::ImageUrl {
                        image_url: ImageUrl {
                            url:    "https://example.com/food.jpg".into(),
                            detail: Some("high".into()),
                        },
                    },
                },
            ]),
        };
        assert!(msg.content.has_image());
        assert_eq!(msg.content.as_text(), "What is in this image?");
    }

    #[test]
    fn all_flagship_models_are_vision_capable() {
        assert!(is_vision_capable("claude-opus-4-7"));
        assert!(is_vision_capable("gpt-5.5"));
        assert!(is_vision_capable("gemini-3.1-pro"));
        assert!(!is_vision_capable("deepseek-v4-flash")); // text-only
        assert!(!is_vision_capable("grok-4.3"));          // text-only
    }

    #[test]
    fn anthropic_conversion_base64() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "image".into(),
                    kind: PartKind::ImageBase64 {
                        image_data: ImageBase64 {
                            media_type: "image/png".into(),
                            data:       "abc123".into(),
                        },
                    },
                },
            ]),
        };
        let v = to_anthropic_content(&msg);
        let source = &v["content"][0]["source"];
        assert_eq!(source["type"], "base64");
        assert_eq!(source["media_type"], "image/png");
    }
}
