// ============================================================================
// COPY-PASTE READY CODE FOR SMART ROUTING
// Add this to your routefuel-final/src/route_engine.rs
// ============================================================================

// ADD THIS TO route_engine.rs (after the existing select_provider function)

/// Smart routing that scores models based on cost, latency, quality, and context
///
/// Weights:
/// - Cost: 35% (favor cheaper models)
/// - Latency: 25% (favor faster models)
/// - Quality: 30% (favor higher quality)
/// - Context: 10% (penalty if context doesn't fit)
pub fn select_provider_smart(
    &self,
    request: &ChatCompletionRequest,
    input_tokens: u32,
) -> Result<(Provider, String)> {
    let models = self.models.read();

    let output_tokens = request.max_tokens.unwrap_or(1024);
    let mut scored_models = Vec::new();

    for model in models.iter() {
        // Calculate cost for this specific request
        let cost_input_cents =
            (input_tokens as f64 / 1_000_000.0) * model.cost_per_1m_input;
        let cost_output_cents =
            (output_tokens as f64 / 1_000_000.0) * model.cost_per_1m_output;
        let total_cost_cents = cost_input_cents + cost_output_cents;

        // Cost score: cheaper = higher score
        // Using: score = 1 / (1 + cost/100)
        let cost_score = 1.0 / (1.0 + (total_cost_cents / 100.0));

        // Latency score: faster = higher score
        // Using: score = 1 / (1 + latency/100)
        let latency_score = 1.0 / (1.0 + (model.latency_ms as f64 / 100.0));

        // Quality score: from config (GPT-4o=1.0, DeepSeek=0.7, etc)
        let quality_score = model.quality_score as f64;

        // Context score: penalty if context doesn't fit in model window
        let context_score = if input_tokens < model.context_window {
            1.0
        } else {
            0.1 // Heavy penalty if context doesn't fit
        };

        // WEIGHTED FINAL SCORE
        let final_score = (0.35 * cost_score)    // Cost is most important
                        + (0.25 * latency_score)  // Latency second
                        + (0.30 * quality_score)  // Quality important
                        + (0.10 * context_score); // Context fit

        debug!(
            model = %model.name,
            cost_score = cost_score,
            latency_score = latency_score,
            quality_score = quality_score,
            context_score = context_score,
            final_score = final_score,
            total_cost_cents = total_cost_cents,
            "Scored model"
        );

        scored_models.push((model.clone(), final_score));
    }

    // Sort by score (highest first) and pick the winner
    scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_model, best_score) = scored_models.first()
        .ok_or_else(|| anyhow!("No models available"))?
        .clone();

    info!(
        model = %best_model.name,
        score = best_score,
        provider = ?best_model.provider,
        "Smart routing selected best model"
    );

    Ok((best_model.provider, best_model.name))
}

/// Route based on specific task type (for meeting assistant)
///
/// This is more direct: "I know what you're doing, here's the best model for it"
pub fn select_for_task(
    &self,
    task_type: &str,
) -> Result<(Provider, String)> {
    let models = self.models.read();

    let model = match task_type {
        // Summarization: need big context, natural language understanding
        "summarize" | "meeting_summary" => {
            models
                .iter()
                .find(|m| m.name == "claude-3-5-sonnet")
                .or_else(|| models.iter().find(|m| m.name == "gpt-4o"))
                .ok_or_else(|| anyhow!("No summarization model found"))?
        }

        // Question answering: need speed and accuracy
        "answer_question" | "qa" => {
            models
                .iter()
                .find(|m| m.name == "gpt-4o")
                .ok_or_else(|| anyhow!("GPT-4o not found"))?
        }

        // Simple extraction: prioritize cost
        "extract_action_items" | "extract" => {
            models
                .iter()
                .find(|m| m.name == "deepseek-v3")
                .or_else(|| models.iter().find(|m| m.name == "claude-3-5-sonnet"))
                .ok_or_else(|| anyhow!("No extraction model found"))?
        }

        // Response generation: need natural language, prefer Claude
        "generate_response" | "generate" => {
            models
                .iter()
                .find(|m| m.name == "claude-opus") // Most capable
                .or_else(|| models.iter().find(|m| m.name == "claude-3-5-sonnet")) // Good fallback
                .or_else(|| models.iter().find(|m| m.name == "gpt-4o"))
                .ok_or_else(|| anyhow!("No generation model found"))?
        }

        // Default: use smart routing
        _ => {
            return Err(anyhow!(
                "Unknown task type: {}. Use: summarize, answer_question, extract_action_items, generate_response",
                task_type
            ));
        }
    };

    info!(
        task_type = task_type,
        model = %model.name,
        provider = ?model.provider,
        "Selected model for task"
    );

    Ok((model.provider, model.name.clone()))
}

// ============================================================================
// INTEGRATION EXAMPLE FOR MAIN.RS
// Replace the existing chat_completions_handler with this
// ============================================================================

#[instrument(skip(state, request), fields(request_id))]
async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let request_id = Uuid::new_v4().to_string();
    tracing::Span::current().record("request_id", &request_id);

    let start = Instant::now();
    info!(
        model = %request.model,
        message_count = request.messages.len(),
        "Received chat completion request"
    );

    // ========================================================================
    // STEP 1: COUNT INPUT TOKENS
    // ========================================================================

    let input_tokens = tokens::count_request_tokens(&request.messages, &request.model)
        .map_err(|e| {
            error!("Token counting failed: {}", e);
            ApiError::InternalError(format!("Token counting failed: {}", e))
        })?;

    let estimated_output = tokens::estimate_output_tokens(request.max_tokens, &request.model);

    debug!(
        input_tokens = input_tokens,
        estimated_output_tokens = estimated_output,
        "Counted request tokens"
    );

    // ========================================================================
    // STEP 2: SMART ROUTING DECISION
    // ========================================================================

    let routing_start = Instant::now();

    // Check if user specified "auto" or a specific task type
    let (selected_provider, selected_model_name) = if request.model == "auto" {
        // Let RouteFuel decide based on cost/quality/speed
        state
            .route_engine
            .select_provider_smart(&request, input_tokens)
            .map_err(|e| {
                error!("Smart routing failed: {}", e);
                ApiError::ProviderError("No available providers".to_string())
            })?
    } else if request.model.starts_with("task:") {
        // Task-based routing: "task:summarize", "task:answer_question", etc
        let task_type = request.model.strip_prefix("task:").unwrap();
        state
            .route_engine
            .select_for_task(task_type)
            .map_err(|e| {
                error!("Task-based routing failed: {}", e);
                ApiError::BadRequest(e.to_string())
            })?
    } else {
        // Specific model requested
        let provider = state
            .route_engine
            .select_provider(&request.model)
            .map_err(|e| {
                error!("Model lookup failed: {}", e);
                ApiError::ProviderError("Model not found".to_string())
            })?;
        (provider, request.model.clone())
    };

    let routing_decision_ms = routing_start.elapsed().as_millis() as u64;

    debug!(
        selected_provider = ?selected_provider,
        selected_model = %selected_model_name,
        routing_decision_ms = routing_decision_ms,
        "Routing decision completed"
    );

    // Update the request with the actual model we're using
    request.model = selected_model_name.clone();

    // ========================================================================
    // STEP 3: CIRCUIT BREAKER CHECK
    // ========================================================================

    if state.circuit_breaker.is_open(selected_provider) {
        error!("Circuit breaker open for provider: {:?}", selected_provider);
        return Err(ApiError::CircuitOpen);
    }

    // ========================================================================
    // STEP 4: CALL THE SELECTED PROVIDER
    // ========================================================================

    let connector_result = state
        .connector_manager
        .call(selected_provider, &request)
        .await
        .map_err(|e| {
            error!("Connector error: {}", e);

            // Record failure in cost tracker
            state.cost_tracker.record_error(
                request_id.clone(),
                selected_provider,
                selected_model_name.clone(),
                e.to_string(),
                start.elapsed().as_millis() as u64,
                routing_decision_ms,
                None,
            );

            match e {
                connectors::ConnectorError::CircuitBreakerOpen => ApiError::CircuitOpen,
                connectors::ConnectorError::RateLimited => ApiError::RateLimited,
                connectors::ConnectorError::ProviderServerError { .. } => {
                    ApiError::ProviderError("Provider returned an error".to_string())
                }
                _ => ApiError::ProviderError(e.to_string()),
            }
        })?;

    let response = connector_result.response.clone();
    let latency_ms = connector_result.latency_ms;
    let output_tokens = connector_result.output_tokens;

    // ========================================================================
    // STEP 5: VERIFY OUTPUT TOKENS
    // ========================================================================

    let response_text = response
        .choices
        .first()
        .map(|c| c.message.content.as_str())
        .unwrap_or("");

    if let Ok((counted, matches)) = tokens::verify_output_tokens(response_text, output_tokens) {
        debug!(
            reported = output_tokens,
            counted = counted,
            matches = matches,
            "Verified output tokens"
        );
    }

    // ========================================================================
    // STEP 6: CALCULATE COST WITH PRECISE TOKENS
    // ========================================================================

    let (cost_per_1m_input, cost_per_1m_output) = state
        .route_engine
        .get_pricing(selected_provider, &selected_model_name)
        .map_err(|e| {
            error!("Pricing lookup failed: {}", e);
            ApiError::InternalError("Pricing lookup failed".to_string())
        })?;

    let token_cost = TokenCostBreakdown::new(
        input_tokens,
        output_tokens,
        cost_per_1m_input,
        cost_per_1m_output,
    );

    // Calculate baseline cost (GPT-4o pricing: 250 input, 1000 output)
    let baseline_cost = TokenCostBreakdown::new(input_tokens, output_tokens, 250.0, 1000.0);

    let cost_saved = baseline_cost.total_cost_cents - token_cost.total_cost_cents;
    let savings_pct = (cost_saved / baseline_cost.total_cost_cents) * 100.0;

    debug!(
        cost_cents = token_cost.total_cost_cents,
        baseline_cents = baseline_cost.total_cost_cents,
        cost_saved_cents = cost_saved,
        savings_pct = savings_pct,
        "Calculated costs"
    );

    // ========================================================================
    // STEP 7: RECORD TO POSTGRES (non-blocking via tokio::spawn)
    // ========================================================================

    state.cost_tracker.record_request(
        request_id.clone(),
        selected_provider,
        selected_model_name.clone(),
        &token_cost,
        baseline_cost.total_cost_cents,
        latency_ms,
        routing_decision_ms,
        None, // TODO: Extract from auth header
        None, // TODO: Extract from headers
        None, // TODO: Extract from headers
    );

    // ========================================================================
    // STEP 8: RETURN RESPONSE
    // ========================================================================

    let total_latency = start.elapsed().as_millis() as u64;

    info!(
        request_id = %request_id,
        provider = ?selected_provider,
        model = %selected_model_name,
        latency_ms = total_latency,
        cost_cents = token_cost.total_cost_cents,
        saved_cents = cost_saved,
        savings_pct = savings_pct,
        "Request completed successfully"
    );

    Ok(Json(response))
}

// ============================================================================
// EXAMPLE API CALLS FOR YOUR MEETING ASSISTANT
// ============================================================================

/*

EXAMPLE 1: Let RouteFuel decide the best model
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "system",
        "content": "You are a meeting assistant."
      },
      {
        "role": "user",
        "content": "[15,000 token meeting transcript...] Summarize this meeting in 200 words"
      }
    ],
    "max_tokens": 200
  }'
```

Response includes:
```json
{
  "id": "...",
  "model": "claude-3-5-sonnet",  // ← RouteFuel selected this
  "choices": [...],
  "usage": {
    "prompt_tokens": 15050,
    "completion_tokens": 185,
    "total_tokens": 15235
  }
}
```

EXAMPLE 2: Task-based routing (for meeting assistant features)
```bash
# Summarization
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "task:summarize",
    "messages": [...]
  }'

# Answer questions
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "task:answer_question",
    "messages": [
      {
        "role": "user",
        "content": "What should I do about the budget issue?"
      }
    ]
  }'

# Extract action items
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "task:extract_action_items",
    "messages": [
      {
        "role": "user",
        "content": "[meeting transcript] Extract all action items"
      }
    ]
  }'

# Generate meeting response
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "task:generate_response",
    "messages": [
      {
        "role": "user",
        "content": "Generate a professional response to the client feedback"
      }
    ]
  }'
```

EXAMPLE 3: Specific model request
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [...]
  }'
```

*/
