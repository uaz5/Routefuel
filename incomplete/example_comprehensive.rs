// ============================================================================
// COMPREHENSIVE EXAMPLE: RouteFuel in Production
// ============================================================================
//
// This example demonstrates:
// 1. Initializing the routing engine with multiple models
// 2. Processing requests with different priorities
// 3. Cost tracking and optimization
// 4. Real-time analytics
//
// ============================================================================

use routefuel::{
    RouteEngine, ModelConfig, ModelProvider, RoutingContext, UserTier, RequestPriority,
    AsyncRequestHandler, RoutingRequest, CostTracker, BatchProcessor,
};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for observability
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Initializing RouteFuel Engine...\n");

    // ========================================================================
    // STEP 1: Create the RouteEngine and register models
    // ========================================================================

    let engine = Arc::new(RouteEngine::new("gpt-4o".to_string()));

    // Define our model lineup
    let models = vec![
        ModelConfig {
            id: "gpt-4o".to_string(),
            name: "GPT-4o (Baseline)".to_string(),
            provider: ModelProvider::OpenAI,
            cost_per_1m_input_tokens: 3.0,
            cost_per_1m_output_tokens: 6.0,
            latency_ms: 200,
            throughput: 5.0,
            context_window: 128000,
            enabled: true,
            quality_score: 1.0,
        },
        ModelConfig {
            id: "deepseek-v3".to_string(),
            name: "DeepSeek V3".to_string(),
            provider: ModelProvider::DeepSeek,
            cost_per_1m_input_tokens: 0.14,
            cost_per_1m_output_tokens: 0.28,
            latency_ms: 150,
            throughput: 8.0,
            context_window: 64000,
            enabled: true,
            quality_score: 0.95,
        },
        ModelConfig {
            id: "gpt-4-turbo".to_string(),
            name: "GPT-4 Turbo".to_string(),
            provider: ModelProvider::OpenAI,
            cost_per_1m_input_tokens: 1.0,
            cost_per_1m_output_tokens: 3.0,
            latency_ms: 180,
            throughput: 6.0,
            context_window: 128000,
            enabled: true,
            quality_score: 0.98,
        },
        ModelConfig {
            id: "claude-3-opus".to_string(),
            name: "Claude 3 Opus".to_string(),
            provider: ModelProvider::Anthropic,
            cost_per_1m_input_tokens: 1.5,
            cost_per_1m_output_tokens: 7.5,
            latency_ms: 220,
            throughput: 4.5,
            context_window: 200000,
            enabled: true,
            quality_score: 1.0,
        },
    ];

    // Register all models
    engine.register_models(models).await?;
    println!("✓ Registered {} models", 4);

    // ========================================================================
    // STEP 2: Create async handler and cost tracker
    // ========================================================================

    let handler = Arc::new(AsyncRequestHandler::new(engine.clone(), 50));
    let tracker = CostTracker::new(10000);

    println!(
        "✓ Created AsyncRequestHandler with max concurrency: {}",
        handler.stats().max_concurrent
    );
    println!("✓ Created CostTracker with 10k record capacity\n");

    // ========================================================================
    // STEP 3: Demonstrate different routing scenarios
    // ========================================================================

    println!("📊 ROUTING SCENARIOS:\n");

    // Scenario A: Cost-optimized routing
    println!("Scenario A: Free-tier user, cost-optimized routing");
    let request_a = RoutingRequest {
        request_id: Uuid::new_v4().to_string(),
        prompt: "Summarize the benefits of renewable energy in 100 words".to_string(),
        estimated_tokens: Some(500),
        user_tier: UserTier::Free,
        priority: RequestPriority::Cost,
        shadow_mode: false,
        timeout_ms: 5000,
    };

    let response_a = handler.handle_request(request_a).await;
    print_response(&response_a);

    // Record the decision
    tracker
        .record_decision(
            response_a.request_id.clone(),
            response_a.decision.selected_model_id.clone(),
            response_a.decision.metadata.cost_analysis.selected_model_cost,
            response_a.decision.metadata.cost_analysis.baseline_cost,
            500,
            "Free".to_string(),
        )
        .await?;

    println!();

    // Scenario B: Balanced routing
    println!("Scenario B: Pro-tier user, balanced routing");
    let request_b = RoutingRequest {
        request_id: Uuid::new_v4().to_string(),
        prompt: "Write a technical specification for a distributed cache system".to_string(),
        estimated_tokens: Some(2000),
        user_tier: UserTier::Pro,
        priority: RequestPriority::Balanced,
        shadow_mode: true, // Enable shadow mode for A/B testing
        timeout_ms: 5000,
    };

    let response_b = handler.handle_request(request_b).await;
    print_response(&response_b);

    tracker
        .record_decision(
            response_b.request_id.clone(),
            response_b.decision.selected_model_id.clone(),
            response_b.decision.metadata.cost_analysis.selected_model_cost,
            response_b.decision.metadata.cost_analysis.baseline_cost,
            2000,
            "Pro".to_string(),
        )
        .await?;

    println!();

    // Scenario C: Quality-optimized routing
    println!("Scenario C: Enterprise user, quality-optimized routing");
    let request_c = RoutingRequest {
        request_id: Uuid::new_v4().to_string(),
        prompt: "Design a novel neural architecture for visual understanding".to_string(),
        estimated_tokens: Some(3000),
        user_tier: UserTier::Enterprise,
        priority: RequestPriority::Quality,
        shadow_mode: true,
        timeout_ms: 10000,
    };

    let response_c = handler.handle_request(request_c).await;
    print_response(&response_c);

    tracker
        .record_decision(
            response_c.request_id.clone(),
            response_c.decision.selected_model_id.clone(),
            response_c.decision.metadata.cost_analysis.selected_model_cost,
            response_c.decision.metadata.cost_analysis.baseline_cost,
            3000,
            "Enterprise".to_string(),
        )
        .await?;

    println!();

    // ========================================================================
    // STEP 4: Batch processing with analytics
    // ========================================================================

    println!("📦 BATCH PROCESSING:\n");

    let batch_requests = vec![
        RoutingRequest {
            request_id: Uuid::new_v4().to_string(),
            prompt: "Q1: What is machine learning?".to_string(),
            estimated_tokens: Some(300),
            user_tier: UserTier::Free,
            priority: RequestPriority::Cost,
            shadow_mode: false,
            timeout_ms: 5000,
        },
        RoutingRequest {
            request_id: Uuid::new_v4().to_string(),
            prompt: "Q2: Explain transformer architectures in detail".to_string(),
            estimated_tokens: Some(1500),
            user_tier: UserTier::Pro,
            priority: RequestPriority::Balanced,
            shadow_mode: true,
            timeout_ms: 5000,
        },
        RoutingRequest {
            request_id: Uuid::new_v4().to_string(),
            prompt: "Q3: Design a scalable recommendation system".to_string(),
            estimated_tokens: Some(2500),
            user_tier: UserTier::Enterprise,
            priority: RequestPriority::Quality,
            shadow_mode: true,
            timeout_ms: 10000,
        },
    ];

    let batch_processor = BatchProcessor::new(handler.clone()).enable_shadow_mode();
    let analytics = batch_processor
        .process_batch_with_analytics(batch_requests)
        .await;

    println!("Batch Analytics:");
    println!("  Total requests: {}", analytics.total_requests);
    println!("  Successful: {}", analytics.successful);
    println!("  Failed: {}", analytics.failed);
    println!(
        "  Average processing time: {}ms",
        analytics.avg_processing_time_ms
    );
    println!(
        "  Total potential savings: ${:.4}",
        analytics.total_potential_savings / 100.0
    );

    // Record batch requests
    for response in &analytics.responses {
        if response.success {
            tracker
                .record_decision(
                    response.request_id.clone(),
                    response.decision.selected_model_id.clone(),
                    response.decision.metadata.cost_analysis.selected_model_cost,
                    response.decision.metadata.cost_analysis.baseline_cost,
                    1000,
                    "Mixed".to_string(),
                )
                .await?;
        }
    }

    println!();

    // ========================================================================
    // STEP 5: Cost tracking and analytics
    // ========================================================================

    println!("💰 COST ANALYTICS:\n");

    let all_stats = tracker.get_all_model_stats().await;
    println!("Model Statistics:");
    for stat in &all_stats {
        println!(
            "  {} - Selections: {}, Total Cost: ${:.4}, Avg Cost: ${:.6}, Savings: ${:.4}",
            stat.model_id,
            stat.selection_count,
            stat.total_cost / 100.0,
            stat.avg_cost / 100.0,
            stat.total_savings / 100.0
        );
    }

    println!();

    // Get current time for summary query
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;

    let summary = tracker.get_cost_summary(now - 86400, now + 86400).await;
    println!("24-Hour Cost Summary:");
    println!("  Total requests: {}", summary.request_count);
    println!("  Total cost: ${:.4}", summary.total_cost / 100.0);
    println!("  Total savings: ${:.4}", summary.total_savings / 100.0);
    println!(
        "  Avg cost per request: ${:.6}",
        summary.avg_cost_per_request / 100.0
    );

    println!();

    // ========================================================================
    // STEP 6: Shadow mode results
    // ========================================================================

    if !analytics.shadow_results.is_empty() {
        println!("🔄 SHADOW MODE RESULTS:\n");
        for result in &analytics.shadow_results {
            println!(
                "Primary: {} | Shadow: {} | Cost Difference: ${:.6}",
                result.primary_model_id,
                result.shadow_model_id,
                (result.shadow_cost - result.primary_cost) / 100.0
            );
        }
        println!();
    }

    // ========================================================================
    // STEP 7: Concurrency stats
    // ========================================================================

    let stats = handler.stats();
    println!("📈 CONCURRENCY STATS:\n");
    println!(
        "  Available permits: {}/{}",
        stats.available_permits, stats.max_concurrent
    );
    println!(
        "  Utilization: {:.2}%",
        stats.utilization_percent
    );

    println!("\n✅ RouteFuel example completed successfully!");

    Ok(())
}

fn print_response(response: &routefuel::RoutingResponse) {
    if response.success {
        println!(
            "  ✓ Selected: {} (confidence: {:.2}%)",
            response.decision.selected_model_id,
            response.decision.confidence * 100.0
        );
        println!("    Reason: {}", response.decision.reason);
        println!(
            "    Cost: ${:.6} | Baseline: ${:.6} | Savings: ${:.6}",
            response.decision.metadata.cost_analysis.selected_model_cost / 100.0,
            response.decision.metadata.cost_analysis.baseline_cost / 100.0,
            response.decision.metadata.cost_analysis.potential_savings / 100.0
        );
        println!(
            "    Savings %: {:.2}%",
            response.decision.metadata.cost_analysis.savings_percentage
        );

        if let Some(shadow_model) = &response.decision.shadow_model_id {
            println!("    Shadow Model: {}", shadow_model);
        }

        println!(
            "    Processing Time: {}ms",
            response.processing_time_ms
        );
    } else {
        println!(
            "  ✗ Error: {}",
            response.error.as_ref().unwrap_or(&"Unknown error".to_string())
        );
    }
}
