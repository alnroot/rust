//! API Routes Configuration

use axum::{
    routing::{get, post, delete},
    Extension, Router,
};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    compression::CompressionLayer,
};

use super::auth::{ApiKeyConfig, ApiKeyState};
use super::handlers::{self, AppState};

/// Rate limiting configuration
#[derive(Clone)]
pub struct RateLimitConfig {
    /// Maximum concurrent requests (acts as rate limiter)
    pub max_concurrent_requests: usize,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            // 100 concurrent requests by default
            max_concurrent_requests: 100,
        }
    }
}

/// Combined server configuration
#[derive(Clone)]
pub struct RouterConfig {
    pub rate_limit: RateLimitConfig,
    pub api_keys: ApiKeyConfig,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            rate_limit: RateLimitConfig::default(),
            api_keys: ApiKeyConfig::from_env(),
        }
    }
}

/// Build the application router with all routes (uses env for config)
pub fn create_router(state: AppState) -> Router {
    create_router_with_config(state, RouterConfig::default())
}

/// Build the application router with custom configuration
pub fn create_router_with_config(state: AppState, config: RouterConfig) -> Router {
    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(handlers::health_check));

    // Protected routes (require auth if enabled)
    let protected_routes = Router::new()
        .route("/stats", get(handlers::get_stats))
        // Embeddings CRUD
        .route("/embeddings", post(handlers::create_embedding))
        .route("/embeddings", get(handlers::list_embeddings))
        .route("/embeddings/:id", get(handlers::get_embedding))
        .route("/embeddings/:id", delete(handlers::delete_embedding))
        // Batch operations
        .route("/embeddings/batch", post(handlers::create_embeddings_batch))
        // Search
        .route("/embeddings/search", post(handlers::search_embeddings));

    // Combine routes
    let api_routes = Router::new()
        .merge(public_routes)
        .merge(protected_routes);

    // Build middleware stack
    // Note: For production rate limiting (requests/second), consider using
    // `tower_governor` crate which provides proper sliding window rate limiting.
    // ConcurrencyLimitLayer limits concurrent in-flight requests.
    let middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive()) // TODO: Configure for production
        .layer(CompressionLayer::new())
        .layer(ConcurrencyLimitLayer::new(config.rate_limit.max_concurrent_requests))
        .layer(Extension(ApiKeyState(Arc::new(config.api_keys))));

    // Main router with middleware
    Router::new()
        .nest("/api/v1", api_routes)
        .layer(middleware_stack)
        .with_state(state)
}
