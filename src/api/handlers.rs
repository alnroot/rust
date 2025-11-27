//! Request handlers for the API endpoints
//!
//! Each handler is an async function that processes HTTP requests
//! and returns JSON responses.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use std::sync::Arc;
use std::time::Instant;

use crate::application::EmbeddingService;
use crate::ports::SearchParams;
use super::auth::RequireApiKey;
use super::error::ApiResult;
use super::models::*;

/// Shared state that all handlers can access
#[derive(Clone)]
pub struct AppState {
    pub service: Arc<EmbeddingService>,
    pub start_time: Instant,
    pub version: String,
}

/// Simple health check - tells you the server is alive and how long it's been running
pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: state.version.clone(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

/// Returns stats about what's stored: how many embeddings, which model, etc.
pub async fn get_stats(
    _auth: RequireApiKey,
    State(state): State<AppState>,
) -> ApiResult<Json<StatsResponse>> {
    let stats = state.service.statistics().await?;

    Ok(Json(StatsResponse {
        total_embeddings: stats.total_embeddings,
        indexed_embeddings: stats.indexed_embeddings,
        model: stats.generator_model,
        dimensions: stats.dimensions,
    }))
}

/// Takes your text and turns it into an embedding vector
pub async fn create_embedding(
    _auth: RequireApiKey,
    State(state): State<AppState>,
    Json(req): Json<CreateEmbeddingRequest>,
) -> ApiResult<(StatusCode, Json<CreateEmbeddingResponse>)> {
    tracing::info!("Creating embedding for text: {} chars", req.text.len());

    // Generate the embedding using our model
    let embedding = state.service.create_embedding(&req.text).await?;

    let response = CreateEmbeddingResponse {
        id: embedding.id().to_string(),
        vector: embedding.vector().data().to_vec(),
        dimensions: embedding.vector().dimensions(),
        source: req.source.unwrap_or_else(|| embedding.metadata().source.clone()),
        model: embedding.metadata().model.clone(),
        created_at: embedding.metadata().created_at.to_rfc3339(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Create multiple embeddings in batch
/// POST /embeddings/batch
pub async fn create_embeddings_batch(
    _auth: RequireApiKey,
    State(state): State<AppState>,
    Json(req): Json<BatchCreateRequest>,
) -> ApiResult<(StatusCode, Json<BatchCreateResponse>)> {
    tracing::info!("Creating batch of {} embeddings", req.texts.len());

    let embeddings = state.service.create_embeddings_batch(&req.texts).await?;

    let ids: Vec<String> = embeddings.iter().map(|e| e.id().to_string()).collect();

    Ok((
        StatusCode::CREATED,
        Json(BatchCreateResponse {
            count: ids.len(),
            ids,
        }),
    ))
}

/// Get an embedding by ID
/// GET /embeddings/:id
pub async fn get_embedding(
    _auth: RequireApiKey,
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<EmbeddingResponse>> {
    tracing::debug!("Getting embedding: {}", id);

    let embedding = state
        .service
        .get_embedding(&id)
        .await?
        .ok_or_else(|| crate::error::EmbeddingError::not_found(&id))?;

    Ok(Json(EmbeddingResponse {
        id: embedding.id().to_string(),
        vector: embedding.vector().data().to_vec(),
        dimensions: embedding.vector().dimensions(),
        metadata: EmbeddingMetadata {
            source: embedding.metadata().source.clone(),
            model: embedding.metadata().model.clone(),
            tags: embedding.metadata().tags.clone(),
            created_at: embedding.metadata().created_at.to_rfc3339(),
        },
    }))
}

/// List all embeddings (without vector data)
/// GET /embeddings
pub async fn list_embeddings(
    _auth: RequireApiKey,
    State(state): State<AppState>,
) -> ApiResult<Json<ListEmbeddingsResponse>> {
    tracing::debug!("Listing embeddings");

    let embeddings = state.service.list_embeddings().await?;

    let items: Vec<EmbeddingListItem> = embeddings
        .iter()
        .map(|e| EmbeddingListItem {
            id: e.id().to_string(),
            source: e.metadata().source.clone(),
            model: e.metadata().model.clone(),
            dimensions: e.vector().dimensions(),
            created_at: e.metadata().created_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(ListEmbeddingsResponse {
        total: items.len(),
        embeddings: items,
    }))
}

/// Delete an embedding
/// DELETE /embeddings/:id
pub async fn delete_embedding(
    _auth: RequireApiKey,
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    tracing::info!("Deleting embedding: {}", id);

    state.service.delete_embedding(&id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Search for similar embeddings
/// POST /embeddings/search
pub async fn search_embeddings(
    _auth: RequireApiKey,
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    tracing::info!("Searching for: '{}' (k={})", req.query, req.k);

    let start = Instant::now();

    let params = SearchParams {
        k: req.k,
        threshold: req.threshold,
        metric: req.metric,
    };

    let results = state.service.find_similar(&req.query, params).await?;

    let query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let items: Vec<SearchResultItem> = results
        .iter()
        .map(|r| SearchResultItem {
            id: r.embedding.id().to_string(),
            score: r.score,
            source: r.embedding.metadata().source.clone(),
            text: None, // Could be populated from metadata if stored
        })
        .collect();

    Ok(Json(SearchResponse {
        total: items.len(),
        results: items,
        query_time_ms,
    }))
}
