//! REST API Layer
//!
//! Provides HTTP REST API endpoints for the embedding service.

pub mod server;
pub mod routes;
pub mod handlers;
pub mod models;
pub mod error;
pub mod auth;

pub use server::ApiServer;
pub use models::{CreateEmbeddingRequest, CreateEmbeddingResponse, SearchRequest, SearchResponse};
pub use auth::{ApiKeyConfig, RequireApiKey};
