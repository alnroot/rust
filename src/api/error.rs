//! API Error Handling

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use crate::error::EmbeddingError;
use super::models::ErrorResponse;

/// API Error wrapper
pub struct ApiError(EmbeddingError);

impl From<EmbeddingError> for ApiError {
    fn from(err: EmbeddingError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self.0 {
            EmbeddingError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg.clone()),
            EmbeddingError::InvalidDimensions { .. } => {
                (StatusCode::BAD_REQUEST, "invalid_dimensions", self.0.to_string())
            }
            EmbeddingError::InvalidVector(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_vector", msg.clone())
            }
            EmbeddingError::ValidationError(msg) => {
                (StatusCode::BAD_REQUEST, "validation_error", msg.clone())
            }
            EmbeddingError::StorageError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "storage_error", msg.clone())
            }
            EmbeddingError::GenerationError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "generation_error", msg.clone())
            }
            EmbeddingError::ConfigError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "config_error", msg.clone())
            }
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "An internal error occurred".to_string(),
            ),
        };

        let body = Json(ErrorResponse {
            error: error_type.to_string(),
            message,
            details: None,
        });

        (status, body).into_response()
    }
}

/// Result type for API handlers
pub type ApiResult<T> = Result<T, ApiError>;
