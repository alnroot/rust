//! Error types for the embeddings library
//!
//! This module defines all error types using the thiserror crate
//! for clean and idiomatic error handling.

use thiserror::Error;

/// Result type alias for embedding operations
pub type Result<T> = std::result::Result<T, EmbeddingError>;

/// Main error type for the embeddings library
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// Embedding not found
    #[error("Embedding not found: {0}")]
    NotFound(String),

    /// Invalid embedding dimensions
    #[error("Invalid embedding dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    /// Invalid vector data
    #[error("Invalid vector data: {0}")]
    InvalidVector(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Generation error
    #[error("Failed to generate embedding: {0}")]
    GenerationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Generic error
    #[error("An error occurred: {0}")]
    Other(#[from] anyhow::Error),
}

impl EmbeddingError {
    /// Create a not found error
    pub fn not_found(id: impl Into<String>) -> Self {
        Self::NotFound(id.into())
    }

    /// Create an invalid dimensions error
    pub fn invalid_dimensions(expected: usize, actual: usize) -> Self {
        Self::InvalidDimensions { expected, actual }
    }

    /// Create an invalid vector error
    pub fn invalid_vector(msg: impl Into<String>) -> Self {
        Self::InvalidVector(msg.into())
    }

    /// Create a storage error
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::StorageError(msg.into())
    }

    /// Create a generation error
    pub fn generation(msg: impl Into<String>) -> Self {
        Self::GenerationError(msg.into())
    }

    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }

    /// Create a processing error
    pub fn processing(msg: impl Into<String>) -> Self {
        Self::ProcessingError(msg.into())
    }
}
