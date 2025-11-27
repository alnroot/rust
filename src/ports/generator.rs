//! Embedding Generator Port
//!
//! Defines the interface for generating embeddings from text.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::domain::Embedding;
use crate::error::Result;

/// Backend type for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelBackend {
    /// ONNX local model (best performance)
    Onnx {
        model_name: String,
        /// Optional: truncate dimensions for Matryoshka models
        truncate_dim: Option<usize>,
    },
    /// HTTP API (Docker or remote service)
    Http {
        url: String,
    },
    /// OpenAI API
    OpenAI {
        api_key: String,
        model: String,
    },
}

/// Configuration for embedding generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Backend to use
    pub backend: ModelBackend,

    /// Expected embedding dimensions
    pub dimensions: usize,

    /// Maximum batch size for batch processing
    pub max_batch_size: usize,

    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            backend: ModelBackend::Onnx {
                model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                truncate_dim: None,
            },
            dimensions: 384,
            max_batch_size: 32,
            num_threads: None,
        }
    }
}

impl GeneratorConfig {
    pub fn builder() -> GeneratorConfigBuilder {
        GeneratorConfigBuilder::default()
    }
}

/// Builder for GeneratorConfig
#[derive(Default)]
pub struct GeneratorConfigBuilder {
    backend: Option<ModelBackend>,
    dimensions: Option<usize>,
    max_batch_size: Option<usize>,
    num_threads: Option<usize>,
}

impl GeneratorConfigBuilder {
    pub fn backend(mut self, backend: ModelBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn onnx(mut self, model_name: impl Into<String>) -> Self {
        self.backend = Some(ModelBackend::Onnx {
            model_name: model_name.into(),
            truncate_dim: None,
        });
        self
    }

    pub fn onnx_with_truncation(mut self, model_name: impl Into<String>, truncate_dim: usize) -> Self {
        self.backend = Some(ModelBackend::Onnx {
            model_name: model_name.into(),
            truncate_dim: Some(truncate_dim),
        });
        self
    }

    pub fn http(mut self, url: impl Into<String>) -> Self {
        self.backend = Some(ModelBackend::Http {
            url: url.into(),
        });
        self
    }

    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }

    pub fn build(self) -> GeneratorConfig {
        let backend = self.backend.unwrap_or_else(|| ModelBackend::Onnx {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            truncate_dim: None,
        });

        let dimensions = self.dimensions.unwrap_or(384);
        let max_batch_size = self.max_batch_size.unwrap_or(32);

        GeneratorConfig {
            backend,
            dimensions,
            max_batch_size,
            num_threads: self.num_threads,
        }
    }
}

/// Trait for generating embeddings
#[async_trait]
pub trait EmbeddingGenerator: Send + Sync {
    /// Generate embedding from text
    async fn generate_from_text(&self, text: &str) -> Result<Embedding>;

    /// Generate embeddings from multiple texts (batch processing)
    async fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>>;

    /// Get the model name or identifier
    fn model_name(&self) -> &str;

    /// Get the embedding dimensions
    fn dimensions(&self) -> usize;

    /// Check if the generator supports batching efficiently
    fn supports_batching(&self) -> bool {
        true
    }
}
