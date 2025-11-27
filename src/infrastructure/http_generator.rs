//! HTTP Embedding Generator - Connect to external API
//!
//! This generator connects to an HTTP API (Docker, cloud service, etc.)
//! for generating embeddings.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{Embedding, EmbeddingMetadata, Vector};
use crate::error::{EmbeddingError, Result};
use crate::ports::EmbeddingGenerator;

#[derive(Serialize)]
struct EmbeddingRequest {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

/// HTTP-based embedding generator
pub struct HttpEmbeddingGenerator {
    url: String,
    model_name: String,
    dimensions: usize,
    client: reqwest::Client,
}

impl HttpEmbeddingGenerator {
    /// Create a new HTTP generator
    ///
    /// # Arguments
    /// * `url` - Base URL of the embedding service
    /// * `model_name` - Name of the model (for metadata)
    /// * `dimensions` - Expected dimensions
    pub fn new(url: impl Into<String>, model_name: impl Into<String>, dimensions: usize) -> Self {
        Self {
            url: url.into(),
            model_name: model_name.into(),
            dimensions,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingGenerator for HttpEmbeddingGenerator {
    async fn generate_from_text(&self, text: &str) -> Result<Embedding> {
        let request = EmbeddingRequest {
            texts: vec![text.to_string()],
        };

        let response = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::generation(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(EmbeddingError::generation(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let response_data: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::generation(format!("Failed to parse response: {}", e)))?;

        let embedding_vec = response_data
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::generation("No embeddings in response"))?;

        let vector = Vector::new(embedding_vec)?;
        let metadata = EmbeddingMetadata::new(text, &self.model_name);

        Ok(Embedding::new(vector, metadata))
    }

    async fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let request = EmbeddingRequest {
            texts: texts.to_vec(),
        };

        let response = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::generation(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(EmbeddingError::generation(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let response_data: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::generation(format!("Failed to parse response: {}", e)))?;

        let embeddings = response_data
            .embeddings
            .into_iter()
            .zip(texts.iter())
            .map(|(vec, text)| {
                let vector = Vector::new(vec)?;
                let metadata = EmbeddingMetadata::new(text, &self.model_name);
                Ok(Embedding::new(vector, metadata))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(embeddings)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
