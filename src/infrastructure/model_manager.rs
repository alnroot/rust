//! Model Manager - Handles automatic model downloading and caching
//!
//! This module provides automatic downloading of ONNX models from Hugging Face
//! and caches them locally for fast subsequent loads.

use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use crate::error::{EmbeddingError, Result};

/// Information about a pre-configured model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name/identifier
    pub name: String,

    /// Hugging Face repository ID
    pub repo_id: String,

    /// Model file name in the repository
    pub model_file: String,

    /// Tokenizer file name
    pub tokenizer_file: String,

    /// Expected embedding dimensions
    pub dimensions: usize,

    /// Whether this model supports Matryoshka embeddings (truncation)
    pub supports_truncation: bool,
}

/// Get pre-configured models ready to use
///
/// Uses ONNX-optimized models from the Xenova namespace (optimum-converted)
pub fn get_available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            name: "all-MiniLM-L6-v2".to_string(),
            repo_id: "Xenova/all-MiniLM-L6-v2".to_string(),
            model_file: "onnx/model_quantized.onnx".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            dimensions: 384,
            supports_truncation: false,
        },
        ModelInfo {
            name: "bge-small-en-v1.5".to_string(),
            repo_id: "Xenova/bge-small-en-v1.5".to_string(),
            model_file: "onnx/model_quantized.onnx".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            dimensions: 384,
            supports_truncation: false,
        },
        ModelInfo {
            name: "all-mpnet-base-v2".to_string(),
            repo_id: "Xenova/all-mpnet-base-v2".to_string(),
            model_file: "onnx/model_quantized.onnx".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            dimensions: 768,
            supports_truncation: false,
        },
    ]
}

/// Manages model downloading and caching
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| EmbeddingError::config("Could not determine cache directory"))?
            .join("rust-embeddings");

        fs::create_dir_all(&cache_dir)?;

        tracing::info!("Model cache directory: {:?}", cache_dir);

        Ok(Self { cache_dir })
    }

    /// Create a model manager with custom cache directory
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        fs::create_dir_all(&cache_dir)?;

        Ok(Self { cache_dir })
    }

    /// Get the path for a model, downloading it if necessary
    pub async fn ensure_model(&self, model_name: &str) -> Result<PathBuf> {
        // Find model info
        let model_info = self.find_model_info(model_name)?;

        let model_dir = self.cache_dir.join(&model_info.name);

        // Check if model already exists
        if self.is_model_downloaded(&model_dir, &model_info) {
            tracing::info!("Model '{}' found in cache", model_name);
            return Ok(model_dir);
        }

        // Download the model
        tracing::info!("Model '{}' not found in cache, downloading...", model_name);
        self.download_model(&model_info, &model_dir).await?;

        Ok(model_dir)
    }

    /// Check if a model is already downloaded
    fn is_model_downloaded(&self, model_dir: &Path, _model_info: &ModelInfo) -> bool {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        model_path.exists() && tokenizer_path.exists()
    }

    /// Download a model from Hugging Face
    async fn download_model(&self, model_info: &ModelInfo, dest_dir: &Path) -> Result<()> {
        fs::create_dir_all(dest_dir)?;

        tracing::info!("Downloading model from {}", model_info.repo_id);

        // Download model file
        let model_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_info.repo_id, model_info.model_file
        );

        tracing::info!("Downloading model file from: {}", model_url);
        self.download_file(&model_url, &dest_dir.join("model.onnx")).await?;

        // Download tokenizer file
        let tokenizer_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_info.repo_id, model_info.tokenizer_file
        );

        tracing::info!("Downloading tokenizer from: {}", tokenizer_url);
        self.download_file(&tokenizer_url, &dest_dir.join("tokenizer.json")).await?;

        // Save model info
        let info_path = dest_dir.join("model_info.json");
        let info_json = serde_json::to_string_pretty(model_info)?;
        fs::write(info_path, info_json)?;

        tracing::info!("Model '{}' downloaded successfully", model_info.name);

        Ok(())
    }

    /// Download a file from a URL
    async fn download_file(&self, url: &str, dest: &Path) -> Result<()> {
        let response = reqwest::get(url)
            .await
            .map_err(|e| EmbeddingError::generation(format!("Failed to download file: {}", e)))?;

        if !response.status().is_success() {
            return Err(EmbeddingError::generation(format!(
                "Failed to download file: HTTP {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| EmbeddingError::generation(format!("Failed to read response: {}", e)))?;

        fs::write(dest, bytes)?;

        Ok(())
    }

    /// Find model information by name or repo ID
    fn find_model_info(&self, identifier: &str) -> Result<ModelInfo> {
        // Check pre-configured models
        let available_models = get_available_models();
        if let Some(info) = available_models
            .iter()
            .find(|m| m.name == identifier || m.repo_id == identifier)
        {
            return Ok(info.clone());
        }

        // If not found, create a ModelInfo from repo ID
        // Assume it follows sentence-transformers convention
        Ok(ModelInfo {
            name: identifier.split('/').last().unwrap_or(identifier).to_string(),
            repo_id: identifier.to_string(),
            model_file: "onnx/model_quantized.onnx".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            dimensions: 384, // Default
            supports_truncation: false,
        })
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// List downloaded models
    pub fn list_downloaded_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(models);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    models.push(name.to_string());
                }
            }
        }

        Ok(models)
    }

    /// Delete a model from cache
    pub fn delete_model(&self, model_name: &str) -> Result<()> {
        let model_dir = self.cache_dir.join(model_name);

        if model_dir.exists() {
            fs::remove_dir_all(model_dir)?;
            tracing::info!("Deleted model '{}'  from cache", model_name);
        }

        Ok(())
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().expect("Failed to create ModelManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_model_info() {
        let manager = ModelManager::new().unwrap();

        // Test pre-configured model
        let info = manager.find_model_info("all-MiniLM-L6-v2").unwrap();
        assert_eq!(info.dimensions, 384);

        // Test by repo ID
        let info = manager.find_model_info("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        assert_eq!(info.name, "all-MiniLM-L6-v2");
    }
}
