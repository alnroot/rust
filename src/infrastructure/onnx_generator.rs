//! ONNX Embedding Generator - Local model inference
//!
//! High-performance embedding generation using ONNX Runtime.
//! Supports:
//! - Quantized models (INT8) for faster inference
//! - Batch processing with Rayon
//! - Matryoshka embedding truncation
//! - **Session pool for true parallel inference** (no single Mutex bottleneck)

use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::path::PathBuf;
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Value};
use tokenizers::Tokenizer;
use rayon::prelude::*;

use crate::domain::{Embedding, EmbeddingMetadata, Vector};
use crate::error::{EmbeddingError, Result};
use crate::ports::EmbeddingGenerator;
use super::model_manager::{ModelManager, ModelInfo};

/// Execution mode for the ONNX generator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    /// Single session with maximum internal parallelism.
    /// Best for: batch processing, CLI tools, single-user scenarios.
    /// Maximizes throughput for sequential workloads.
    Batch,

    /// Multiple sessions for concurrent request handling.
    /// Best for: API servers, multi-user scenarios, worker queues.
    /// Minimizes latency under concurrent load.
    Server,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Batch
    }
}

/// Pool of ONNX sessions for parallel inference
///
/// Each session can run inference independently, eliminating the
/// single-Mutex bottleneck that serializes all inference calls.
struct SessionPool {
    sessions: Vec<Mutex<Session>>,
    next_index: AtomicUsize,
}

impl SessionPool {
    /// Create a new session pool
    fn new(sessions: Vec<Session>) -> Self {
        Self {
            sessions: sessions.into_iter().map(Mutex::new).collect(),
            next_index: AtomicUsize::new(0),
        }
    }

    /// Get the next session using round-robin selection
    /// Returns the index and a reference to the Mutex<Session>
    fn get_session(&self) -> (usize, &Mutex<Session>) {
        let idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        (idx, &self.sessions[idx])
    }
}

/// ONNX-based embedding generator
///
/// Supports two execution modes optimized for different deployment patterns:
/// - `Batch`: Single session with max internal parallelism (CLI, batch jobs)
/// - `Server`: Multiple sessions for concurrent requests (API servers)
///
/// # Example
/// ```no_run
/// use rust_embeddings::infrastructure::{OnnxEmbeddingGenerator, ExecutionMode};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // For batch processing (default, optimal for throughput)
/// let batch_gen = OnnxEmbeddingGenerator::new("all-MiniLM-L6-v2", None).await?;
///
/// // For API server (optimal for concurrent requests)
/// let server_gen = OnnxEmbeddingGenerator::with_mode(
///     "all-MiniLM-L6-v2",
///     None,
///     ExecutionMode::Server,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct OnnxEmbeddingGenerator {
    /// Pool of ONNX sessions for parallel inference
    session_pool: Arc<SessionPool>,

    /// Tokenizer (thread-safe, stateless for encoding)
    tokenizer: Arc<Tokenizer>,

    /// Model name
    model_name: String,

    /// Path to model directory (for creating additional sessions)
    model_path: PathBuf,

    /// Embedding dimensions
    dimensions: usize,

    /// Optional truncation for Matryoshka models
    truncate_dim: Option<usize>,

    /// Maximum batch size
    max_batch_size: usize,

    /// Number of sessions in the pool
    pool_size: usize,

    /// Execution mode
    mode: ExecutionMode,
}

impl OnnxEmbeddingGenerator {
    /// Create a new ONNX generator with default settings (Batch mode)
    ///
    /// Uses `ExecutionMode::Batch` which is optimal for:
    /// - CLI tools and scripts
    /// - Batch processing pipelines
    /// - Single-user scenarios
    ///
    /// For API servers with concurrent requests, use `with_mode(ExecutionMode::Server)`.
    ///
    /// # Example
    /// ```no_run
    /// use rust_embeddings::infrastructure::OnnxEmbeddingGenerator;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let generator = OnnxEmbeddingGenerator::new("all-MiniLM-L6-v2", None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(model_name: &str, truncate_dim: Option<usize>) -> Result<Self> {
        Self::with_mode(model_name, truncate_dim, ExecutionMode::Batch).await
    }

    /// Create a new ONNX generator with a specific execution mode
    ///
    /// # Execution Modes
    ///
    /// - `ExecutionMode::Batch`: Single session, all CPU threads for internal parallelism.
    ///   Best throughput for sequential batch processing.
    ///
    /// - `ExecutionMode::Server`: Multiple sessions (num_cpus/2), fewer threads each.
    ///   Best latency under concurrent load (API servers).
    ///
    /// # Example
    /// ```no_run
    /// use rust_embeddings::infrastructure::{OnnxEmbeddingGenerator, ExecutionMode};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // For an API server handling concurrent requests
    /// let generator = OnnxEmbeddingGenerator::with_mode(
    ///     "all-MiniLM-L6-v2",
    ///     None,
    ///     ExecutionMode::Server,
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_mode(
        model_name: &str,
        truncate_dim: Option<usize>,
        mode: ExecutionMode,
    ) -> Result<Self> {
        let pool_size = match mode {
            ExecutionMode::Batch => 1,  // Single session, max internal parallelism
            ExecutionMode::Server => (num_cpus::get() / 2).max(2).min(8),  // Multiple sessions
        };
        Self::with_config(model_name, truncate_dim, mode, pool_size).await
    }

    /// Create a new ONNX generator with full configuration control
    ///
    /// For advanced users who need fine-grained control over pool size.
    pub async fn with_pool_size(
        model_name: &str,
        truncate_dim: Option<usize>,
        pool_size: usize,
    ) -> Result<Self> {
        let mode = if pool_size == 1 {
            ExecutionMode::Batch
        } else {
            ExecutionMode::Server
        };
        Self::with_config(model_name, truncate_dim, mode, pool_size).await
    }

    /// Internal constructor with all configuration options
    async fn with_config(
        model_name: &str,
        truncate_dim: Option<usize>,
        mode: ExecutionMode,
        pool_size: usize,
    ) -> Result<Self> {
        let pool_size = pool_size.max(1);

        let manager = ModelManager::new()?;
        let model_dir = manager.ensure_model(model_name).await?;

        let model_path = model_dir.join("model.onnx");

        // Create session pool with appropriate thread configuration
        let sessions = Self::create_session_pool(&model_path, pool_size, mode)?;
        let session_pool = SessionPool::new(sessions);

        // Load tokenizer (shared across all sessions - it's stateless for encoding)
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbeddingError::generation(format!("Failed to load tokenizer: {}", e)))?;

        // Try to load model info for dimensions
        let info_path = model_dir.join("model_info.json");
        let dimensions = if info_path.exists() {
            let info_json = std::fs::read_to_string(&info_path)?;
            let info: ModelInfo = serde_json::from_str(&info_json)?;
            info.dimensions
        } else {
            384 // Default dimensions
        };

        // Validate truncation
        if let Some(truncate) = truncate_dim {
            if truncate > dimensions {
                return Err(EmbeddingError::config(format!(
                    "Truncation dimension {} exceeds model dimensions {}",
                    truncate, dimensions
                )));
            }
        }

        tracing::info!(
            "ONNX generator ready: mode={:?}, sessions={}, dimensions={}{}",
            mode,
            pool_size,
            dimensions,
            truncate_dim.map(|d| format!(" (truncated to {})", d)).unwrap_or_default()
        );

        Ok(Self {
            session_pool: Arc::new(session_pool),
            tokenizer: Arc::new(tokenizer),
            model_name: model_name.to_string(),
            model_path,
            dimensions,
            truncate_dim,
            max_batch_size: 32,
            pool_size,
            mode,
        })
    }

    /// Create ONNX sessions with mode-appropriate thread configuration
    fn create_session_pool(
        model_path: &PathBuf,
        pool_size: usize,
        mode: ExecutionMode,
    ) -> Result<Vec<Session>> {
        let mut sessions = Vec::with_capacity(pool_size);
        let total_threads = num_cpus::get();

        // Thread allocation strategy based on mode
        let threads_per_session = match mode {
            // Batch: single session gets all threads for max internal parallelism
            ExecutionMode::Batch => total_threads,
            // Server: distribute threads across sessions, min 1 each
            ExecutionMode::Server => (total_threads / pool_size).max(1),
        };

        for i in 0..pool_size {
            let session = Session::builder()
                .map_err(|e| EmbeddingError::generation(format!("Failed to create ONNX builder: {}", e)))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| EmbeddingError::generation(format!("Failed to set optimization: {}", e)))?
                .with_inter_threads(threads_per_session)
                .map_err(|e| EmbeddingError::generation(format!("Failed to set threads: {}", e)))?
                .commit_from_file(model_path)
                .map_err(|e| EmbeddingError::generation(format!(
                    "Failed to load ONNX model for session {}: {}", i, e
                )))?;

            sessions.push(session);
        }

        tracing::debug!(
            "Created {} ONNX sessions ({:?} mode) with {} threads each",
            pool_size,
            mode,
            threads_per_session
        );

        Ok(sessions)
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Tokenize text
    fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| EmbeddingError::generation(format!("Tokenization failed: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        Ok((input_ids, attention_mask))
    }

    /// Get number of sessions in the pool
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Run inference on tokenized input using a session from the pool
    fn run_inference(&self, input_ids: Vec<i64>, attention_mask: Vec<i64>) -> Result<Vec<f32>> {
        let batch_size = 1;
        let seq_len = input_ids.len();

        // Clone attention_mask before moving it, as we need it for mean pooling
        let attention_mask_for_pooling = attention_mask.clone();

        // Create input tensors using tuple format (shape, data)
        let input_ids_value = Value::from_array((
            vec![batch_size, seq_len],
            input_ids
        )).map_err(|e| EmbeddingError::generation(format!("Failed to create input tensor: {}", e)))?;

        let attention_mask_value = Value::from_array((
            vec![batch_size, seq_len],
            attention_mask
        )).map_err(|e| EmbeddingError::generation(format!("Failed to create attention tensor: {}", e)))?;

        // Token type IDs (all zeros for single-sentence encoding)
        let token_type_ids: Vec<i64> = vec![0; seq_len];
        let token_type_ids_value = Value::from_array((
            vec![batch_size, seq_len],
            token_type_ids
        )).map_err(|e| EmbeddingError::generation(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Get a session from the pool (round-robin)
        let (_idx, session_mutex) = self.session_pool.get_session();
        let mut session = session_mutex.lock()
            .map_err(|e| EmbeddingError::generation(format!("Failed to lock session: {}", e)))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
            "token_type_ids" => token_type_ids_value,
        ]).map_err(|e| EmbeddingError::generation(format!("ONNX inference failed: {}", e)))?;

        // Extract embeddings from output - try common output names
        let output_tensor = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("sentence_embedding"))
            .or_else(|| outputs.get("output"))
            .ok_or_else(|| EmbeddingError::generation("No output tensor found with expected names"))?;

        let (shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::generation(format!("Failed to extract output: {}", e)))?;

        // Reconstruct as ndarray for mean pooling
        let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        if dims.len() != 3 {
            return Err(EmbeddingError::generation(format!(
                "Expected 3D output tensor, got {} dimensions",
                dims.len()
            )));
        }

        let output_array = ndarray::Array3::from_shape_vec(
            (dims[0], dims[1], dims[2]),
            data.to_vec()
        ).map_err(|e| EmbeddingError::generation(format!("Failed to reshape output: {}", e)))?;

        // Mean pooling over sequence length
        let embedding = self.mean_pooling(&output_array.view(), &attention_mask_for_pooling)?;

        // Apply truncation if specified
        if let Some(truncate) = self.truncate_dim {
            Ok(embedding.into_iter().take(truncate).collect())
        } else {
            Ok(embedding)
        }
    }

    /// Mean pooling over sequence dimension
    fn mean_pooling(
        &self,
        hidden_state: &ndarray::ArrayView3<f32>,
        attention_mask: &[i64],
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_state.shape()[1];
        let hidden_size = hidden_state.shape()[2];

        let mut pooled = vec![0.0f32; hidden_size];
        let mut mask_sum = 0.0f32;

        for i in 0..seq_len {
            let mask_value = attention_mask[i] as f32;
            mask_sum += mask_value;

            for j in 0..hidden_size {
                pooled[j] += hidden_state[[0, i, j]] * mask_value;
            }
        }

        // Avoid division by zero
        if mask_sum > 0.0 {
            for value in &mut pooled {
                *value /= mask_sum;
            }
        }

        Ok(pooled)
    }

    /// Generate embedding from text (internal)
    fn generate_embedding_sync(&self, text: &str) -> Result<Embedding> {
        let (input_ids, attention_mask) = self.tokenize(text)?;
        let embedding_vec = self.run_inference(input_ids, attention_mask)?;

        let vector = Vector::new(embedding_vec)?;
        let metadata = EmbeddingMetadata::new(text, &self.model_name);

        Ok(Embedding::new(vector, metadata))
    }
}

#[async_trait]
impl EmbeddingGenerator for OnnxEmbeddingGenerator {
    async fn generate_from_text(&self, text: &str) -> Result<Embedding> {
        // Run in blocking thread pool to avoid blocking async runtime
        let text = text.to_string();
        let generator = self.clone_for_thread();

        tokio::task::spawn_blocking(move || generator.generate_embedding_sync(&text))
            .await
            .map_err(|e| EmbeddingError::generation(format!("Task join error: {}", e)))?
    }

    async fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        // Process in parallel using Rayon
        let generator = self.clone_for_thread();
        let texts = texts.to_vec();

        tokio::task::spawn_blocking(move || {
            texts
                .par_chunks(generator.max_batch_size)
                .flat_map(|chunk| {
                    chunk
                        .par_iter()
                        .map(|text| generator.generate_embedding_sync(text))
                        .collect::<Vec<_>>()
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| EmbeddingError::generation(format!("Task join error: {}", e)))?
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimensions(&self) -> usize {
        self.truncate_dim.unwrap_or(self.dimensions)
    }

    fn supports_batching(&self) -> bool {
        true
    }
}

impl OnnxEmbeddingGenerator {
    /// Clone for use in a thread
    ///
    /// This creates a lightweight clone that shares the session pool.
    /// The pool itself handles thread-safe access via round-robin selection.
    fn clone_for_thread(&self) -> Self {
        Self {
            session_pool: Arc::clone(&self.session_pool),
            tokenizer: Arc::clone(&self.tokenizer),
            model_name: self.model_name.clone(),
            model_path: self.model_path.clone(),
            dimensions: self.dimensions,
            truncate_dim: self.truncate_dim,
            max_batch_size: self.max_batch_size,
            pool_size: self.pool_size,
            mode: self.mode,
        }
    }

    /// Get the execution mode
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }
}
