//! Configuration module
//!
//! Handles application configuration using environment variables
//! and configuration files.
//!
//! ## Memory Allocation Patterns
//!
//! This module demonstrates different memory allocation patterns in Rust:
//!
//! ### Static Memory
//! - Constants like `DEFAULT_EMBEDDING_DIMENSIONS` live in the binary's data segment
//! - They have a 'static lifetime and are known at compile time
//! - No runtime allocation overhead
//! - Shared across all threads without synchronization overhead
//!
//! ### Heap Memory
//! - Configuration structs (AppConfig, StorageConfig, etc.) are typically heap-allocated
//! - Strings within configs use heap allocation (String type)
//! - Dynamic sizing allows runtime configuration changes
//!
//! ### Stack Memory
//! - Function parameters and local variables use stack allocation
//! - Fast allocation/deallocation (just moving the stack pointer)
//! - Fixed size known at compile time

use serde::{Deserialize, Serialize};
use crate::error::{EmbeddingError, Result};

// ============================================================================
// STATIC MEMORY: Constants stored in binary data segment
// ============================================================================
// These constants demonstrate STATIC MEMORY allocation:
// - Stored in the program's data segment (not stack, not heap)
// - Lifetime: 'static (available for entire program duration)
// - Zero runtime allocation cost
// - Can be used across multiple threads without synchronization

/// Default embedding dimensions (Static Memory)
/// This value is baked into the binary at compile time
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;

/// Maximum allowed embedding dimensions (Static Memory)
pub const MAX_EMBEDDING_DIMENSIONS: usize = 4096;

/// Minimum embedding dimensions (Static Memory)
pub const MIN_EMBEDDING_DIMENSIONS: usize = 1;

/// Default batch size for processing (Static Memory)
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Maximum batch size to prevent memory exhaustion (Static Memory)
pub const MAX_BATCH_SIZE: usize = 1000;

/// Default cache size in memory repository (Static Memory)
pub const DEFAULT_CACHE_SIZE: usize = 1000;

/// Application version (Static Memory - string slice)
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Valid log levels (Static Memory - array of string slices)
pub const VALID_LOG_LEVELS: &[&str] = &["trace", "debug", "info", "warn", "error"];

/// Default storage type (Static Memory)
pub const DEFAULT_STORAGE_TYPE: &str = "memory";

/// Default distance metric (Static Memory)
pub const DEFAULT_DISTANCE_METRIC: &str = "cosine";

// ============================================================================
// HEAP MEMORY: Dynamic configuration structures
// ============================================================================
// These structs demonstrate HEAP MEMORY allocation:
// - String fields allocate on heap (growable)
// - Option<T> may allocate on heap depending on T
// - Cloning these structs involves heap allocations

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Storage configuration
    pub storage: StorageConfig,

    /// Generator configuration
    pub generator: GeneratorSettings,

    /// Vector store configuration
    pub vector_store: VectorStoreConfig,

    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage type (memory, file, database)
    pub storage_type: String,

    /// Connection string or file path
    pub connection: Option<String>,

    /// Maximum cache size
    pub max_cache_size: Option<usize>,
}

/// Generator settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorSettings {
    /// Default model to use
    pub default_model: String,

    /// Default dimensions
    pub default_dimensions: usize,

    /// API key (if required)
    pub api_key: Option<String>,

    /// Batch size for batch processing
    pub batch_size: usize,
}

/// Vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Vector store type (memory, faiss, pinecone)
    pub store_type: String,

    /// Distance metric to use
    pub distance_metric: String,

    /// Index type (if applicable)
    pub index_type: Option<String>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,

    /// Enable JSON formatting
    pub json_format: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                storage_type: DEFAULT_STORAGE_TYPE.to_string(),
                connection: None,
                max_cache_size: Some(DEFAULT_CACHE_SIZE),
            },
            generator: GeneratorSettings {
                default_model: "default".to_string(),
                default_dimensions: DEFAULT_EMBEDDING_DIMENSIONS,
                api_key: None,
                batch_size: DEFAULT_BATCH_SIZE,
            },
            vector_store: VectorStoreConfig {
                store_type: DEFAULT_STORAGE_TYPE.to_string(),
                distance_metric: DEFAULT_DISTANCE_METRIC.to_string(),
                index_type: None,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                json_format: false,
            },
        }
    }
}

impl AppConfig {
    /// Load configuration from a file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| EmbeddingError::config(format!("Failed to read config file: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| EmbeddingError::config(format!("Failed to parse config: {}", e)))
    }

    /// Save configuration to a file
    pub fn to_file(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| EmbeddingError::config(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| EmbeddingError::config(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        if let Ok(storage_type) = std::env::var("EMBEDDINGS_STORAGE_TYPE") {
            config.storage.storage_type = storage_type;
        }

        if let Ok(connection) = std::env::var("EMBEDDINGS_STORAGE_CONNECTION") {
            config.storage.connection = Some(connection);
        }

        if let Ok(model) = std::env::var("EMBEDDINGS_DEFAULT_MODEL") {
            config.generator.default_model = model;
        }

        if let Ok(api_key) = std::env::var("EMBEDDINGS_API_KEY") {
            config.generator.api_key = Some(api_key);
        }

        if let Ok(dimensions) = std::env::var("EMBEDDINGS_DIMENSIONS") {
            config.generator.default_dimensions = dimensions
                .parse()
                .map_err(|e| EmbeddingError::config(format!("Invalid dimensions: {}", e)))?;
        }

        if let Ok(log_level) = std::env::var("EMBEDDINGS_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate dimensions using static constants
        if self.generator.default_dimensions < MIN_EMBEDDING_DIMENSIONS {
            return Err(EmbeddingError::config(format!(
                "Dimensions must be at least {}",
                MIN_EMBEDDING_DIMENSIONS
            )));
        }

        if self.generator.default_dimensions > MAX_EMBEDDING_DIMENSIONS {
            return Err(EmbeddingError::config(format!(
                "Dimensions cannot exceed {}",
                MAX_EMBEDDING_DIMENSIONS
            )));
        }

        // Validate batch size using static constants
        if self.generator.batch_size == 0 {
            return Err(EmbeddingError::config("Batch size must be greater than 0"));
        }

        if self.generator.batch_size > MAX_BATCH_SIZE {
            return Err(EmbeddingError::config(format!(
                "Batch size cannot exceed {}",
                MAX_BATCH_SIZE
            )));
        }

        // Validate log level using static constant
        if !VALID_LOG_LEVELS.contains(&self.logging.level.as_str()) {
            return Err(EmbeddingError::config(format!(
                "Invalid log level: {}. Must be one of: {}",
                self.logging.level,
                VALID_LOG_LEVELS.join(", ")
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.storage.storage_type, "memory");
        assert_eq!(config.generator.default_dimensions, 384);
    }

    #[test]
    fn test_config_validation() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = AppConfig::default();
        invalid_config.generator.default_dimensions = 0;
        assert!(invalid_config.validate().is_err());
    }
}
