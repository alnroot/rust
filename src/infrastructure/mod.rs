//! Infrastructure layer - Concrete implementations
//!
//! Contains implementations of ports (interfaces).

pub mod model_manager;
pub mod onnx_generator;
pub mod http_generator;
pub mod in_memory_repository;
pub mod in_memory_vector_store;

// Production features
pub mod sqlite_repository;
pub mod hnsw_vector_store;

pub use model_manager::{ModelManager, ModelInfo};
pub use onnx_generator::{OnnxEmbeddingGenerator, ExecutionMode};
pub use http_generator::HttpEmbeddingGenerator;
pub use in_memory_repository::InMemoryRepository;
pub use in_memory_vector_store::InMemoryVectorStore;
pub use sqlite_repository::SqliteRepository;
pub use hnsw_vector_store::{HnswVectorStore, HnswConfig};
