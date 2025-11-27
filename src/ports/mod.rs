//! Ports module - Interfaces and abstractions
//!
//! This module defines the traits (ports) that must be implemented
//! by the infrastructure layer.

pub mod generator;
pub mod repository;
pub mod vector_store;

pub use generator::{EmbeddingGenerator, GeneratorConfig, ModelBackend};
pub use repository::EmbeddingRepository;
pub use vector_store::{VectorStore, SearchParams, SimilarityResult, DistanceMetric};
