//! Application layer - Use cases and services

pub mod service;
pub mod use_cases;
pub mod log_analyzer;

pub mod services {
    pub use super::service::*;
}

pub use service::EmbeddingService;
pub use use_cases::*;
pub use log_analyzer::{LogAnalyzer, LogSearchResult, LogStatistics};
