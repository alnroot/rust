//! Log Entry Domain Model
//!
//! Domain model for log entries with temporal analysis capabilities.

use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Log severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Fatal => write!(f, "FATAL"),
        }
    }
}

impl LogLevel {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "ERROR" | "ERR" => Some(LogLevel::Error),
            "FATAL" | "CRITICAL" => Some(LogLevel::Fatal),
            _ => None,
        }
    }

    /// Get numeric severity (higher = more severe)
    pub fn severity(&self) -> u8 {
        match self {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
            LogLevel::Fatal => 5,
        }
    }
}

/// Log entry with temporal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp of the log
    pub timestamp: DateTime<Utc>,

    /// Log level
    pub level: LogLevel,

    /// Log message content
    pub message: String,

    /// Service/component that generated the log
    pub source: String,

    /// Optional trace ID for distributed tracing
    pub trace_id: Option<String>,

    /// Optional span ID
    pub span_id: Option<String>,

    /// Additional structured fields
    pub fields: HashMap<String, String>,

    /// Request duration (if applicable)
    pub duration_ms: Option<f64>,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(
        timestamp: DateTime<Utc>,
        level: LogLevel,
        message: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            timestamp,
            level,
            message: message.into(),
            source: source.into(),
            trace_id: None,
            span_id: None,
            fields: HashMap::new(),
            duration_ms: None,
        }
    }

    /// Builder pattern
    pub fn builder() -> LogEntryBuilder {
        LogEntryBuilder::default()
    }

    /// Add a custom field
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Set trace ID
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration_ms: f64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Get searchable text representation
    pub fn to_searchable_text(&self) -> String {
        format!(
            "{} [{}] {} - {} {}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S"),
            self.level,
            self.source,
            self.message,
            self.fields.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    /// Check if log is within time window
    pub fn is_within(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> bool {
        self.timestamp >= start && self.timestamp <= end
    }

    /// Check if log is an error or worse
    pub fn is_error(&self) -> bool {
        self.level.severity() >= LogLevel::Error.severity()
    }

    /// Check if log has slow duration (if duration tracking enabled)
    pub fn is_slow(&self, threshold_ms: f64) -> bool {
        self.duration_ms.map(|d| d > threshold_ms).unwrap_or(false)
    }
}

/// Builder for LogEntry
#[derive(Default)]
pub struct LogEntryBuilder {
    timestamp: Option<DateTime<Utc>>,
    level: Option<LogLevel>,
    message: Option<String>,
    source: Option<String>,
    trace_id: Option<String>,
    span_id: Option<String>,
    fields: HashMap<String, String>,
    duration_ms: Option<f64>,
}

impl LogEntryBuilder {
    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    pub fn now(mut self) -> Self {
        self.timestamp = Some(Utc::now());
        self
    }

    pub fn level(mut self, level: LogLevel) -> Self {
        self.level = Some(level);
        self
    }

    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    pub fn span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    pub fn duration_ms(mut self, duration: f64) -> Self {
        self.duration_ms = Some(duration);
        self
    }

    pub fn build(self) -> LogEntry {
        LogEntry {
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            level: self.level.unwrap_or(LogLevel::Info),
            message: self.message.unwrap_or_default(),
            source: self.source.unwrap_or_else(|| "unknown".to_string()),
            trace_id: self.trace_id,
            span_id: self.span_id,
            fields: self.fields,
            duration_ms: self.duration_ms,
        }
    }
}

/// Time window for log queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeWindow {
    /// Create a time window
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    /// Last N minutes
    pub fn last_minutes(minutes: i64) -> Self {
        let end = Utc::now();
        let start = end - Duration::minutes(minutes);
        Self { start, end }
    }

    /// Last N hours
    pub fn last_hours(hours: i64) -> Self {
        let end = Utc::now();
        let start = end - Duration::hours(hours);
        Self { start, end }
    }

    /// Last N days
    pub fn last_days(days: i64) -> Self {
        let end = Utc::now();
        let start = end - Duration::days(days);
        Self { start, end }
    }

    /// Duration of the window
    pub fn duration(&self) -> Duration {
        self.end - self.start
    }

    /// Check if timestamp is within window
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_builder() {
        let entry = LogEntry::builder()
            .now()
            .level(LogLevel::Error)
            .message("Test error")
            .source("test-service")
            .field("user_id", "123")
            .duration_ms(150.5)
            .build();

        assert_eq!(entry.level, LogLevel::Error);
        assert_eq!(entry.message, "Test error");
        assert_eq!(entry.source, "test-service");
        assert_eq!(entry.duration_ms, Some(150.5));
        assert!(entry.is_error());
        assert!(entry.is_slow(100.0));
    }

    #[test]
    fn test_time_window() {
        // Use a fixed reference time to avoid race conditions
        let reference_time = Utc::now();
        let window = TimeWindow::new(
            reference_time - Duration::hours(1),
            reference_time,
        );

        assert!(window.contains(reference_time));
        assert!(window.contains(reference_time - Duration::minutes(30)));
        assert!(!window.contains(reference_time - Duration::hours(2)));

        // Also test the convenience constructors
        let recent_window = TimeWindow::last_hours(1);
        assert!(recent_window.duration() == Duration::hours(1));
    }

    #[test]
    fn test_log_level_severity() {
        assert!(LogLevel::Error.severity() > LogLevel::Info.severity());
        assert!(LogLevel::Fatal.severity() > LogLevel::Error.severity());
    }
}
