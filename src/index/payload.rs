//! RocksDB-backed payload storage for per-document structured metadata.
//!
//! Stores schemaless JSON payloads keyed by VectorId. Supports basic
//! filter operations on payload fields.

use crate::types::VectorId;
use rocksdb::{Options, DB};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PayloadError {
    #[error("rocksdb error: {0}")]
    RocksDb(#[from] rocksdb::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("not found: {0}")]
    NotFound(String),
}

/// Filter condition for payload queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    /// field == value
    Eq(String, Value),
    /// field != value
    Ne(String, Value),
    /// field > value (numeric)
    Gt(String, f64),
    /// field >= value (numeric)
    Gte(String, f64),
    /// field < value (numeric)
    Lt(String, f64),
    /// field <= value (numeric)
    Lte(String, f64),
    /// field IN [values]
    In(String, Vec<Value>),
    /// field contains substring (string)
    Contains(String, String),
    /// All conditions must match
    And(Vec<FilterCondition>),
    /// Any condition must match
    Or(Vec<FilterCondition>),
    /// Negate condition
    Not(Box<FilterCondition>),
}

/// RocksDB payload store.
pub struct PayloadStore {
    db: DB,
}

impl PayloadStore {
    /// Open (or create) a payload store at the given path.
    pub fn open(path: &Path) -> Result<Self, PayloadError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let db = DB::open(&opts, path)?;
        Ok(Self { db })
    }

    /// Store a payload for a document.
    pub fn put(&self, id: &VectorId, payload: &Value) -> Result<(), PayloadError> {
        let data = serde_json::to_vec(payload)?;
        self.db.put(id.as_bytes(), &data)?;
        Ok(())
    }

    /// Get a payload by document ID.
    pub fn get(&self, id: &VectorId) -> Result<Option<Value>, PayloadError> {
        match self.db.get(id.as_bytes())? {
            Some(data) => {
                let value: Value = serde_json::from_slice(&data)?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Delete a payload by document ID.
    pub fn delete(&self, id: &VectorId) -> Result<(), PayloadError> {
        self.db.delete(id.as_bytes())?;
        Ok(())
    }

    /// Check if a payload matches a filter condition.
    pub fn matches_filter(payload: &Value, filter: &FilterCondition) -> bool {
        match filter {
            FilterCondition::Eq(field, expected) => {
                payload.get(field).map_or(false, |v| v == expected)
            }
            FilterCondition::Ne(field, expected) => {
                payload.get(field).map_or(true, |v| v != expected)
            }
            FilterCondition::Gt(field, threshold) => payload
                .get(field)
                .and_then(|v| v.as_f64())
                .map_or(false, |v| v > *threshold),
            FilterCondition::Gte(field, threshold) => payload
                .get(field)
                .and_then(|v| v.as_f64())
                .map_or(false, |v| v >= *threshold),
            FilterCondition::Lt(field, threshold) => payload
                .get(field)
                .and_then(|v| v.as_f64())
                .map_or(false, |v| v < *threshold),
            FilterCondition::Lte(field, threshold) => payload
                .get(field)
                .and_then(|v| v.as_f64())
                .map_or(false, |v| v <= *threshold),
            FilterCondition::In(field, values) => payload
                .get(field)
                .map_or(false, |v| values.contains(v)),
            FilterCondition::Contains(field, substring) => payload
                .get(field)
                .and_then(|v| v.as_str())
                .map_or(false, |s| s.contains(substring.as_str())),
            FilterCondition::And(conditions) => {
                conditions.iter().all(|c| Self::matches_filter(payload, c))
            }
            FilterCondition::Or(conditions) => {
                conditions.iter().any(|c| Self::matches_filter(payload, c))
            }
            FilterCondition::Not(condition) => !Self::matches_filter(payload, condition),
        }
    }

    /// Filter a set of IDs by payload conditions. Returns IDs that pass the filter.
    pub fn filter_ids(
        &self,
        ids: &[VectorId],
        filter: &FilterCondition,
    ) -> Result<Vec<VectorId>, PayloadError> {
        let mut result = Vec::new();
        for id in ids {
            if let Some(payload) = self.get(id)? {
                if Self::matches_filter(&payload, filter) {
                    result.push(id.clone());
                }
            }
        }
        Ok(result)
    }

    /// Count all documents.
    pub fn count(&self) -> usize {
        self.db.iterator(rocksdb::IteratorMode::Start).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn open_test_store() -> (PayloadStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = PayloadStore::open(dir.path()).unwrap();
        (store, dir)
    }

    #[test]
    fn test_put_get_delete() {
        let (store, _dir) = open_test_store();

        let payload = json!({"name": "test", "score": 42, "tags": ["a", "b"]});
        store.put(&"doc1".to_string(), &payload).unwrap();

        let retrieved = store.get(&"doc1".to_string()).unwrap().unwrap();
        assert_eq!(retrieved, payload);

        store.delete(&"doc1".to_string()).unwrap();
        assert!(store.get(&"doc1".to_string()).unwrap().is_none());
    }

    #[test]
    fn test_filter_eq() {
        let payload = json!({"color": "red", "size": 10});
        assert!(PayloadStore::matches_filter(
            &payload,
            &FilterCondition::Eq("color".to_string(), json!("red"))
        ));
        assert!(!PayloadStore::matches_filter(
            &payload,
            &FilterCondition::Eq("color".to_string(), json!("blue"))
        ));
    }

    #[test]
    fn test_filter_numeric() {
        let payload = json!({"price": 25.5});
        assert!(PayloadStore::matches_filter(
            &payload,
            &FilterCondition::Gt("price".to_string(), 20.0)
        ));
        assert!(!PayloadStore::matches_filter(
            &payload,
            &FilterCondition::Gt("price".to_string(), 30.0)
        ));
        assert!(PayloadStore::matches_filter(
            &payload,
            &FilterCondition::Lte("price".to_string(), 25.5)
        ));
    }

    #[test]
    fn test_filter_and_or() {
        let payload = json!({"color": "red", "size": 10});
        let filter = FilterCondition::And(vec![
            FilterCondition::Eq("color".to_string(), json!("red")),
            FilterCondition::Gte("size".to_string(), 5.0),
        ]);
        assert!(PayloadStore::matches_filter(&payload, &filter));

        let filter = FilterCondition::Or(vec![
            FilterCondition::Eq("color".to_string(), json!("blue")),
            FilterCondition::Eq("color".to_string(), json!("red")),
        ]);
        assert!(PayloadStore::matches_filter(&payload, &filter));
    }

    #[test]
    fn test_filter_not() {
        let payload = json!({"active": true});
        let filter = FilterCondition::Not(Box::new(FilterCondition::Eq(
            "active".to_string(),
            json!(false),
        )));
        assert!(PayloadStore::matches_filter(&payload, &filter));
    }

    #[test]
    fn test_filter_contains() {
        let payload = json!({"description": "the quick brown fox"});
        assert!(PayloadStore::matches_filter(
            &payload,
            &FilterCondition::Contains("description".to_string(), "brown".to_string())
        ));
    }

    #[test]
    fn test_filter_in() {
        let payload = json!({"status": "active"});
        assert!(PayloadStore::matches_filter(
            &payload,
            &FilterCondition::In(
                "status".to_string(),
                vec![json!("active"), json!("pending")]
            )
        ));
    }

    #[test]
    fn test_filter_ids() {
        let (store, _dir) = open_test_store();

        store
            .put(&"d1".to_string(), &json!({"color": "red"}))
            .unwrap();
        store
            .put(&"d2".to_string(), &json!({"color": "blue"}))
            .unwrap();
        store
            .put(&"d3".to_string(), &json!({"color": "red"}))
            .unwrap();

        let ids: Vec<VectorId> = vec!["d1".into(), "d2".into(), "d3".into()];
        let filter = FilterCondition::Eq("color".to_string(), json!("red"));
        let filtered = store.filter_ids(&ids, &filter).unwrap();
        assert_eq!(filtered, vec!["d1".to_string(), "d3".to_string()]);
    }
}
