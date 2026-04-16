//! BM25 full-text search powered by Tantivy.
//!
//! Each shard maintains a Tantivy index with configurable text fields.
//! Documents are identified by their VectorId. BM25 search returns scored
//! results that can be fused with dense/sparse results.

use crate::types::{ScoredId, VectorId};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{self, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};
use thiserror::Error;

use std::path::Path;

#[derive(Debug, Error)]
pub enum BM25Error {
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),
    #[error("field not found: {0}")]
    FieldNotFound(String),
}

/// Configuration for the BM25 index.
#[derive(Debug, Clone)]
pub struct BM25Config {
    /// Text field names to index (e.g., ["title", "description", "transcript"]).
    pub text_fields: Vec<String>,
    /// Heap size for the index writer (bytes). Default: 50 MB.
    pub writer_heap_size: usize,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            text_fields: vec!["text".to_string()],
            writer_heap_size: 50_000_000,
        }
    }
}

/// BM25 full-text search index wrapping Tantivy.
pub struct BM25Index {
    index: Index,
    writer: IndexWriter,
    #[allow(dead_code)]
    schema: Schema,
    id_field: schema::Field,
    text_fields: Vec<(String, schema::Field)>,
}

impl BM25Index {
    /// Create a new BM25 index in a directory.
    pub fn new(path: &Path, config: &BM25Config) -> Result<Self, BM25Error> {
        let mut schema_builder = Schema::builder();

        // Always have an ID field (stored + indexed for deletion)
        let id_field = schema_builder.add_text_field("_id", STRING | STORED);

        // Add configurable text fields
        let text_fields: Vec<(String, schema::Field)> = config
            .text_fields
            .iter()
            .map(|name| {
                let field = schema_builder.add_text_field(name, TEXT | STORED);
                (name.clone(), field)
            })
            .collect();

        let schema = schema_builder.build();
        let index = Index::create_in_dir(path, schema.clone())?;
        let writer = index.writer(config.writer_heap_size)?;

        Ok(Self {
            index,
            writer,
            schema,
            id_field,
            text_fields,
        })
    }

    /// Create an in-memory BM25 index (for testing).
    pub fn new_in_memory(config: &BM25Config) -> Result<Self, BM25Error> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("_id", STRING | STORED);
        let text_fields: Vec<(String, schema::Field)> = config
            .text_fields
            .iter()
            .map(|name| {
                let field = schema_builder.add_text_field(name, TEXT | STORED);
                (name.clone(), field)
            })
            .collect();

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let writer = index.writer(config.writer_heap_size)?;

        Ok(Self {
            index,
            writer,
            schema,
            id_field,
            text_fields,
        })
    }

    /// Insert a document into the BM25 index.
    ///
    /// `fields` is a map of field_name → text content.
    pub fn insert(
        &mut self,
        id: &VectorId,
        fields: &std::collections::HashMap<String, String>,
    ) -> Result<(), BM25Error> {
        let mut doc = doc!( self.id_field => id.as_str() );

        for (name, field) in &self.text_fields {
            if let Some(text) = fields.get(name) {
                doc.add_text(*field, text);
            }
        }

        self.writer.add_document(doc)?;
        Ok(())
    }

    /// Delete all documents with the given ID.
    pub fn delete(&mut self, id: &VectorId) -> Result<(), BM25Error> {
        let term = tantivy::Term::from_field_text(self.id_field, id.as_str());
        self.writer.delete_term(term);
        Ok(())
    }

    /// Commit pending writes to the index.
    pub fn commit(&mut self) -> Result<(), BM25Error> {
        self.writer.commit()?;
        Ok(())
    }

    /// Search the BM25 index.
    ///
    /// `query_str` is a Tantivy query string (supports AND, OR, NOT, phrases).
    /// Returns top-k results scored by BM25.
    pub fn search(&self, query_str: &str, k: usize) -> Result<Vec<ScoredId>, BM25Error> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        reader.reload()?;
        let searcher = reader.searcher();

        // Build query parser over all text fields
        let field_refs: Vec<schema::Field> = self.text_fields.iter().map(|(_, f)| *f).collect();
        let query_parser = QueryParser::for_index(&self.index, field_refs);
        let query = query_parser.parse_query(query_str)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(k))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let retrieved_doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;
            if let Some(id_value) = retrieved_doc.get_first(self.id_field) {
                if let Some(id_text) = id_value.as_str() {
                    results.push(ScoredId {
                        id: id_text.to_string(),
                        score,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Get the number of documents in the index.
    pub fn num_docs(&self) -> Result<u64, BM25Error> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(reader.searcher().num_docs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn default_config() -> BM25Config {
        BM25Config {
            text_fields: vec!["title".to_string(), "body".to_string()],
            writer_heap_size: 15_000_000,
        }
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = BM25Index::new_in_memory(&default_config()).unwrap();

        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Rust programming language".to_string());
        fields.insert(
            "body".to_string(),
            "Rust is a systems programming language focused on safety".to_string(),
        );
        index.insert(&"doc1".to_string(), &fields).unwrap();

        fields.clear();
        fields.insert("title".to_string(), "Python programming".to_string());
        fields.insert(
            "body".to_string(),
            "Python is a high-level interpreted language".to_string(),
        );
        index.insert(&"doc2".to_string(), &fields).unwrap();

        index.commit().unwrap();

        let results = index.search("rust safety", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_delete() {
        let mut index = BM25Index::new_in_memory(&default_config()).unwrap();

        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "test document".to_string());
        index.insert(&"doc1".to_string(), &fields).unwrap();
        index.commit().unwrap();

        index.delete(&"doc1".to_string()).unwrap();
        index.commit().unwrap();

        let results = index.search("test", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_search() {
        let index = BM25Index::new_in_memory(&default_config()).unwrap();
        let results = index.search("nonexistent", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_phrase_query() {
        let mut index = BM25Index::new_in_memory(&default_config()).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "body".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        );
        index.insert(&"doc1".to_string(), &fields).unwrap();
        index.commit().unwrap();

        let results = index.search("\"quick brown fox\"", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }
}
