//! WAL writer — append-only binary log with CRC32 checksums.
//!
//! Each entry is: [u64 seq_number][u64 timestamp_ns][u8 op_type][u32 data_len][data bytes][u32 crc32]
//!
//! The WAL is used for:
//! - Crash recovery (replay from last checkpoint)
//! - Change streams (tail the WAL for insert/update/delete events)
//! - Time-travel queries (replay to reconstruct state at any point)

use crate::types::{SparseVector, VectorId};
use crc32fast::Hasher as Crc32Hasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WalError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] bincode::Error),
    #[error("corrupt entry at sequence {0}: CRC mismatch")]
    CorruptEntry(u64),
}

/// WAL operation types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Upsert a document (dense vector, sparse vector, text fields, metadata).
    Upsert {
        id: VectorId,
        dense_vector: Option<Vec<f32>>,
        sparse_vector: Option<SparseVector>,
        text_fields: Option<std::collections::HashMap<String, String>>,
        payload: Option<Value>,
    },
    /// Delete a document.
    Delete { id: VectorId },
    /// Batch delete.
    BatchDelete { ids: Vec<VectorId> },
}

/// A single WAL entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub sequence_number: u64,
    pub timestamp_ns: u64,
    pub operation: WalOperation,
}

/// Append-only WAL writer.
pub struct WalWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    sequence: u64,
    /// Number of entries written since last fsync.
    pending_sync: usize,
    /// Fsync after this many entries.
    sync_interval: usize,
}

impl WalWriter {
    /// Open or create a WAL file.
    pub fn open(path: &Path, sync_interval: usize) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let writer = BufWriter::new(file);

        // Count existing entries to determine starting sequence number
        let sequence = if path.exists() && path.metadata()?.len() > 0 {
            count_entries(path)?
        } else {
            0
        };

        Ok(Self {
            writer,
            path: path.to_path_buf(),
            sequence,
            pending_sync: 0,
            sync_interval,
        })
    }

    /// Append an operation to the WAL. Returns the sequence number.
    pub fn append(&mut self, operation: WalOperation) -> Result<u64, WalError> {
        self.sequence += 1;
        let seq = self.sequence;

        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let entry = WalEntry {
            sequence_number: seq,
            timestamp_ns,
            operation,
        };

        let data = bincode::serialize(&entry)?;
        let data_len = data.len() as u32;

        // Compute CRC32 over the data
        let mut crc = Crc32Hasher::new();
        crc.update(&data);
        let checksum = crc.finalize();

        // Write: [data_len: u32][data: bytes][checksum: u32]
        self.writer.write_all(&data_len.to_le_bytes())?;
        self.writer.write_all(&data)?;
        self.writer.write_all(&checksum.to_le_bytes())?;

        self.pending_sync += 1;
        if self.pending_sync >= self.sync_interval {
            self.sync()?;
        }

        Ok(seq)
    }

    /// Force sync to disk.
    pub fn sync(&mut self) -> Result<(), WalError> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        self.pending_sync = 0;
        Ok(())
    }

    /// Get the current sequence number.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Get the WAL file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Count entries in an existing WAL file (for resuming sequence numbers).
fn count_entries(path: &Path) -> Result<u64, WalError> {
    use std::io::Read;
    let mut file = File::open(path)?;
    let mut count = 0u64;

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(WalError::Io(e)),
        }
        let data_len = u32::from_le_bytes(len_buf) as usize;

        // Skip data + checksum
        let mut skip = vec![0u8; data_len + 4];
        file.read_exact(&mut skip)?;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_entries() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");
        let mut writer = WalWriter::open(&wal_path, 10).unwrap();

        let seq1 = writer
            .append(WalOperation::Upsert {
                id: "doc1".to_string(),
                dense_vector: Some(vec![1.0, 2.0, 3.0]),
                sparse_vector: None,
                text_fields: None,
                payload: None,
            })
            .unwrap();
        assert_eq!(seq1, 1);

        let seq2 = writer
            .append(WalOperation::Delete {
                id: "doc1".to_string(),
            })
            .unwrap();
        assert_eq!(seq2, 2);

        writer.sync().unwrap();
        assert_eq!(writer.sequence(), 2);
    }

    #[test]
    fn test_resume_sequence() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut writer = WalWriter::open(&wal_path, 10).unwrap();
            writer
                .append(WalOperation::Delete {
                    id: "x".to_string(),
                })
                .unwrap();
            writer
                .append(WalOperation::Delete {
                    id: "y".to_string(),
                })
                .unwrap();
            writer.sync().unwrap();
        }

        // Reopen — should resume at sequence 2
        let writer = WalWriter::open(&wal_path, 10).unwrap();
        assert_eq!(writer.sequence(), 2);
    }
}
