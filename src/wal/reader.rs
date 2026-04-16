//! WAL reader — replay and tailing support.
//!
//! Reads WAL entries sequentially with CRC32 verification.

use super::writer::{WalEntry, WalError};
use crc32fast::Hasher as Crc32Hasher;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// WAL reader for replay and tailing.
pub struct WalReader {
    file: File,
    entries_read: u64,
}

impl WalReader {
    /// Open a WAL file for reading.
    pub fn open(path: &Path) -> Result<Self, WalError> {
        let file = File::open(path)?;
        Ok(Self {
            file,
            entries_read: 0,
        })
    }

    /// Read the next entry. Returns None at EOF.
    pub fn next_entry(&mut self) -> Result<Option<WalEntry>, WalError> {
        // Read data length
        let mut len_buf = [0u8; 4];
        match self.file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(WalError::Io(e)),
        }
        let data_len = u32::from_le_bytes(len_buf) as usize;

        // Read data
        let mut data = vec![0u8; data_len];
        self.file.read_exact(&mut data)?;

        // Read checksum
        let mut crc_buf = [0u8; 4];
        self.file.read_exact(&mut crc_buf)?;
        let stored_crc = u32::from_le_bytes(crc_buf);

        // Verify CRC
        let mut crc = Crc32Hasher::new();
        crc.update(&data);
        let computed_crc = crc.finalize();

        if stored_crc != computed_crc {
            return Err(WalError::CorruptEntry(self.entries_read + 1));
        }

        let entry: WalEntry = bincode::deserialize(&data)?;
        self.entries_read += 1;
        Ok(Some(entry))
    }

    /// Read all remaining entries.
    pub fn read_all(&mut self) -> Result<Vec<WalEntry>, WalError> {
        let mut entries = Vec::new();
        while let Some(entry) = self.next_entry()? {
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Read entries starting from a given sequence number.
    pub fn read_from(&mut self, from_sequence: u64) -> Result<Vec<WalEntry>, WalError> {
        let mut entries = Vec::new();
        while let Some(entry) = self.next_entry()? {
            if entry.sequence_number >= from_sequence {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// Number of entries read so far.
    pub fn entries_read(&self) -> u64 {
        self.entries_read
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::writer::{WalOperation, WalWriter};
    use tempfile::TempDir;

    #[test]
    fn test_write_and_read_back() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        // Write entries
        {
            let mut writer = WalWriter::open(&path, 10).unwrap();
            writer
                .append(WalOperation::Upsert {
                    id: "a".to_string(),
                    dense_vector: Some(vec![1.0, 2.0]),
                    sparse_vector: None,
                    text_fields: None,
                    payload: None,
                })
                .unwrap();
            writer
                .append(WalOperation::Delete {
                    id: "b".to_string(),
                })
                .unwrap();
            writer.sync().unwrap();
        }

        // Read back
        let mut reader = WalReader::open(&path).unwrap();
        let entries = reader.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sequence_number, 1);
        assert_eq!(entries[1].sequence_number, 2);
    }

    #[test]
    fn test_read_from_sequence() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut writer = WalWriter::open(&path, 10).unwrap();
            for i in 0..5 {
                writer
                    .append(WalOperation::Delete {
                        id: format!("doc{i}"),
                    })
                    .unwrap();
            }
            writer.sync().unwrap();
        }

        let mut reader = WalReader::open(&path).unwrap();
        let entries = reader.read_from(3).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].sequence_number, 3);
    }

    #[test]
    fn test_empty_wal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.wal");
        std::fs::File::create(&path).unwrap();

        let mut reader = WalReader::open(&path).unwrap();
        let entries = reader.read_all().unwrap();
        assert!(entries.is_empty());
    }
}
