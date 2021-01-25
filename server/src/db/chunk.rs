use arrow_deps::{
    arrow::{datatypes::SchemaRef, error::Result as ArrowResult, record_batch::RecordBatch},
    datafusion::{
        logical_plan::LogicalPlan,
        physical_plan::{RecordBatchStream, SendableRecordBatchStream},
    },
    util::str_iter_to_batch,
};
use data_types::{schema::Schema, selection::Selection};
use query::{predicate::Predicate, util::make_scan_plan, PartitionChunk};
use read_buffer::{Database as ReadBufferDb, ReadFilterResults};
use snafu::{ResultExt, Snafu};

use std::{
    sync::{Arc, RwLock},
    task::{Context, Poll},
};

use super::pred::to_read_buffer_predicate;

use async_trait::async_trait;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Mutable Buffer Chunk Error: {}", source))]
    MutableBufferChunk {
        source: mutable_buffer::chunk::Error,
    },

    #[snafu(display("Read Buffer Error in chunk {}: {}", chunk_id, source))]
    ReadBufferChunk {
        source: read_buffer::Error,
        chunk_id: u32,
    },

    #[snafu(display("Internal Predicate Conversion Error: {}", source))]
    InternalPredicateConversion { source: super::pred::Error },

    #[snafu(display("internal error creating plan: {}", source))]
    InternalPlanCreation {
        source: arrow_deps::datafusion::error::DataFusionError,
    },

    #[snafu(display("arrow conversion error: {}", source))]
    ArrowConversion {
        source: arrow_deps::arrow::error::ArrowError,
    },
}
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A IOx DatabaseChunk can come from one of three places:
/// MutableBuffer, ReadBuffer, or a ParquetFile
#[derive(Debug)]
pub enum DBChunk {
    MutableBuffer {
        chunk: Arc<mutable_buffer::chunk::Chunk>,
    },
    ReadBuffer {
        db: Arc<RwLock<ReadBufferDb>>,
        partition_key: String,
        chunk_id: u32,
    },
    ParquetFile, // TODO add appropriate type here
}

impl DBChunk {
    /// Create a new mutable buffer chunk
    pub fn new_mb(chunk: Arc<mutable_buffer::chunk::Chunk>) -> Arc<Self> {
        Arc::new(Self::MutableBuffer { chunk })
    }

    /// create a new read buffer chunk
    pub fn new_rb(
        db: Arc<RwLock<ReadBufferDb>>,
        partition_key: impl Into<String>,
        chunk_id: u32,
    ) -> Arc<Self> {
        let partition_key = partition_key.into();
        Arc::new(Self::ReadBuffer {
            db,
            chunk_id,
            partition_key,
        })
    }
}

#[async_trait]
impl PartitionChunk for DBChunk {
    type Error = Error;

    fn id(&self) -> u32 {
        match self {
            Self::MutableBuffer { chunk } => chunk.id(),
            Self::ReadBuffer { chunk_id, .. } => *chunk_id,
            Self::ParquetFile => unimplemented!("parquet file not implemented"),
        }
    }

    fn table_stats(&self) -> Result<Vec<data_types::partition_metadata::Table>, Self::Error> {
        match self {
            Self::MutableBuffer { chunk } => chunk.table_stats().context(MutableBufferChunk),
            Self::ReadBuffer { .. } => unimplemented!("read buffer not implemented"),
            Self::ParquetFile => unimplemented!("parquet file not implemented"),
        }
    }

    async fn table_names(&self, predicate: &Predicate) -> Result<LogicalPlan, Self::Error> {
        match self {
            Self::MutableBuffer { chunk } => {
                let names: Vec<Option<&str>> = if chunk.is_empty() {
                    Vec::new()
                } else {
                    let chunk_predicate = chunk
                        .compile_predicate(predicate)
                        .context(MutableBufferChunk)?;

                    chunk
                        .table_names(&chunk_predicate)
                        .context(MutableBufferChunk)?
                        .into_iter()
                        .map(Some)
                        .collect()
                };

                let batch = str_iter_to_batch("tables", names).context(ArrowConversion)?;

                make_scan_plan(batch).context(InternalPlanCreation)
            }
            Self::ReadBuffer {
                db,
                partition_key,
                chunk_id,
            } => {
                let chunk_id = *chunk_id;

                let rb_predicate =
                    to_read_buffer_predicate(&predicate).context(InternalPredicateConversion)?;

                let db = db.read().unwrap();
                let batch = db
                    .table_names(partition_key, &[chunk_id], rb_predicate)
                    .context(ReadBufferChunk { chunk_id })?;
                make_scan_plan(batch).context(InternalPlanCreation)
            }
            Self::ParquetFile => {
                unimplemented!("parquet file not implemented")
            }
        }
    }

    async fn table_schema(
        &self,
        table_name: &str,
        selection: Selection<'_>,
    ) -> Result<Schema, Self::Error> {
        match self {
            DBChunk::MutableBuffer { chunk } => chunk
                .table_schema(table_name, selection)
                .context(MutableBufferChunk),
            DBChunk::ReadBuffer {
                db,
                partition_key,
                chunk_id,
            } => {
                let chunk_id = *chunk_id;
                let db = db.read().unwrap();

                // TODO: Andrew -- I think technically this reordering
                // should be happening inside the read buffer, but
                // we'll see when we get to read_filter as the same
                // issue will appear when actually reading columns
                // back
                let needs_sort = matches!(selection, Selection::All);

                // For now, since read_filter is evaluated lazily,
                // "run" a query with no predicates simply to get back the
                // schema
                let predicate = read_buffer::Predicate::default();
                let mut schema = db
                    .read_filter(partition_key, table_name, &[chunk_id], predicate, selection)
                    .context(ReadBufferChunk { chunk_id })?
                    .schema()
                    .context(ReadBufferChunk { chunk_id })?;

                // Ensure the order of the output columns is as
                // specified
                if needs_sort {
                    schema = schema.sort_fields_by_name()
                }

                Ok(schema)
            }
            DBChunk::ParquetFile => {
                unimplemented!("parquet file not implemented for table schema")
            }
        }
    }

    async fn has_table(&self, table_name: &str) -> bool {
        match self {
            DBChunk::MutableBuffer { chunk } => chunk.has_table(table_name).await,
            DBChunk::ReadBuffer {
                db,
                partition_key,
                chunk_id,
            } => {
                let chunk_id = *chunk_id;
                let db = db.read().unwrap();
                db.has_table(partition_key, table_name, &[chunk_id])
            }
            DBChunk::ParquetFile => {
                unimplemented!("parquet file not implemented for has_table")
            }
        }
    }

    async fn read_filter(
        &self,
        table_name: &str,
        predicate: &Predicate,
        selection: Selection<'_>,
    ) -> Result<SendableRecordBatchStream, Self::Error> {
        match self {
            DBChunk::MutableBuffer { chunk } => {
                // Mutable buffer doesn'really support predicate
                // pushdown (other than pruning out the entire chunk
                // via `might_pass_predicate)

                // TODO make some sort of lazy evaluator. For now, materialize here
                let mut batches = Vec::new();
                chunk
                    .table_to_arrow(&mut batches, table_name, selection.clone())
                    .context(MutableBufferChunk)?;
                let schema = if let Some(batch) = batches.iter().next() {
                    batch.schema()
                } else {
                    self.table_schema(table_name, selection).await?.into()
                };

                Ok(Box::pin(MemoryStream::new(batches, schema)))
            }
            DBChunk::ReadBuffer {
                db,
                partition_key,
                chunk_id,
            } => {
                let chunk_id = *chunk_id;
                let rb_predicate =
                    to_read_buffer_predicate(&predicate).context(InternalPredicateConversion)?;
                let db = db.read().expect("mutex poisoned");
                let read_results = db
                    .read_filter(
                        partition_key,
                        table_name,
                        &[chunk_id],
                        rb_predicate,
                        selection,
                    )
                    .context(ReadBufferChunk { chunk_id })?;

                let schema = read_results
                    .schema()
                    .context(ReadBufferChunk { chunk_id })?
                    .into();

                Ok(Box::pin(ReadFilterResultsStream {
                    read_results,
                    schema,
                }))
            }
            DBChunk::ParquetFile => {
                unimplemented!("parquet file not implemented for scan_data")
            }
        }
    }

    fn could_pass_predicate(&self, _predicate: &Predicate) -> Result<bool> {
        match self {
            DBChunk::MutableBuffer { .. } => {
                // For now, we might get an error if we try and
                // compile a chunk predicate (e.g. for tables that
                // don't exist in the chunk) which could signal the
                // chunk can't pass the predicate.

                // However, we can also get an error if there is some
                // unsupported operation and we need a way to
                // distinguish the two cases.
                return Ok(true);
            }
            DBChunk::ReadBuffer { .. } => {
                // TODO: ask Edd how he wants this wired up in the read buffer
                Ok(true)
            }
            DBChunk::ParquetFile => {
                // TODO proper filtering for parquet files
                Ok(true)
            }
        }
    }
}

/// Create something which will stream out a pre-defined set of  record batches
pub(crate) struct MemoryStream {
    /// Vector of record batches to send in reverse order (send data[len-1]
    /// next)
    data: Vec<RecordBatch>,
    /// Schema
    schema: SchemaRef,
    /// Index into the data to next send
    index: usize,
}

impl MemoryStream {
    /// Create an iterator for a vector of record batches
    pub fn new(mut data: Vec<RecordBatch>, schema: SchemaRef) -> Self {
        data.reverse();

        Self {
            data,
            schema,
            index: 0,
        }
    }
}

impl RecordBatchStream for MemoryStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl futures::Stream for MemoryStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Poll::Ready(if self.index < self.data.len() {
            let batch = self.data[self.index].clone();
            self.index += 1;
            Some(Ok(batch))
        } else {
            None
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }
}

/// Adapter which will take a ReadFilterResults and make it an async stream
struct ReadFilterResultsStream {
    read_results: ReadFilterResults,
    schema: SchemaRef,
}

impl RecordBatchStream for ReadFilterResultsStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl futures::Stream for ReadFilterResultsStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Poll::Ready(Ok(self.read_results.next()).transpose())
    }

    // TODO is there a useful size_hint to pass?
}
