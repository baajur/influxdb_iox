//! Implementation of a DataFusion TableProvider in terms of PartitionChunks
//!
//! This allows DataFusion to see data from Chunks as a single table

use std::sync::Arc;

use arrow_deps::{
    arrow::datatypes::SchemaRef,
    datafusion::{
        datasource::{datasource::Statistics, TableProvider},
        error::DataFusionError,
        logical_plan::Expr,
        physical_plan::{ExecutionPlan, Partitioning, SendableRecordBatchStream},
    },
};
use data_types::selection::Selection;

use crate::{predicate::Predicate, PartitionChunk};

use async_trait::async_trait;
use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display(
        "Chunk schema not compatible for table '{}'. They must be identical. Existing: {:?}, New: {:?}",
        table_name,
        existing_schema,
        chunk_schema
    ))]
    ChunkSchemaNotCompatible {
        table_name: String,
        existing_schema: SchemaRef,
        chunk_schema: SchemaRef,
    },

    #[snafu(display("No rows found in table {}", table_name))]
    InternalNoRowsInTable { table_name: String },
}
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Builds a ChunkTableProvider that ensures the schema across chunks is
/// compatible
#[derive(Debug)]
pub struct ProviderBuilder<C: PartitionChunk + 'static> {
    table_name: String,
    schema: Option<SchemaRef>,
    chunks: Vec<Arc<C>>,
}

impl<C: PartitionChunk> ProviderBuilder<C> {
    pub fn new(table_name: impl Into<String>) -> Self {
        Self {
            table_name: table_name.into(),
            schema: None,
            chunks: Vec::new(),
        }
    }

    pub fn add_chunk(mut self, chunk: Arc<C>, chunk_table_schema: SchemaRef) -> Result<Self> {
        self.schema = Some(if let Some(existing_schema) = self.schema.take() {
            self.check_schema(existing_schema, chunk_table_schema)?
        } else {
            chunk_table_schema
        });
        self.chunks.push(chunk);
        Ok(self)
    }

    /// returns Ok(combined_schema) if the schema of chunk is compatible with
    /// `existing_schema`, Err() with why otherwise
    fn check_schema(
        &self,
        existing_schema: SchemaRef,
        chunk_schema: SchemaRef,
    ) -> Result<SchemaRef> {
        // For now, use strict equality. Eventually should union the schema
        if existing_schema != chunk_schema {
            ChunkSchemaNotCompatible {
                table_name: &self.table_name,
                existing_schema,
                chunk_schema,
            }
            .fail()
        } else {
            Ok(chunk_schema)
        }
    }

    pub fn build(self) -> Result<ChunkTableProvider<C>> {
        let Self {
            table_name,
            schema,
            chunks,
        } = self;

        // TODO proper error handling
        let schema = schema.unwrap();

        // if the table was reported to exist, it should not be empty (eventually we
        // should get the schema and table data separtely)
        if chunks.is_empty() {
            return InternalNoRowsInTable { table_name }.fail();
        }

        Ok(ChunkTableProvider {
            table_name,
            schema,
            chunks,
        })
    }
}

// Implementation of a DataFusion TableProvider in terms of PartitionChunks
#[derive(Debug)]
pub struct ChunkTableProvider<C: PartitionChunk> {
    // TODO make this an Arc
    table_name: String,
    schema: SchemaRef,
    chunks: Vec<Arc<C>>,
}

impl<C: PartitionChunk + 'static> TableProvider for ChunkTableProvider<C> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn scan(
        &self,
        projection: &Option<Vec<usize>>,
        _batch_size: usize,
        filters: &[Expr],
    ) -> std::result::Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // TODO make a Predicate from the Expr (to implement pushdown
        // predicates). Note that the predicates don't actually need to be
        // evaluated in the scan for the plans to be correct
        let predicate = Predicate::default();

        let filters = filters.to_vec();

        let plan = IOxReadFilterNode {
            table_name: self.table_name.clone(),
            schema: self.schema.clone(),
            chunks: self.chunks.clone(),
            predicate,
            projection: projection.clone(),
            filters,
        };

        Ok(Arc::new(plan))
    }

    fn statistics(&self) -> Statistics {
        // TODO translate IOx stats to DataFusion statistics
        Statistics::default()
    }
}

// Structure around a bunch of sendable streams
#[derive(Debug)]
struct IOxReadFilterNode<C: PartitionChunk + 'static> {
    table_name: String,
    schema: SchemaRef,
    chunks: Vec<Arc<C>>,
    predicate: Predicate,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
}

#[async_trait]
impl<C: PartitionChunk + 'static> ExecutionPlan for IOxReadFilterNode<C> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.chunks.len())
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        // no inputs
        vec![]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> arrow_deps::datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        assert!(children.is_empty(), "no children expected in iox plan");

        // For some reason when I used an automatically derived `Clone` implementation
        // the compiler didn't recognize the trait implementation
        let new_self = Self {
            table_name: self.table_name.clone(),
            schema: self.schema.clone(),
            chunks: self.chunks.clone(),
            predicate: self.predicate.clone(),
            projection: self.projection.clone(),
            filters: self.filters.clone(),
        };

        Ok(Arc::new(new_self))
    }

    async fn execute(
        &self,
        partition: usize,
    ) -> arrow_deps::datafusion::error::Result<SendableRecordBatchStream> {
        let fields = self.schema.fields();
        let selection_cols = self.projection.as_ref().map(|projection| {
            projection
                .iter()
                .map(|&index| fields[index].name() as &str)
                .collect::<Vec<_>>()
        });

        let selection = if let Some(selection_cols) = selection_cols.as_ref() {
            Selection::Some(&selection_cols)
        } else {
            Selection::All
        };

        let chunk = &self.chunks[partition];
        chunk
            .read_filter(&self.table_name, &self.predicate, selection)
            .await
            .map_err(|e| {
                DataFusionError::Execution(format!(
                    "Error creating scan for table {} chunk {}: {}",
                    self.table_name,
                    chunk.id(),
                    e
                ))
            })
    }
}
