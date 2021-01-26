//! Implementation of a DataFusion TableProvider in terms of PartitionChunks
//!
//! This allows DataFusion to see data from Chunks as a single table

use std::sync::Arc;

use arrow_deps::{
    arrow::datatypes::SchemaRef,
    datafusion::{
        datasource::TableProvider, error::DataFusionError, logical_plan::Expr,
        physical_plan::ExecutionPlan,
    },
};

use crate::PartitionChunk;

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
        _projection: &Option<Vec<usize>>,
        _batch_size: usize,
        _filters: &[Expr],
    ) -> std::result::Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // make an execution plan that just holds the sendable streams from each chunk

        todo!()
    }

    fn statistics(&self) -> arrow_deps::datafusion::datasource::datasource::Statistics {
        todo!()
    }
}
