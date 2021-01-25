//! This represents selecting a set of columns
#[derive(Debug)]
pub enum Selection<'a> {
    /// Return all columns (e.g. SELECT *)
    AllColumns,

    /// Return only the named columns
    SpecifiedColumns(&'a [&'a str]),
}

impl<'a> From<&'a [&'a str]> for Selection<'a> {
    fn from(names: &'a [&'a str]) -> Self {
        Self::SpecifiedColumns(names)
    }
}
