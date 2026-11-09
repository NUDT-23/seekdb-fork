//! Semantic error codes returned to the driver.
//!
//! The public problem package states that until the official error-code
//! mapping is frozen, any non-empty error code is accepted; the codes below
//! follow the semantic names used in the public samples (`duplicate_key`,
//! `dimension_mismatch`) and natural snake_case extensions for the rest.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbError {
    /// INSERT hits an existing primary key.
    DuplicateKey,
    /// Statement is outside the supported SQL subset (ALTER TABLE, JOIN, ...).
    UnsupportedSql,
    /// Vector literal has wrong dimension for the column.
    DimensionMismatch,
    /// Vector literal contains NaN/Infinity or a value that overflows float32.
    InvalidVector,
    /// BIGINT literal outside i64 range.
    ValueOverflow,
    /// VARCHAR value longer (in Unicode chars) than declared.
    StringTooLong,
    TableExists,
    TableNotFound,
    ColumnNotFound,
    IndexExists,
    SyntaxError,
    /// BEGIN inside an active transaction / COMMIT|ROLLBACK without one.
    InvalidTransactionState,
    /// Operand types do not compare (e.g. number vs string).
    TypeMismatch,
    /// Internal invariant violation (should never happen).
    Internal,
}

impl DbError {
    pub fn code(&self) -> &'static str {
        match self {
            DbError::DuplicateKey => "duplicate_key",
            DbError::UnsupportedSql => "unsupported_sql",
            DbError::DimensionMismatch => "dimension_mismatch",
            DbError::InvalidVector => "invalid_vector",
            DbError::ValueOverflow => "value_overflow",
            DbError::StringTooLong => "string_too_long",
            DbError::TableExists => "table_exists",
            DbError::TableNotFound => "table_not_found",
            DbError::ColumnNotFound => "column_not_found",
            DbError::IndexExists => "index_exists",
            DbError::SyntaxError => "syntax_error",
            DbError::InvalidTransactionState => "invalid_transaction_state",
            DbError::TypeMismatch => "type_mismatch",
            DbError::Internal => "internal_error",
        }
    }
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

impl std::error::Error for DbError {}

pub type DbResult<T> = Result<T, DbError>;
