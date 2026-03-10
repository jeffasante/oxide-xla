pub mod parser;
pub mod graph;
pub mod ops;
pub mod codegen;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
