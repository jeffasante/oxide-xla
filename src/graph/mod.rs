mod dag;
mod shape;

pub use dag::{IrGraph, IrNode, JaxOp, DType};
pub use shape::infer_shapes;
