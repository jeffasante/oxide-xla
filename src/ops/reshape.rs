// OxideXLA -- Reshape operator mappings
//
// Maps shape manipulation ONNX ops to their JAX equivalents.

use std::collections::HashMap;
use anyhow::Result;

use crate::graph::JaxOp;
use crate::parser::OnnxAttribute;

/// Reshape -> jnp.reshape(X, target_shape)
///
/// In ONNX, the target shape comes from a second input tensor,
/// not from an attribute. During graph construction we may not
/// have the shape available as a compile-time constant. In that
/// case the target_shape will be empty and must be resolved
/// during shape inference from the initializer data.
pub fn map_reshape(_attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    // The actual target shape is determined during shape inference
    // by reading the second input (which is typically a constant initializer).
    Ok(JaxOp::Reshape {
        target_shape: vec![],
    })
}

/// Transpose -> jnp.transpose(X, perm)
///
/// The "perm" attribute specifies the permutation of dimensions.
/// If not provided, reverses all dimensions (standard transpose).
pub fn map_transpose(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let perm = match attributes.get("perm") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![], // Empty means reverse all dims.
    };
    Ok(JaxOp::Transpose { perm })
}

/// Concat -> jnp.concatenate
pub fn map_concat(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axis = match attributes.get("axis") {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => 0,
    };
    Ok(JaxOp::Concat { axis })
}

/// Gather -> jnp.take
pub fn map_gather(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axis = match attributes.get("axis") {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => 0,
    };
    Ok(JaxOp::Gather { axis })
}
