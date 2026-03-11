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

/// Split -> jnp.split or indexing
///
/// ONNX Split has an "axis" attribute and optionally a "split" attribute
/// (list of sizes) or "num_outputs" attribute. In newer opsets (>=13),
/// the split sizes come from a second input tensor.
pub fn map_split(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axis = match attributes.get("axis") {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => 0,
    };
    let num_outputs = match attributes.get("num_outputs") {
        Some(OnnxAttribute::Int(v)) => *v as usize,
        _ => {
            // Try "split" attribute for older opsets
            match attributes.get("split") {
                Some(OnnxAttribute::Ints(v)) => v.len(),
                _ => 3, // Default for QKV split in transformers
            }
        }
    };
    Ok(JaxOp::Split { axis, num_outputs })
}

/// Slice -> jax.lax.dynamic_slice or Python slice syntax
///
/// ONNX Slice can have attributes (opset < 10) or inputs (opset >= 10).
/// For opset >= 10, starts/ends/axes/steps come from input tensors,
/// which we resolve during graph construction.
pub fn map_slice(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let starts = match attributes.get("starts") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    let ends = match attributes.get("ends") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    let axes = match attributes.get("axes") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    let steps = match attributes.get("steps") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    Ok(JaxOp::Slice { starts, ends, axes, steps })
}

/// Squeeze -> jnp.squeeze(X, axis=axes)
///
/// ONNX Squeeze (opset >= 13) takes axes from the second input.
/// For older opsets, axes come from an attribute.
pub fn map_squeeze(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axes = match attributes.get("axes") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    Ok(JaxOp::Squeeze { axes })
}

/// Unsqueeze -> jnp.expand_dims(X, axis=axes)
///
/// ONNX Unsqueeze (opset >= 13) takes axes from the second input.
/// For older opsets, axes come from an attribute.
pub fn map_unsqueeze(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axes = match attributes.get("axes") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    Ok(JaxOp::Unsqueeze { axes })
}

/// Pad -> jnp.pad(X, pad_width, mode='constant', constant_values=0)
pub fn map_pad(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let pads = match attributes.get("pads") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    };
    let mode = match attributes.get("mode") {
        Some(OnnxAttribute::String(s)) => s.clone(),
        _ => "constant".to_string(),
    };
    let constant_value = match attributes.get("value") {
        Some(OnnxAttribute::Float(v)) => *v as f64,
        _ => 0.0,
    };
    Ok(JaxOp::Pad { pads, mode, constant_value })
}

/// Cast -> x.astype(target_dtype)
///
/// ONNX Cast has a "to" attribute with the target data type code.
pub fn map_cast(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let to_dtype = match attributes.get("to") {
        Some(OnnxAttribute::Int(v)) => *v as i32,
        _ => 1, // Default to float32
    };
    Ok(JaxOp::Cast { to_dtype })
}

/// Constant -> literal value
///
/// ONNX Constant nodes carry their value in the "value" attribute
/// (as a TensorProto) or as "value_float", "value_int", etc.
pub fn map_constant(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    // Try value_float first (simplest case)
    if let Some(OnnxAttribute::Float(v)) = attributes.get("value_float") {
        return Ok(JaxOp::Constant { value: *v as f64 });
    }
    // Try value_int
    if let Some(OnnxAttribute::Int(v)) = attributes.get("value_int") {
        return Ok(JaxOp::Constant { value: *v as f64 });
    }
    // Try value (TensorProto) — extract scalar if possible
    if let Some(OnnxAttribute::Tensor { float_data, raw_data, int64_data, shape, data_type, .. }) = attributes.get("value") {
        // Check if it's a scalar or small tensor
        let is_scalar = shape.is_empty() || shape.iter().all(|&d| d == 1);

        if !float_data.is_empty() {
            return Ok(JaxOp::Constant { value: float_data[0] as f64 });
        }
        if !int64_data.is_empty() {
            return Ok(JaxOp::Constant { value: int64_data[0] as f64 });
        }
        if !raw_data.is_empty() && is_scalar {
            // Try to interpret raw_data based on data_type
            match data_type {
                1 if raw_data.len() >= 4 => {
                    // float32
                    let val = f32::from_le_bytes([raw_data[0], raw_data[1], raw_data[2], raw_data[3]]);
                    return Ok(JaxOp::Constant { value: val as f64 });
                }
                7 if raw_data.len() >= 8 => {
                    // int64
                    let val = i64::from_le_bytes([
                        raw_data[0], raw_data[1], raw_data[2], raw_data[3],
                        raw_data[4], raw_data[5], raw_data[6], raw_data[7],
                    ]);
                    return Ok(JaxOp::Constant { value: val as f64 });
                }
                11 if raw_data.len() >= 8 => {
                    // float64
                    let val = f64::from_le_bytes([
                        raw_data[0], raw_data[1], raw_data[2], raw_data[3],
                        raw_data[4], raw_data[5], raw_data[6], raw_data[7],
                    ]);
                    return Ok(JaxOp::Constant { value: val });
                }
                6 if raw_data.len() >= 4 => {
                    // int32
                    let val = i32::from_le_bytes([raw_data[0], raw_data[1], raw_data[2], raw_data[3]]);
                    return Ok(JaxOp::Constant { value: val as f64 });
                }
                _ => {}
            }
        }
    }
    // Fallback
    Ok(JaxOp::Constant { value: 0.0 })
}
