// OxideXLA -- Math operator mappings
//
// Maps arithmetic ONNX ops to their JAX equivalents.

use std::collections::HashMap;
use anyhow::Result;

use crate::graph::JaxOp;
use crate::parser::OnnxAttribute;

/// Helper to extract reduction attributes (axes, keepdims).
fn get_reduction_attrs(attributes: &HashMap<String, OnnxAttribute>) -> (Vec<i64>, bool) {
    let axes = if let Some(OnnxAttribute::Ints(v)) = attributes.get("axes") {
        v.clone()
    } else {
        vec![]
    };

    let keepdims = if let Some(OnnxAttribute::Int(val)) = attributes.get("keepdims") {
        *val == 1
    } else {
        true // ONNX default is often 1
    };

    (axes, keepdims)
}

/// Helper to extract float attribute by name.
fn get_float_attr(attributes: &HashMap<String, OnnxAttribute>, name: &str, default: f32) -> f32 {
    if let Some(OnnxAttribute::Float(val)) = attributes.get(name) {
        *val
    } else {
        default
    }
}

/// Helper to extract int attribute as bool.
fn get_bool_attr(attributes: &HashMap<String, OnnxAttribute>, name: &str, default: bool) -> bool {
    if let Some(OnnxAttribute::Int(val)) = attributes.get(name) {
        *val == 1
    } else {
        default
    }
}

/// Map Gemm.
pub fn map_gemm(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let alpha = get_float_attr(attributes, "alpha", 1.0);
    let beta = get_float_attr(attributes, "beta", 1.0);
    let trans_a = get_bool_attr(attributes, "transA", false);
    let trans_b = get_bool_attr(attributes, "transB", false);
    
    Ok(JaxOp::Gemm { alpha, beta, trans_a, trans_b })
}

/// Map ReduceMean.
pub fn map_reducemean(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let (axes, keepdims) = get_reduction_attrs(attributes);
    Ok(JaxOp::ReduceMean { axes, keepdims })
}

/// Map ReduceMax.
pub fn map_reducemax(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let (axes, keepdims) = get_reduction_attrs(attributes);
    Ok(JaxOp::ReduceMax { axes, keepdims })
}

/// Map ReduceMin.
pub fn map_reducemin(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let (axes, keepdims) = get_reduction_attrs(attributes);
    Ok(JaxOp::ReduceMin { axes, keepdims })
}

/// Map ReduceSum.
pub fn map_reducesum(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let (axes, keepdims) = get_reduction_attrs(attributes);
    Ok(JaxOp::ReduceSum { axes, keepdims })
}

/// Map ReduceProd.
pub fn map_reduceprod(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let (axes, keepdims) = get_reduction_attrs(attributes);
    Ok(JaxOp::ReduceProd { axes, keepdims })
}

/// Map Activation attributes (Relu, Elu, etc).
pub fn map_activation(op_type: &str, attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    match op_type {
        "Relu" => Ok(JaxOp::Relu),
        "LeakyRelu" => Ok(JaxOp::LeakyRelu { alpha: get_float_attr(attributes, "alpha", 0.01) }),
        "Elu" => Ok(JaxOp::Elu { alpha: get_float_attr(attributes, "alpha", 1.0) }),
        "Sigmoid" => Ok(JaxOp::Sigmoid),
        "Tanh" => Ok(JaxOp::Tanh),
        "Selu" => Ok(JaxOp::Selu { alpha: get_float_attr(attributes, "alpha", 1.67326), gamma: get_float_attr(attributes, "gamma", 1.0507) }),
        "HardSigmoid" => Ok(JaxOp::HardSigmoid { alpha: get_float_attr(attributes, "alpha", 0.2), beta: get_float_attr(attributes, "beta", 0.5) }),
        "HardSwish" => Ok(JaxOp::HardSwish),
        "Softplus" => Ok(JaxOp::Softplus),
        "Softsign" => Ok(JaxOp::Softsign),
        "ThresholdedRelu" => Ok(JaxOp::ThresholdedRelu { alpha: get_float_attr(attributes, "alpha", 1.0) }),
        "Gelu" => Ok(JaxOp::Gelu),
        "Mish" => Ok(JaxOp::Mish),
        _ => Err(anyhow::anyhow!("Unknown activation: {}", op_type)),
    }
}
