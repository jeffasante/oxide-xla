// OxideXLA -- Neural network operator mappings
//
// Maps neural network ONNX ops (activations, normalization,
// convolution) to their JAX equivalents.

use std::collections::HashMap;
use anyhow::Result;

use crate::graph::JaxOp;
use crate::parser::OnnxAttribute;

/// Relu -> jax.nn.relu
pub fn map_relu() -> Result<JaxOp> {
    Ok(JaxOp::Relu)
}

/// Sigmoid -> jax.nn.sigmoid
pub fn map_sigmoid() -> Result<JaxOp> {
    Ok(JaxOp::Sigmoid)
}

/// Tanh -> jnp.tanh
pub fn map_tanh() -> Result<JaxOp> {
    Ok(JaxOp::Tanh)
}

/// Softmax -> jax.nn.softmax(X, axis=axis)
///
/// ONNX Softmax has an "axis" attribute (default -1 in opset >= 13).
pub fn map_softmax(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axis = match attributes.get("axis") {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => -1, // Default: last axis
    };
    Ok(JaxOp::Softmax { axis })
}

/// BatchNormalization -> manual decomposition
///
/// JAX does not have a built-in batch norm op, so we decompose it:
///   output = (input - mean) / sqrt(variance + epsilon) * scale + bias
pub fn map_batchnorm(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let epsilon = match attributes.get("epsilon") {
        Some(OnnxAttribute::Float(v)) => *v,
        _ => 1e-5, // Standard default
    };
    Ok(JaxOp::BatchNorm { epsilon })
}

/// LayerNormalization -> manual decomposition or jax.nn.standardize
pub fn map_layernorm(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let axis = match attributes.get("axis") {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => -1, // typical for layernorm
    };
    let epsilon = match attributes.get("epsilon") {
        Some(OnnxAttribute::Float(v)) => *v,
        _ => 1e-5,
    };
    Ok(JaxOp::LayerNormalization { axis, epsilon })
}

/// Conv -> jax.lax.conv_general_dilated
///
/// Reads strides, pads, dilations, and group from ONNX attributes.
pub fn map_conv(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let strides = match attributes.get("strides") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![1, 1],
    };

    let pads = match attributes.get("pads") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![0, 0, 0, 0],
    };

    let dilations = match attributes.get("dilations") {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![1, 1],
    };

    let group = match attributes.get("group") {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => 1,
    };

    Ok(JaxOp::Conv {
        strides,
        pads,
        dilations,
        group,
    })
}

/// Map MaxPool.
pub fn map_maxpool(attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    let strides = if let Some(OnnxAttribute::Ints(v)) = attributes.get("strides") {
        v.clone()
    } else {
        vec![1, 1]
    };

    let kernel_shape = if let Some(OnnxAttribute::Ints(v)) = attributes.get("kernel_shape") {
        v.clone()
    } else {
        vec![1, 1]
    };

    let pads = if let Some(OnnxAttribute::Ints(v)) = attributes.get("pads") {
        v.clone()
    } else {
        vec![0, 0, 0, 0]
    };

    Ok(JaxOp::MaxPool {
        strides,
        kernel_shape,
        pads,
    })
}

/// Map AveragePool / GlobalAveragePool.
pub fn map_global_avg_pool() -> Result<JaxOp> {
    Ok(JaxOp::ReduceMean { axes: vec![2, 3], keepdims: true })
}
