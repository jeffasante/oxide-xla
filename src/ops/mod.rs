use std::collections::HashMap;
use anyhow::Result;

use crate::graph::JaxOp;
use crate::parser::OnnxAttribute;

pub mod nn;
pub mod math;
pub mod reshape;

/// Map a single ONNX operator to a JaxOp variant.
pub fn map_onnx_op(op_type: &str, attributes: &HashMap<String, OnnxAttribute>) -> Result<JaxOp> {
    match op_type {
        // Basic Math (binary/elementwise)
        "Add" => Ok(JaxOp::Add),
        "Sub" => Ok(JaxOp::Sub),
        "Mul" => Ok(JaxOp::Mul),
        "Div" => Ok(JaxOp::Div),
        "Pow" => Ok(JaxOp::Pow),
        "Sqrt" => Ok(JaxOp::Sqrt),
        "Exp" => Ok(JaxOp::Exp),
        "Log" => Ok(JaxOp::Log),
        "Abs" => Ok(JaxOp::Abs),
        "Neg" => Ok(JaxOp::Neg),
        "Ceil" => Ok(JaxOp::Ceil),
        "Floor" => Ok(JaxOp::Floor),
        "Sign" => Ok(JaxOp::Sign),
        "Reciprocal" => Ok(JaxOp::Reciprocal),
        "Round" => Ok(JaxOp::Round),
        "Cos" => Ok(JaxOp::Cos),
        "Sin" => Ok(JaxOp::Sin),
        "Tan" => Ok(JaxOp::Tan),
        "Acos" => Ok(JaxOp::Acos),
        "Asin" => Ok(JaxOp::Asin),
        "Atan" => Ok(JaxOp::Atan),
        "Cosh" => Ok(JaxOp::Cosh),
        "Sinh" => Ok(JaxOp::Sinh),
        "Tanh" => Ok(JaxOp::Tanh),
        "Acosh" => Ok(JaxOp::Acosh),
        "Asinh" => Ok(JaxOp::Asinh),
        "Atanh" => Ok(JaxOp::Atanh),
        "Erf" => Ok(JaxOp::Erf),

        // Logic
        "And" => Ok(JaxOp::And),
        "Or" => Ok(JaxOp::Or),
        "Xor" => Ok(JaxOp::Xor),
        "Not" => Ok(JaxOp::Not),
        "Equal" => Ok(JaxOp::Equal),
        "Greater" => Ok(JaxOp::Greater),
        "GreaterOrEqual" => Ok(JaxOp::GreaterOrEqual),
        "Less" => Ok(JaxOp::Less),
        "LessOrEqual" => Ok(JaxOp::LessOrEqual),
        "Where" => Ok(JaxOp::Where),

        // Neural Network
        "Conv" => nn::map_conv(attributes),
        "ConvTranspose" => nn::map_conv_transpose(attributes),
        "MaxPool" => nn::map_maxpool(attributes),
        "AveragePool" => nn::map_averagepool(attributes),
        "GlobalAveragePool" => nn::map_global_avg_pool(),
        "BatchNormalization" => nn::map_batchnorm(attributes),
        "LayerNormalization" => nn::map_layernorm(attributes),
        "Softmax" => nn::map_softmax(attributes),
        "Flatten" => reshape::map_reshape(attributes),

        // Activations
        "Relu" | "LeakyRelu" | "Elu" | "Sigmoid" | "Selu" | "HardSigmoid" | "HardSwish" | 
        "Softplus" | "Softsign" | "ThresholdedRelu" | "Gelu" | "Mish" => {
            math::map_activation(op_type, attributes)
        }
        "PRelu" => Ok(JaxOp::PRelu),

        // Reductions
        "ReduceMean" => math::map_reducemean(attributes),
        "ReduceMax" => math::map_reducemax(attributes),
        "ReduceMin" => math::map_reducemin(attributes),
        "ReduceSum" => math::map_reducesum(attributes),
        "ReduceProd" => math::map_reduceprod(attributes),

        // Linear Algebra
        "MatMul" => Ok(JaxOp::MatMul),
        "Gemm" => math::map_gemm(attributes),

        // Shape manipulation
        "Reshape" => reshape::map_reshape(attributes),
        "Transpose" => reshape::map_transpose(attributes),
        "Concat" => reshape::map_concat(attributes),
        "Gather" => reshape::map_gather(attributes),
        "Split" => reshape::map_split(attributes),
        "Slice" => reshape::map_slice(attributes),
        "Squeeze" => reshape::map_squeeze(attributes),
        "Unsqueeze" => reshape::map_unsqueeze(attributes),
        "Resize" => Ok(JaxOp::Resize),
        "DepthToSpace" => Ok(JaxOp::DepthToSpace),
        "Shape" => Ok(JaxOp::Shape),
        "Identity" => Ok(JaxOp::Identity),
        "Tile" => Ok(JaxOp::Tile),
        "Expand" => Ok(JaxOp::Expand),
        "Pad" => reshape::map_pad(attributes),
        "Cast" => reshape::map_cast(attributes),

        // Quantization / CV Stubs
        "DynamicQuantizeLinear" => Ok(JaxOp::DynamicQuantizeLinear),
        "Constant" => reshape::map_constant(attributes),
        "Clip" => Ok(JaxOp::Clip),

        // Falling back to Unknown for unmapped operators
        op => Ok(JaxOp::Unknown(op.to_string())),
    }
}
