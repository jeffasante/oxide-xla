// OxideXLA -- Per-Node JAX Code Emission
//
// This module contains the logic for turning a single IrNode into
// one line of Python code. Each JaxOp variant maps to exactly one
// JAX function call.

use anyhow::{Result, bail};

use crate::graph::{IrNode, JaxOp};

/// Convert ONNX dtype code to JAX numpy dtype string.
fn onnx_dtype_to_jnp(code: i32) -> &'static str {
    match code {
        1 => "jnp.float32",
        2 => "jnp.uint8",
        3 => "jnp.int8",
        5 => "jnp.int16",
        6 => "jnp.int32",
        7 => "jnp.int64",
        9 => "jnp.bool_",
        10 => "jnp.float16",
        11 => "jnp.float64",
        12 => "jnp.uint32",
        13 => "jnp.uint64",
        16 => "jnp.bfloat16",
        _ => "jnp.float32",
    }
}

/// Emit one line of Python code for the given IR node.
pub fn emit_node(
    node: &IrNode,
    input_vars: &[String],
    output_var: &str,
) -> Result<String> {
    let code = match &node.op {
        JaxOp::Input | JaxOp::Param => {
            return Ok(format!("# {} (input/param, no code needed)", output_var));
        }

        // Linear Algebra
        JaxOp::MatMul => {
            ensure_inputs("MatMul", input_vars, 2)?;
            format!("{} = jnp.matmul({}, {})", output_var, input_vars[0], input_vars[1])
        }
        JaxOp::Gemm { alpha, beta, trans_a, trans_b } => {
            ensure_inputs("Gemm", input_vars, 2)?;
            let a = if *trans_a { format!("{}.T", input_vars[0]) } else { input_vars[0].clone() };
            let b = if *trans_b { format!("{}.T", input_vars[1]) } else { input_vars[1].clone() };
            let mut code = format!("{} = {} * jnp.matmul({}, {})", output_var, alpha, a, b);
            if input_vars.len() >= 3 {
                code.push_str(&format!(" + {} * {}", beta, input_vars[2]));
            }
            code
        }

        // Basic Math
        JaxOp::Add => {
            ensure_inputs("Add", input_vars, 2)?;
            format!("{} = jnp.add({}, {})", output_var, input_vars[0], input_vars[1])
        }
        JaxOp::Sub => {
            ensure_inputs("Sub", input_vars, 2)?;
            format!("{} = jnp.subtract({}, {})", output_var, input_vars[0], input_vars[1])
        }
        JaxOp::Mul => {
            ensure_inputs("Mul", input_vars, 2)?;
            format!("{} = jnp.multiply({}, {})", output_var, input_vars[0], input_vars[1])
        }
        JaxOp::Div => {
            ensure_inputs("Div", input_vars, 2)?;
            format!("{} = jnp.divide({}, {})", output_var, input_vars[0], input_vars[1])
        }
        JaxOp::Pow => {
            ensure_inputs("Pow", input_vars, 2)?;
            format!("{} = jnp.power({}, {})", output_var, input_vars[0], input_vars[1])
        }
        JaxOp::Abs => format!("{} = jnp.abs({})", output_var, input_vars[0]),
        JaxOp::Exp => format!("{} = jnp.exp({})", output_var, input_vars[0]),
        JaxOp::Log => format!("{} = jnp.log({})", output_var, input_vars[0]),
        JaxOp::Neg => format!("{} = jnp.negative({})", output_var, input_vars[0]),
        JaxOp::Sqrt => format!("{} = jnp.sqrt({})", output_var, input_vars[0]),
        JaxOp::Ceil => format!("{} = jnp.ceil({})", output_var, input_vars[0]),
        JaxOp::Floor => format!("{} = jnp.floor({})", output_var, input_vars[0]),
        JaxOp::Round => format!("{} = jnp.round({})", output_var, input_vars[0]),
        JaxOp::Reciprocal => format!("{} = jnp.reciprocal({})", output_var, input_vars[0]),
        JaxOp::Sign => format!("{} = jnp.sign({})", output_var, input_vars[0]),
        JaxOp::Cos => format!("{} = jnp.cos({})", output_var, input_vars[0]),
        JaxOp::Sin => format!("{} = jnp.sin({})", output_var, input_vars[0]),
        JaxOp::Tan => format!("{} = jnp.tan({})", output_var, input_vars[0]),
        JaxOp::Acos => format!("{} = jnp.arccos({})", output_var, input_vars[0]),
        JaxOp::Asin => format!("{} = jnp.arcsin({})", output_var, input_vars[0]),
        JaxOp::Atan => format!("{} = jnp.arctan({})", output_var, input_vars[0]),
        JaxOp::Cosh => format!("{} = jnp.cosh({})", output_var, input_vars[0]),
        JaxOp::Sinh => format!("{} = jnp.sinh({})", output_var, input_vars[0]),
        JaxOp::Tanh => format!("{} = jnp.tanh({})", output_var, input_vars[0]),
        JaxOp::Acosh => format!("{} = jnp.arccosh({})", output_var, input_vars[0]),
        JaxOp::Asinh => format!("{} = jnp.arcsinh({})", output_var, input_vars[0]),
        JaxOp::Atanh => format!("{} = jnp.arctanh({})", output_var, input_vars[0]),
        JaxOp::Erf => format!("{} = jax.scipy.special.erf({})", output_var, input_vars[0]),

        // Logic
        JaxOp::And => format!("{} = jnp.logical_and({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::Or => format!("{} = jnp.logical_or({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::Xor => format!("{} = jnp.logical_xor({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::Not => format!("{} = jnp.logical_not({})", output_var, input_vars[0]),
        JaxOp::Equal => format!("{} = jnp.equal({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::Greater => format!("{} = jnp.greater({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::GreaterOrEqual => format!("{} = jnp.greater_equal({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::Less => format!("{} = jnp.less({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::LessOrEqual => format!("{} = jnp.less_equal({}, {})", output_var, input_vars[0], input_vars[1]),
        JaxOp::Where => format!("{} = jnp.where({}, {}, {})", output_var, input_vars[0], input_vars[1], input_vars[2]),

        // Reductions
        JaxOp::ReduceMax { axes, keepdims } => {
            format!("{} = jnp.max({}, axis=({}), keepdims={})", output_var, input_vars[0], axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "), if *keepdims { "True" } else { "False" })
        }
        JaxOp::ReduceMin { axes, keepdims } => {
            format!("{} = jnp.min({}, axis=({}), keepdims={})", output_var, input_vars[0], axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "), if *keepdims { "True" } else { "False" })
        }
        JaxOp::ReduceSum { axes, keepdims } => {
            format!("{} = jnp.sum({}, axis=({}), keepdims={})", output_var, input_vars[0], axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "), if *keepdims { "True" } else { "False" })
        }
        JaxOp::ReduceProd { axes, keepdims } => {
            format!("{} = jnp.prod({}, axis=({}), keepdims={})", output_var, input_vars[0], axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "), if *keepdims { "True" } else { "False" })
        }
        JaxOp::ReduceMean { axes, keepdims } => {
            format!("{} = jnp.mean({}, axis=({}), keepdims={})", output_var, input_vars[0], axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "), if *keepdims { "True" } else { "False" })
        }

        // Activations
        JaxOp::Relu => format!("{} = jax.nn.relu({})", output_var, input_vars[0]),
        JaxOp::LeakyRelu { alpha } => format!("{} = jax.nn.leaky_relu({}, negative_slope={})", output_var, input_vars[0], alpha),
        JaxOp::Elu { alpha } => format!("{} = jax.nn.elu({}, alpha={})", output_var, input_vars[0], alpha),
        JaxOp::Sigmoid => format!("{} = jax.nn.sigmoid({})", output_var, input_vars[0]),
        JaxOp::Selu { alpha, gamma } => format!("{} = ({} * jax.nn.elu({}, alpha={}))", output_var, gamma, input_vars[0], alpha),
        JaxOp::HardSigmoid { alpha, beta } => format!("{} = jax.nn.hard_sigmoid({} * {} + {})", output_var, alpha, input_vars[0], beta),
        JaxOp::HardSwish => format!("{} = jax.nn.hard_swish({})", output_var, input_vars[0]),
        JaxOp::Softplus => format!("{} = jax.nn.softplus({})", output_var, input_vars[0]),
        JaxOp::Softsign => format!("{} = jax.nn.soft_sign({})", output_var, input_vars[0]),
        JaxOp::ThresholdedRelu { alpha } => format!("{} = jnp.where({} > {}, {}, 0.0)", output_var, input_vars[0], alpha, input_vars[0]),
        JaxOp::Gelu => format!("{} = jax.nn.gelu({})", output_var, input_vars[0]),
        JaxOp::Mish => format!("{} = {} * jnp.tanh(jax.nn.softplus({}))", output_var, input_vars[0], input_vars[0]),
        JaxOp::PRelu => format!("{} = jnp.where({} >= 0, {}, {} * {})", output_var, input_vars[0], input_vars[0], input_vars[0], input_vars[1]),

        // Neural Network Components
        JaxOp::Conv { strides, pads, dilations, group } => {
            let padding_str = pads.chunks(2).map(|c| format!("({}, {})", c[0], c[1])).collect::<Vec<_>>().join(", ");
            format!("{} = jax.lax.conv_general_dilated({}, {}, window_strides=({}), padding=[{}], rhs_dilation=({}), feature_group_count={})",
                output_var, input_vars[0], input_vars[1], 
                strides.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
                padding_str,
                dilations.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
                group)
        }
        JaxOp::ConvTranspose { strides, pads, dilations, group, .. } => {
            let padding_str = pads.chunks(2).map(|c| format!("({}, {})", c[0], c[1])).collect::<Vec<_>>().join(", ");
            format!("{} = jax.lax.conv_transpose({}, {}, strides=({}), padding=[{}], rhs_dilation=({}), feature_group_count={})",
                output_var, input_vars[0], input_vars[1], 
                strides.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
                padding_str,
                dilations.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
                group)
        }
        JaxOp::BatchNorm { epsilon } => {
            format!("{} = ({} - {}.reshape(1, -1, 1, 1)) / jnp.sqrt({}.reshape(1, -1, 1, 1) + {}) * {}.reshape(1, -1, 1, 1) + {}.reshape(1, -1, 1, 1)",
                output_var, input_vars[0], input_vars[3], input_vars[4], epsilon, input_vars[1], input_vars[2])
        }
        JaxOp::LayerNormalization { axis, epsilon } => {
            format!("{0}_mean = jnp.mean({1}, axis={2}, keepdims=True); {0} = ({1} - {0}_mean) / jnp.sqrt(jnp.var({1}, axis={2}, keepdims=True) + {3}) * {4} + {5}",
                output_var, input_vars[0], axis, epsilon, input_vars[1], input_vars[2])
        }
        JaxOp::MaxPool { strides, kernel_shape, pads } => {
            let padding_str = pads.chunks(2).map(|c| format!("({}, {})", c[0], c[1])).collect::<Vec<_>>().join(", ");
            format!("{} = jax.lax.reduce_window({}, -jnp.inf, jax.lax.max, ({}), ({}), [(0,0),(0,0),{}])",
                output_var, input_vars[0], 
                kernel_shape.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", "),
                strides.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
                padding_str)
        }
        JaxOp::AveragePool { kernel_shape, strides, pads } => {
            // Use jax.lax.reduce_window with addition, then divide by kernel volume
            let kh = kernel_shape.get(0).copied().unwrap_or(1);
            let kw = kernel_shape.get(1).copied().unwrap_or(1);
            let kernel_vol = kh * kw;
            let padding_str = pads.chunks(2).map(|c| format!("({}, {})", c[0], c[1])).collect::<Vec<_>>().join(", ");
            format!("{} = jax.lax.reduce_window({}, 0.0, jax.lax.add, (1, 1, {}, {}), (1, 1, {}, {}), [(0,0),(0,0),{}]) / {}",
                output_var, input_vars[0],
                kh, kw,
                strides.get(0).unwrap_or(&1), strides.get(1).unwrap_or(&1),
                padding_str,
                kernel_vol)
        }
        JaxOp::Softmax { axis } => format!("{} = jax.nn.softmax({}, axis={})", output_var, input_vars[0], axis),

        // Shape Manipulation
        JaxOp::Reshape { target_shape } => {
            if !target_shape.is_empty() {
                format!("{} = jnp.reshape({}, ({}))", output_var, input_vars[0], target_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
            } else if input_vars.len() >= 2 {
                format!("{} = jnp.reshape({}, {})", output_var, input_vars[0], input_vars[1])
            } else {
                format!("{} = jnp.reshape({}, ())", output_var, input_vars[0])
            }
        },
        JaxOp::Transpose { perm } => format!("{} = jnp.transpose({}, ({}))", output_var, input_vars[0], perm.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ")),
        JaxOp::Concat { axis } => format!("{} = jnp.concatenate([{}], axis={})", output_var, input_vars.join(", "), axis),
        JaxOp::Gather { axis } => format!("{} = jnp.take({}, {}, axis={})", output_var, input_vars[0], input_vars[1], axis),

        // Split -> split along axis and return list, then index by output
        JaxOp::Split { axis, num_outputs } => {
            format!("{} = jnp.split({}, {}, axis={})", output_var, input_vars[0], num_outputs, axis)
        }

        // Squeeze with proper axes
        JaxOp::Squeeze { axes } => {
            if axes.is_empty() {
                format!("{} = jnp.squeeze({})", output_var, input_vars[0])
            } else if axes.len() == 1 {
                format!("{} = jnp.squeeze({}, axis={})", output_var, input_vars[0], axes[0])
            } else {
                format!("{} = jnp.squeeze({}, axis=({}))", output_var, input_vars[0],
                    axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "))
            }
        }

        // Unsqueeze with proper axes
        JaxOp::Unsqueeze { axes } => {
            if axes.is_empty() {
                format!("{} = jnp.expand_dims({}, axis=0)", output_var, input_vars[0])
            } else if axes.len() == 1 {
                format!("{} = jnp.expand_dims({}, axis={})", output_var, input_vars[0], axes[0])
            } else {
                format!("{} = jnp.expand_dims({}, axis=({}))", output_var, input_vars[0],
                    axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "))
            }
        }

        // Slice with full parameters
        JaxOp::Slice { starts, ends, axes, steps } => {
            if !starts.is_empty() && !ends.is_empty() {
                // Build Python slice notation directly
                let axes_str = if axes.is_empty() {
                    (0..starts.len() as i64).collect::<Vec<_>>()
                } else {
                    axes.clone()
                };
                let steps_resolved: Vec<i64> = if steps.is_empty() {
                    vec![1; starts.len()]
                } else {
                    steps.clone()
                };

                // Find max axis to know how many dimensions we need
                let max_axis = *axes_str.iter().max().unwrap_or(&0) as usize;
                let mut slices: Vec<String> = vec!["slice(None)".to_string(); max_axis + 1];
                
                for i in 0..starts.len() {
                    let ax = axes_str[i] as usize;
                    let end_str = if ends[i] >= i64::MAX / 2 { "None".to_string() } else { ends[i].to_string() };
                    let step = steps_resolved.get(i).copied().unwrap_or(1);
                    if step == 1 {
                        slices[ax] = format!("slice({}, {})", starts[i], end_str);
                    } else {
                        slices[ax] = format!("slice({}, {}, {})", starts[i], end_str, step);
                    }
                }
                format!("{} = {}[{}]", output_var, input_vars[0],
                    slices.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            } else {
                // Dynamic slicing from input tensors — emit a helper call
                format!("{} = jax.lax.dynamic_slice({}, {}, {})",
                    output_var, input_vars[0],
                    input_vars.get(1).unwrap_or(&"None".to_string()),
                    input_vars.get(3).unwrap_or(&"None".to_string()))
            }
        }

        JaxOp::Tile => format!("{} = jnp.tile({}, {})", output_var, input_vars[0], input_vars.get(1).unwrap_or(&"None".to_string())),
        JaxOp::Expand => format!("{} = jnp.broadcast_to({}, {})", output_var, input_vars[0], input_vars.get(1).unwrap_or(&"None".to_string())),

        // Pad with proper mode and constant value
        JaxOp::Pad { pads, mode, constant_value } => {
            if !pads.is_empty() {
                // ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                let ndim = pads.len() / 2;
                let pad_pairs: Vec<String> = (0..ndim)
                    .map(|i| format!("({}, {})", pads[i], pads[i + ndim]))
                    .collect();
                let pad_str = format!("[{}]", pad_pairs.join(", "));
                match mode.as_str() {
                    "constant" => format!("{} = jnp.pad({}, {}, mode='constant', constant_values={})", 
                        output_var, input_vars[0], pad_str, constant_value),
                    "reflect" => format!("{} = jnp.pad({}, {}, mode='reflect')", 
                        output_var, input_vars[0], pad_str),
                    "edge" => format!("{} = jnp.pad({}, {}, mode='edge')", 
                        output_var, input_vars[0], pad_str),
                    _ => format!("{} = jnp.pad({}, {}, mode='constant', constant_values={})", 
                        output_var, input_vars[0], pad_str, constant_value),
                }
            } else {
                // Pads come from input tensor (opset >= 11)
                format!("{} = jnp.pad({}, jnp.reshape({}, (-1, 2)).tolist(), mode='{}', constant_values={})",
                    output_var, input_vars[0],
                    input_vars.get(1).unwrap_or(&"None".to_string()),
                    mode, constant_value)
            }
        }

        // Cast with proper dtype
        JaxOp::Cast { to_dtype } => {
            let dtype_str = onnx_dtype_to_jnp(*to_dtype);
            format!("{} = {}.astype({})", output_var, input_vars[0], dtype_str)
        }

        // Metadata / Utility
        JaxOp::Shape => format!("{} = jnp.array({}.shape)", output_var, input_vars[0]),
        JaxOp::Identity => format!("{} = {}", output_var, input_vars[0]),

        // Constant with actual value
        JaxOp::Constant { value } => {
            // Format value cleanly: avoid scientific notation for simple values
            if *value == 0.0 {
                format!("{} = 0.0", output_var)
            } else if *value == 1.0 {
                format!("{} = 1.0", output_var)
            } else if *value == -1.0 {
                format!("{} = -1.0", output_var)
            } else if value.fract() == 0.0 && value.abs() < 1e15 {
                format!("{} = {:.1}", output_var, value)
            } else {
                format!("{} = {}", output_var, value)
            }
        }

        JaxOp::DynamicQuantizeLinear => format!("{} = {}", output_var, input_vars[0]),
        JaxOp::Clip => format!("{} = jnp.clip({}, {}, {})", output_var, input_vars[0], input_vars.get(1).unwrap_or(&"None".to_string()), input_vars.get(2).unwrap_or(&"None".to_string())),
        JaxOp::Resize => format!("{} = jax.image.resize({}, {}.shape, method='bilinear')", output_var, input_vars[0], input_vars[0]),

        JaxOp::DepthToSpace => format!("{} = jnp.transpose(jnp.reshape({}, (-1)), (0, 1, 2, 3)) # DepthToSpace stub", output_var, input_vars[0]),
        JaxOp::Unknown(op) => format!("{} = {} # Implementation for '{}' pending", output_var, input_vars.get(0).unwrap_or(&"None".to_string()), op),
    };

    Ok(code)
}

/// Validate that a node has the minimum number of input variables.
fn ensure_inputs(op_name: &str, input_vars: &[String], min_count: usize) -> Result<()> {
    if input_vars.len() < min_count {
        bail!("{} requires at least {} inputs, but got {}", op_name, min_count, input_vars.len());
    }
    Ok(())
}
