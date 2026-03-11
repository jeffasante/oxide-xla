// OxideXLA -- Shape Inference
//
// This module walks the IR graph in topological order and computes
// the output shape of each node based on its operation type and
// the shapes of its inputs.
//
// Shape inference is critical because JAX code must have statically
// known shapes to compile efficiently under XLA. If we emit code
// with incorrect shapes, jax.jit will fail at trace time.

use anyhow::{Result, bail};

use crate::graph::{IrGraph, JaxOp};

/// Run shape inference over the entire graph.
///
/// Iterates in topological order so that when we process a node,
/// all of its inputs already have their shapes computed.
pub fn infer_shapes(ir_graph: &mut IrGraph) -> Result<()> {
    // We need to clone the topo order to avoid borrowing conflicts
    // while mutating the graph nodes.
    let order: Vec<_> = ir_graph.topo_order().to_vec();

    for &node_idx in &order {
        // Use ordered_inputs to preserve the ONNX-specified operand order.
        let input_indices = ir_graph.graph[node_idx].ordered_inputs.clone();

        let input_shapes: Vec<Vec<i64>> = input_indices
            .iter()
            .map(|&idx| ir_graph.graph[idx].output_shape.clone())
            .collect();

        // Input and Param nodes already have their shapes set
        // during graph construction, so skip them.
        let op = ir_graph.graph[node_idx].op.clone();
        let inferred = match &op {
            JaxOp::Input | JaxOp::Param => continue,

            // Linear Algebra
            JaxOp::MatMul => infer_matmul(&input_shapes, false)?,
            JaxOp::Gemm { trans_a, trans_b, .. } => infer_gemm(&input_shapes, *trans_a, *trans_b)?,

            // Element-wise / Broadcast
            JaxOp::Add | JaxOp::Mul | JaxOp::Sub | JaxOp::Div | JaxOp::Pow |
            JaxOp::And | JaxOp::Or | JaxOp::Xor | JaxOp::Equal | JaxOp::Greater |
            JaxOp::GreaterOrEqual | JaxOp::Less | JaxOp::LessOrEqual | JaxOp::Where => {
                infer_broadcast(&input_shapes)?
            }

            // Element-wise (No broadcast)
            JaxOp::Relu | JaxOp::Sigmoid | JaxOp::Tanh | JaxOp::Erf | JaxOp::Abs |
            JaxOp::Acos | JaxOp::Acosh | JaxOp::Asin | JaxOp::Asinh | JaxOp::Atan |
            JaxOp::Atanh | JaxOp::Ceil | JaxOp::Cos | JaxOp::Cosh | JaxOp::Exp |
            JaxOp::Floor | JaxOp::Log | JaxOp::Neg | JaxOp::Reciprocal | JaxOp::Round |
            JaxOp::Sin | JaxOp::Sinh | JaxOp::Tan | JaxOp::Sign | JaxOp::Not |
            JaxOp::Elu { .. } | JaxOp::HardSigmoid { .. } | JaxOp::HardSwish |
            JaxOp::LeakyRelu { .. } | JaxOp::Selu { .. } | JaxOp::Softplus |
            JaxOp::Softsign | JaxOp::ThresholdedRelu { .. } | JaxOp::Gelu |
            JaxOp::Mish | JaxOp::PRelu | JaxOp::Clip | JaxOp::Identity |
            JaxOp::Sqrt | JaxOp::Cast | JaxOp::BatchNorm { .. } | JaxOp::LayerNormalization { .. } |
            JaxOp::Softmax { .. } | JaxOp::Squeeze | JaxOp::Unsqueeze => {
                infer_elementwise(&input_shapes)?
            }

            JaxOp::Conv { strides, pads, dilations, .. } => {
                infer_conv(&input_shapes, strides, pads, dilations)?
            }
            JaxOp::ConvTranspose { strides, pads, dilations, output_padding, .. } => {
                infer_conv_transpose(&input_shapes, strides, pads, dilations, output_padding)?
            }
            JaxOp::MaxPool { strides, kernel_shape, pads } => {
                infer_maxpool(&input_shapes, strides, kernel_shape, pads)?
            }

            // Shape manipulation
            JaxOp::Reshape { target_shape } => {
                if !target_shape.is_empty() {
                    target_shape.clone()
                } else if input_shapes.len() >= 2 {
                    // Try to use second input for shape if available
                    input_shapes[1].clone()
                } else {
                    vec![]
                }
            }
            JaxOp::Transpose { perm } => infer_transpose(&input_shapes, perm)?,
            JaxOp::Concat { axis } => infer_concat(&input_shapes, *axis)?,
            JaxOp::Gather { axis } => infer_gather(&input_shapes, *axis)?,
            JaxOp::Tile | JaxOp::Expand | JaxOp::Resize | JaxOp::DepthToSpace | JaxOp::Slice | JaxOp::Pad { .. } => {
                // Fallback: borrow input shape
                if input_shapes.is_empty() { Vec::new() } else { input_shapes[0].clone() }
            }

            // Reductions
            JaxOp::ReduceMean { axes, keepdims } |
            JaxOp::ReduceMax { axes, keepdims } |
            JaxOp::ReduceMin { axes, keepdims } |
            JaxOp::ReduceSum { axes, keepdims } |
            JaxOp::ReduceProd { axes, keepdims } => {
                infer_reduction(&input_shapes, axes, *keepdims)?
            }

            // Metadata
            JaxOp::Shape => {
                if input_shapes.is_empty() { Vec::new() } else { vec![input_shapes[0].len() as i64] }
            }

            JaxOp::Constant | JaxOp::DynamicQuantizeLinear | JaxOp::Unknown(_) => {
                if input_shapes.is_empty() { Vec::new() } else { input_shapes[0].clone() }
            }
        };

        ir_graph.graph[node_idx].output_shape = inferred;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Per-op shape rules
// ---------------------------------------------------------------------------

/// MatMul: [..., M, K] x [..., K, N] -> [..., M, N]
fn infer_matmul(inputs: &[Vec<i64>], _trans_b: bool) -> Result<Vec<i64>> {
    if inputs.len() < 2 {
        bail!("MatMul requires at least 2 inputs, got {}", inputs.len());
    }
    let a = &inputs[0];
    let b = &inputs[1];

    if a.is_empty() || b.is_empty() {
        return Ok(vec![]);
    }

    let mut result = a[..a.len() - 1].to_vec();
    result.push(*b.last().unwrap_or(&0));
    Ok(result)
}

/// Gemm: C = alpha*A'B' + beta*C
fn infer_gemm(inputs: &[Vec<i64>], trans_a: bool, trans_b: bool) -> Result<Vec<i64>> {
    if inputs.len() < 2 {
        bail!("Gemm requires at least 2 inputs");
    }
    let a = &inputs[0];
    let b = &inputs[1];
    
    if a.len() < 2 || b.len() < 2 {
        return Ok(vec![]);
    }

    let m = if trans_a { a[1] } else { a[0] };
    let n = if trans_b { b[0] } else { b[1] };
    
    Ok(vec![m, n])
}

/// Reduction: Axis/axes are collapsed.
fn infer_reduction(inputs: &[Vec<i64>], axes: &[i64], keepdims: bool) -> Result<Vec<i64>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let input = &inputs[0];
    if input.is_empty() {
        return Ok(vec![]);
    }

    let mut result = Vec::new();
    let axes_set: std::collections::HashSet<i64> = axes.iter().cloned().collect();

    for (i, &dim) in input.iter().enumerate() {
        let axis = i as i64;
        if axes_set.is_empty() {
            // Empty axes in ONNX often means reduce everything
            if keepdims { result.push(1); }
        } else if axes_set.contains(&axis) {
            if keepdims {
                result.push(1);
            }
        } else {
            result.push(dim);
        }
    }

    Ok(result)
}

/// Legacy ReduceMean wrapper
fn infer_reducemean(inputs: &[Vec<i64>], axes: &[i64], keepdims: bool) -> Result<Vec<i64>> {
    infer_reduction(inputs, axes, keepdims)
}

/// Elementwise: Shape remains identical.
fn infer_elementwise(inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    Ok(inputs[0].clone())
}

/// Broadcast: Resolve multidirectional broadcasting rules.
fn infer_broadcast(inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
    if inputs.len() < 2 {
        return inputs.get(0).cloned().ok_or(anyhow::anyhow!("No inputs for broadcast"));
    }

    let mut result = inputs[0].clone();
    for i in 1..inputs.len() {
        let other = &inputs[i];
        let max_rank = std::cmp::max(result.len(), other.len());
        let mut new_shape = vec![1; max_rank];

        for j in 1..=max_rank {
            let dim_a = if j <= result.len() { result[result.len() - j] } else { 1 };
            let dim_b = if j <= other.len() { other[other.len() - j] } else { 1 };

            if dim_a == dim_b {
                new_shape[max_rank - j] = dim_a;
            } else if dim_a == 1 {
                new_shape[max_rank - j] = dim_b;
            } else if dim_b == 1 {
                new_shape[max_rank - j] = dim_a;
            } else {
                // If dimensions don't match and neither is 1, let's just 
                // take the max as a fallback for dynamic shapes.
                new_shape[max_rank - j] = std::cmp::max(dim_a, dim_b);
            }
        }
        result = new_shape;
    }
    Ok(result)
}

/// Concat: One dimension grows, others must match.
fn infer_concat(inputs: &[Vec<i64>], axis: i64) -> Result<Vec<i64>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }

    let mut result = inputs[0].clone();
    let abs_axis = if axis < 0 { (result.len() as i64 + axis) as usize } else { axis as usize };

    for i in 1..inputs.len() {
        if abs_axis < result.len() && abs_axis < inputs[i].len() {
            result[abs_axis] += inputs[i][abs_axis];
        }
    }
    Ok(result)
}

/// Transpose: reorder dimensions.
fn infer_transpose(inputs: &[Vec<i64>], perm: &[i64]) -> Result<Vec<i64>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let input = &inputs[0];
    if perm.is_empty() {
        let mut rev = input.clone();
        rev.reverse();
        return Ok(rev);
    }

    let mut result = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        if (p as usize) < input.len() {
            result[i] = input[p as usize];
        }
    }
    Ok(result)
}

/// Gather: Replace axis dimension with index shape.
fn infer_gather(inputs: &[Vec<i64>], axis: i64) -> Result<Vec<i64>> {
    if inputs.len() < 2 {
        bail!("Gather requires 2 inputs");
    }
    let data = &inputs[0];
    let indices = &inputs[1];
    
    let abs_axis = if axis < 0 { (data.len() as i64 + axis) as usize } else { axis as usize };
    
    let mut result = Vec::new();
    for i in 0..abs_axis {
        result.push(data[i]);
    }
    for &d in indices {
        result.push(d);
    }
    for i in (abs_axis + 1)..data.len() {
        result.push(data[i]);
    }
    
    Ok(result)
}

/// Conv: simplified output size calculation.
fn infer_conv(inputs: &[Vec<i64>], strides: &[i64], pads: &[i64], _dilations: &[i64]) -> Result<Vec<i64>> {
    if inputs.len() < 2 {
        bail!("Conv requires at least 2 inputs");
    }
    let input = &inputs[0];   // [N, Ci, H, W]
    let weight = &inputs[1];  // [Co, Ci/group, kH, kW]
    
    if input.len() < 4 || weight.len() < 4 {
        return Ok(input.clone());
    }

    let mut output = vec![input[0], weight[0]]; // N, Co
    
    for i in 0..2 {
        let dim = i + 2;
        let p = if pads.len() >= 4 { pads[i] + pads[i+2] } else { 0 };
        let s = if strides.len() >= 2 { strides[i] } else { 1 };
        let k = weight[dim];
        
        let out_dim = (input[dim] + p - k) / s + 1;
        output.push(out_dim);
    }
    
    Ok(output)
}

/// ConvTranspose: output = (input - 1) * stride + output_padding + kernel_size - 2 * padding
fn infer_conv_transpose(inputs: &[Vec<i64>], strides: &[i64], pads: &[i64], _dilations: &[i64], output_padding: &[i64]) -> Result<Vec<i64>> {
    if inputs.len() < 2 {
        bail!("ConvTranspose requires at least 2 inputs");
    }
    let input = &inputs[0];   // [N, Ci, H, W]
    let weight = &inputs[1];  // [Ci, Co/group, kH, kW]
    
    if input.len() < 4 || weight.len() < 4 {
        return Ok(input.clone());
    }

    // ONNX ConvTranspose weight is [Ci, Co/group, kH, kW]
    let mut output = vec![input[0], weight[1]]; // N, Co
    
    for i in 0..2 {
        let dim = i + 2;
        let p = if pads.len() >= 4 { pads[i] + pads[i+2] } else { 0 };
        let s = if strides.len() >= 2 { strides[i] } else { 1 };
        let k = weight[dim];
        let op = if output_padding.len() >= 2 { output_padding[i] } else { 0 };
        
        let out_dim = s * (input[dim] - 1) + op + k - p;
        output.push(out_dim);
    }
    
    Ok(output)
}

/// MaxPool: similar to Conv but simpler.
fn infer_maxpool(inputs: &[Vec<i64>], strides: &[i64], kernel_shape: &[i64], pads: &[i64]) -> Result<Vec<i64>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let input = &inputs[0];
    if input.len() < 4 {
        return Ok(input.clone());
    }
    
    let mut output = vec![input[0], input[1]];
    for i in 0..2 {
        let dim = i + 2;
        let p = if pads.len() >= 4 { pads[i] + pads[i+2] } else { 0 };
        let s = if strides.len() >= 2 { strides[i] } else { 1 };
        let k = if kernel_shape.len() >= 2 { kernel_shape[i] } else { 1 };
        
        let out_dim = (input[dim] + p - k) / s + 1;
        output.push(out_dim);
    }
    
    Ok(output)
}
