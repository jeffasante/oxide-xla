// OxideXLA -- IR Graph (Directed Acyclic Graph)
//
// This module defines the intermediate representation used throughout
// the compiler. The ONNX model is first parsed into OnnxModel (parser),
// then converted into an IrGraph here. The IrGraph is what the op mapper
// and code generator work with.
//
// The graph is backed by petgraph::DiGraph. Each node holds an IrNode
// struct and each edge represents a data dependency (tensor flowing
// from producer to consumer).

use anyhow::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use serde::Serialize;
use serde_json;
use std::collections::HashMap;

use crate::parser::OnnxModel;
use crate::ops;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The set of JAX operations that OxideXLA can emit.
/// Each variant corresponds to one JAX function call in the generated code.
#[derive(Debug, Clone, Serialize)]
pub enum JaxOp {
    /// Input placeholder -- not a real op, represents a function argument.
    Input,
    /// Constant / parameter reference -- loaded from the params dict.
    Param,
    
    // Core Ops with custom attributes
    Softmax { axis: i64 },
    Concat { axis: i64 },
    Gather { axis: i64 },
    LayerNormalization { axis: i64, epsilon: f32 },
    Reshape { target_shape: Vec<i64> },
    Transpose { perm: Vec<i64> },
    Conv {
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
    },
    ConvTranspose {
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        output_padding: Vec<i64>,
    },
    BatchNorm { epsilon: f32 },
    MaxPool {
        strides: Vec<i64>,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
    },
    
    // Feature-specific Stubs
    DynamicQuantizeLinear,
    Resize,
    DepthToSpace,
    PRelu,
    Constant { value: f64 },
    Clip,
    Shape,
    Identity,
    Slice { starts: Vec<i64>, ends: Vec<i64>, axes: Vec<i64>, steps: Vec<i64> },
    Split { axis: i64, num_outputs: usize },
    Pow,
    Sqrt,
    Cast { to_dtype: i32 },
    Squeeze { axes: Vec<i64> },
    Unsqueeze { axes: Vec<i64> },
    Pad { pads: Vec<i64>, mode: String, constant_value: f64 },
    AveragePool { kernel_shape: Vec<i64>, strides: Vec<i64>, pads: Vec<i64> },

    // Math (Element-wise)
    Abs, Acos, Acosh, Add, Asin, Asinh, Atan, Atanh, Ceil, Cos, Cosh, Div, Exp, Floor, Log, Mul, Neg, Reciprocal, Round, Sin, Sinh, Sub, Tan, Tanh, Erf, Sign,

    // Logic
    And, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Not, Or, Xor, Where,

    // Reductions
    ReduceMax { axes: Vec<i64>, keepdims: bool },
    ReduceMin { axes: Vec<i64>, keepdims: bool },
    ReduceSum { axes: Vec<i64>, keepdims: bool },
    ReduceProd { axes: Vec<i64>, keepdims: bool },
    ReduceMean { axes: Vec<i64>, keepdims: bool },

    // Activations
    Elu { alpha: f32 },
    HardSigmoid { alpha: f32, beta: f32 },
    HardSwish,
    LeakyRelu { alpha: f32 },
    Selu { alpha: f32, gamma: f32 },
    Sigmoid,
    Softplus,
    Softsign,
    ThresholdedRelu { alpha: f32 },
    Gelu,
    Mish,
    Relu,

    // Linear Algebra
    MatMul,
    Gemm { alpha: f32, beta: f32, trans_a: bool, trans_b: bool },

    // Other
    Tile,
    Expand,

    /// Unmapped / Unknown Operator Fallback
    Unknown(String),
}

/// The data type of a tensor.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Unknown,
}

impl DType {
    /// Convert from ONNX data type code to DType.
    pub fn from_onnx(code: i32) -> Self {
        match code {
            1 => DType::Float32,
            7 => DType::Int64,
            6 => DType::Int32,
            11 => DType::Float64,
            _ => DType::Unknown,
        }
    }
}

/// A single node in the IR computation graph.
#[derive(Debug, Clone, Serialize)]
pub enum NodeType {
    Operator,
    Constant,
    Input,
}

#[derive(Debug, Clone, Serialize)]
pub struct IrNode {
    pub name: String,
    pub op: JaxOp,
    pub output_shape: Vec<i64>,
    pub dtype: DType,
    pub node_type: NodeType,
    /// Preserves the exact input order expected by the ONNX operator.
    #[serde(skip)]
    pub ordered_inputs: Vec<NodeIndex>,
}

/// The intermediate representation graph.
pub struct IrGraph {
    pub graph: DiGraph<IrNode, ()>,
    pub input_indices: Vec<NodeIndex>,
    pub output_indices: Vec<NodeIndex>,
}

impl IrGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            input_indices: Vec::new(),
            output_indices: Vec::new(),
        }
    }

    /// Build the IR graph from an ONNX model.
    pub fn from_onnx(model: &OnnxModel) -> Result<Self> {
        let mut ir_graph = Self::new();
        let mut name_to_index = HashMap::new();

        // 1. Add inputs
        for input in &model.inputs {
            let node = IrNode {
                name: input.0.clone(),
                op: JaxOp::Input,
                output_shape: input.1.clone(),
                dtype: DType::Float32, // Defaulting for simple inputs
                node_type: NodeType::Input,
                ordered_inputs: Vec::new(),
            };
            let idx = ir_graph.graph.add_node(node);
            name_to_index.insert(input.0.clone(), idx);
            ir_graph.input_indices.push(idx);
        }

        // 2. Add initializers (parameters)
        for (name, init) in &model.initializers {
            let node = IrNode {
                name: name.clone(),
                op: JaxOp::Param,
                output_shape: init.shape.clone(),
                dtype: DType::from_onnx(init.data_type),
                node_type: NodeType::Constant,
                ordered_inputs: Vec::new(),
            };
            let idx = ir_graph.graph.add_node(node);
            name_to_index.insert(name.clone(), idx);
        }

        // 3. Add operators and edges
        for node in &model.nodes {
            let op = ops::map_onnx_op(&node.op_type, &node.attributes)?;
            let ir_node = IrNode {
                name: node.outputs[0].clone(),
                op,
                output_shape: Vec::new(), // To be inferred
                dtype: DType::Float32,     // To be inferred
                node_type: NodeType::Operator,
                ordered_inputs: Vec::new(),
            };
            let idx = ir_graph.graph.add_node(ir_node);

            for input_name in &node.inputs {
                if let Some(&src_idx) = name_to_index.get(input_name) {
                    ir_graph.graph.add_edge(src_idx, idx, ());
                    ir_graph.graph[idx].ordered_inputs.push(src_idx);
                }
            }

            for output_name in &node.outputs {
                name_to_index.insert(output_name.clone(), idx);
            }
        }

        // 4. Identify outputs
        for output in &model.outputs {
            if let Some(&idx) = name_to_index.get(&output.0) {
                ir_graph.output_indices.push(idx);
            }
        }

        Ok(ir_graph)
    }

    pub fn topo_order(&self) -> Vec<NodeIndex> {
        toposort(&self.graph, None).unwrap_or_default()
    }

    /// Serialize the graph to a JSON format for the web frontend.
    pub fn to_json(&self) -> anyhow::Result<String> {
        #[derive(Serialize)]
        struct JsonEdge {
            source: usize,
            target: usize,
        }

        #[derive(Serialize)]
        struct JsonGraph {
            nodes: Vec<IrNode>,
            edges: Vec<JsonEdge>,
        }

        let mut nodes = Vec::new();
        // Since IrNode might not include its index, we collect them in order.
        for i in 0..self.graph.node_count() {
            nodes.push(self.graph[NodeIndex::new(i)].clone());
        }

        let mut edges = Vec::new();
        for edge in self.graph.edge_indices() {
            let (source, target) = self.graph.edge_endpoints(edge).unwrap();
            edges.push(JsonEdge {
                source: source.index(),
                target: target.index(),
            });
        }

        let json_graph = JsonGraph { nodes, edges };
        Ok(serde_json::to_string(&json_graph)?)
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn to_ascii(&self) -> String {
        let mut out = String::new();
        out.push_str("\n OxideXLA -- IR Topology\n");
        out.push_str(" ===========================\n\n");
        
        let mut order = self.topo_order();
        if order.is_empty() {
             // Fallback to insertion order if topo fails
             order = (0..self.graph.node_count()).map(NodeIndex::new).collect();
        }

        out.push_str(&format!(" {:<6} | {:<18} | {:<20}\n", "ID", "OPERATOR", "OUTBOUND TO"));
        out.push_str(&format!(" {:-<6}-|-{:-<18}-|-{:-<20}\n", "", "", ""));

        for idx in order {
            let node = &self.graph[idx];
            let op_str = match &node.op {
                JaxOp::Unknown(s) => format!("? {}", s),
                _ => {
                    let s = format!("{:?}", node.op);
                    if s.len() > 18 { format!("{}...", &s[..15]) } else { s }
                }
            };
            
            let outputs: Vec<String> = self.graph.neighbors(idx)
                .map(|neighbor| neighbor.index().to_string())
                .collect();
            
            let out_str = if outputs.is_empty() { 
                "(output)".to_string() 
            } else { 
                outputs.join(", ") 
            };

            out.push_str(&format!(" [{:>3}]  | {:<18} | {:<20}\n", idx.index(), op_str, out_str));
        }
        
        out.push_str("\n Ready for JAX generation.\n");
        out
    }
}
