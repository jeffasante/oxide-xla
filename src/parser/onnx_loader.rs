// OxideXLA -- ONNX Model Loader
//
// This module reads an ONNX protobuf file and converts it into
// a simplified in-memory representation (OnnxModel) that the rest
// of the pipeline can work with.
//
// The ONNX format stores models as protobuf messages. The top-level
// message is ModelProto, which contains a GraphProto, which contains
// lists of NodeProto (operations), TensorProto (weights/initializers),
// and ValueInfoProto (input/output type information).
//
// This loader does NOT use a generated protobuf binding. Instead it
// reads the raw bytes using prost and extracts the fields manually.
// This keeps the dependency tree small and makes the parsing logic
// explicit rather than hidden behind generated code.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A simplified representation of an ONNX model.
/// Contains the graph nodes, initializer tensors (weights), and
/// information about the model's inputs and outputs.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// Human-readable name from the ONNX model metadata, if present.
    pub name: String,
    /// The ordered list of computation nodes.
    pub nodes: Vec<OnnxNode>,
    /// Initializer tensors (weights, biases, constants).
    /// Keyed by tensor name.
    pub initializers: HashMap<String, OnnxTensor>,
    /// Names and shapes of the model inputs (excluding initializers).
    pub inputs: Vec<(String, Vec<i64>)>,
    /// Names and shapes of the model outputs.
    pub outputs: Vec<(String, Vec<i64>)>,
}

/// A single ONNX computation node (one operator invocation).
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// The ONNX operator type string, e.g. "MatMul", "Relu", "Conv".
    pub op_type: String,
    /// Unique name of this node, if the model provides one.
    pub name: String,
    /// Names of the input tensors (data or initializer).
    pub inputs: Vec<String>,
    /// Names of the output tensors produced by this node.
    pub outputs: Vec<String>,
    /// Operator attributes (kernel size, strides, axis, etc.).
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// A tensor stored in the model file (weights, biases, or constants).
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// The name used to reference this tensor.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<i64>,
    /// Element data type (using ONNX type codes).
    /// 1 = float32, 7 = int64, 11 = float64, etc.
    pub data_type: i32,
    /// Raw bytes of the tensor data.
    pub raw_data: Vec<u8>,
    /// Explicit int64 values (optional, used in some ONNX versions instead of raw_data).
    pub int64_data: Vec<i64>,
}

/// An attribute attached to an ONNX node.
/// Attributes are compile-time constants (not tensor data).
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Tensor { shape: Vec<i64>, data_type: i32, float_data: Vec<f32>, raw_data: Vec<u8>, int64_data: Vec<i64> },
}

// ---------------------------------------------------------------------------
// ONNX protobuf field tags
// ---------------------------------------------------------------------------
//
// These are the protobuf field numbers defined in onnx.proto3.
// We decode them manually with prost rather than using generated code.
//
// See: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
//
// ModelProto:
//   field 7 = GraphProto graph
//
// GraphProto:
//   field 1 = string name
//   field 5 = TensorProto[] initializer
//   field 11 = ValueInfoProto[] input
//   field 12 = ValueInfoProto[] output
//   field 1 = NodeProto[] node  (actually field tag is... see below)
//
// We will read these with a streaming protobuf decoder.

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load an ONNX model from a file path.
///
/// This reads the entire file into memory and then parses the protobuf
/// structure. For very large models (multi-GB), a streaming approach
/// would be needed, but for typical models this is fine.
pub fn load_onnx_model(path: &Path) -> Result<OnnxModel> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read ONNX file: {}", path.display()))?;

    load_onnx_from_bytes(&bytes)
        .with_context(|| format!("Failed to parse ONNX model from: {}", path.display()))
}

/// Parse raw ONNX protobuf bytes into an OnnxModel.
///
/// The ONNX protobuf schema nests data as:
///   ModelProto -> GraphProto -> NodeProto[], TensorProto[], ValueInfoProto[]
///
/// We use prost::bytes to do low-level protobuf decoding because it avoids
/// the need for a full .proto compilation step and keeps control explicit.
pub fn load_onnx_from_bytes(data: &[u8]) -> Result<OnnxModel> {
    use prost::Message;

    // Decode the top-level ModelProto.
    // We define a minimal protobuf struct inline. In a production version
    // this would come from a generated .proto file, but for clarity we
    // keep it self-contained here.
    let model_proto = onnx_proto::ModelProto::decode(data)
        .context("Protobuf decode failed -- is this a valid ONNX file?")?;

    let graph = model_proto
        .graph
        .context("ONNX model has no graph field")?;

    // -- Parse nodes --
    let nodes: Vec<OnnxNode> = graph
        .node
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            let attributes = n
                .attribute
                .into_iter()
                .filter_map(|attr| {
                    let name = attr.name.clone();
                    let value = convert_attribute(&attr);
                    value.map(|v| (name, v))
                })
                .collect();

            OnnxNode {
                op_type: n.op_type,
                name: if n.name.is_empty() {
                    format!("node_{}", i)
                } else {
                    n.name
                },
                inputs: n.input,
                outputs: n.output,
                attributes,
            }
        })
        .collect();

    // -- Parse initializers (weights) --
    let initializers: HashMap<String, OnnxTensor> = graph
        .initializer
        .into_iter()
        .map(|t| {
            let name = t.name.clone();
            let tensor = OnnxTensor {
                name: t.name,
                shape: t.dims,
                data_type: t.data_type,
                raw_data: t.raw_data,
                int64_data: t.int64_data,
            };
            (name, tensor)
        })
        .collect();

    // -- Parse inputs --
    let inputs: Vec<(String, Vec<i64>)> = graph
        .input
        .iter()
        .filter(|vi| !initializers.contains_key(&vi.name))
        .map(|vi| {
            let shape = extract_shape(vi);
            (vi.name.clone(), shape)
        })
        .collect();

    // -- Parse outputs --
    let outputs: Vec<(String, Vec<i64>)> = graph
        .output
        .iter()
        .map(|vi| {
            let shape = extract_shape(vi);
            (vi.name.clone(), shape)
        })
        .collect();

    Ok(OnnxModel {
        name: graph.name,
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

/// Extract shape dimensions from a ValueInfoProto.
fn extract_shape(vi: &onnx_proto::ValueInfoProto) -> Vec<i64> {
    vi.r#type
        .as_ref()
        .and_then(|tp| tp.tensor_type.as_ref())
        .and_then(|tt| tt.shape.as_ref())
        .map(|shape| {
            shape
                .dim
                .iter()
                .map(|d| d.dim_value)
                .collect()
        })
        .unwrap_or_default()
}

/// Convert an ONNX AttributeProto into our simplified OnnxAttribute.
fn convert_attribute(attr: &onnx_proto::AttributeProto) -> Option<OnnxAttribute> {
    // AttributeProto.type: 1=FLOAT, 2=INT, 3=STRING, 4=TENSOR, 6=FLOATS, 7=INTS
    match attr.r#type {
        1 => Some(OnnxAttribute::Float(attr.f)),
        2 => Some(OnnxAttribute::Int(attr.i)),
        3 => {
            let s = String::from_utf8_lossy(&attr.s).to_string();
            Some(OnnxAttribute::String(s))
        }
        4 => {
            // TENSOR type - extract the embedded TensorProto
            if let Some(ref t) = attr.t {
                Some(OnnxAttribute::Tensor {
                    shape: t.dims.clone(),
                    data_type: t.data_type,
                    float_data: t.float_data.clone(),
                    raw_data: t.raw_data.clone(),
                    int64_data: t.int64_data.clone(),
                })
            } else {
                None
            }
        }
        6 => Some(OnnxAttribute::Floats(attr.floats.clone())),
        7 => Some(OnnxAttribute::Ints(attr.ints.clone())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf definitions
// ---------------------------------------------------------------------------
//
// These are the minimum structs needed to decode an ONNX file.
// They mirror the official onnx.proto3 schema but only include
// the fields we actually read.
//
// In a production codebase you would generate these from onnx.proto3
// using prost-build. We define them manually here for two reasons:
//   1. No build.rs / .proto file dependency during initial development
//   2. Makes the structure explicit and easy to understand

pub mod onnx_proto {
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct ModelProto {
        #[prost(int64, tag = "1")]
        pub ir_version: i64,
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct GraphProto {
        #[prost(string, tag = "2")]
        pub name: String,
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<NodeProto>,
        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,
        #[prost(message, repeated, tag = "11")]
        pub input: Vec<ValueInfoProto>,
        #[prost(message, repeated, tag = "12")]
        pub output: Vec<ValueInfoProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct NodeProto {
        #[prost(string, repeated, tag = "1")]
        pub input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        pub output: Vec<String>,
        #[prost(string, tag = "3")]
        pub name: String,
        #[prost(string, tag = "4")]
        pub op_type: String,
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct TensorProto {
        #[prost(int64, repeated, packed = "false", tag = "1")]
        pub dims: Vec<i64>,
        #[prost(int32, tag = "2")]
        pub data_type: i32,
        #[prost(string, tag = "8")]
        pub name: String,
        #[prost(bytes = "vec", tag = "9")]
        pub raw_data: Vec<u8>,
        #[prost(float, repeated, packed = "false", tag = "4")]
        pub float_data: Vec<f32>,
        #[prost(int64, repeated, packed = "false", tag = "7")]
        pub int64_data: Vec<i64>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct ValueInfoProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, optional, tag = "2")]
        pub r#type: Option<TypeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct TypeProto {
        #[prost(message, optional, tag = "1")]
        pub tensor_type: Option<TensorTypeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct TensorTypeProto {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<TensorShapeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct TensorShapeProto {
        #[prost(message, repeated, tag = "1")]
        pub dim: Vec<Dimension>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Dimension {
        #[prost(int64, tag = "1")]
        pub dim_value: i64,
        #[prost(string, tag = "2")]
        pub dim_param: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct AttributeProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(float, tag = "2")]
        pub f: f32,
        #[prost(int64, tag = "3")]
        pub i: i64,
        #[prost(bytes = "vec", tag = "4")]
        pub s: Vec<u8>,
        #[prost(message, optional, tag = "5")]
        pub t: Option<TensorProto>,
        #[prost(int32, tag = "20")]
        pub r#type: i32,
        #[prost(float, repeated, packed = "false", tag = "7")]
        pub floats: Vec<f32>,
        #[prost(int64, repeated, packed = "false", tag = "8")]
        pub ints: Vec<i64>,
    }
}
