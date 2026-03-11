// OxideXLA -- Integration Tests: IR Graph Construction
//
// These tests verify that OnnxModel -> IrGraph conversion works
// correctly: topological ordering, node counts, and shape inference.

use std::path::PathBuf;
use oxide_xla::parser::load_onnx_model;
use oxide_xla::graph::{IrGraph, JaxOp};

fn fixture_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("models");
    path.push(name);
    path
}

#[test]
fn build_ir_from_linear_model() {
    let onnx = load_onnx_model(&fixture_path("linear.onnx")).unwrap();
    let ir = IrGraph::from_onnx(&onnx).unwrap();

    // Total nodes in the graph: 1 input + 2 params + 2 ops = 5.
    assert_eq!(ir.node_count(), 5);

    // Total nodes in the graph: same value.
    assert_eq!(ir.graph.node_count(), 5);

    // Topological order should have all 5 nodes.
    assert_eq!(ir.topo_order().len(), 5);
}

#[test]
fn build_ir_from_two_layer_mlp() {
    let onnx = load_onnx_model(&fixture_path("two_layer_mlp.onnx")).unwrap();
    let ir = IrGraph::from_onnx(&onnx).unwrap();

    // Total nodes: 1 input + 4 params + 6 ops = 11.
    assert_eq!(ir.node_count(), 11);

    // Same value from the graph.
    assert_eq!(ir.graph.node_count(), 11);
}

#[test]
fn shape_inference_linear_model() {
    let onnx = load_onnx_model(&fixture_path("linear.onnx")).unwrap();
    let mut ir = IrGraph::from_onnx(&onnx).unwrap();
    oxide_xla::graph::infer_shapes(&mut ir).unwrap();

    // Find the last operation node (the Add that produces the output).
    // Its shape should be [1, 3].
    let order = ir.topo_order();
    let last_op_idx = order
        .iter()
        .rev()
        .find(|&&idx| matches!(ir.graph[idx].op, JaxOp::Add))
        .expect("Should find an Add node");

    let add_node = &ir.graph[*last_op_idx];
    assert_eq!(add_node.output_shape, vec![1, 3]);
}

#[test]
fn ascii_output_not_empty() {
    let onnx = load_onnx_model(&fixture_path("linear.onnx")).unwrap();
    let ir = IrGraph::from_onnx(&onnx).unwrap();

    let ascii = ir.to_ascii();
    assert!(!ascii.is_empty());
    assert!(ascii.contains("OxideXLA"));
    assert!(ascii.contains("MatMul"));
}

#[test]
fn json_output_is_valid() {
    let onnx = load_onnx_model(&fixture_path("linear.onnx")).unwrap();
    let ir = IrGraph::from_onnx(&onnx).unwrap();

    let json_str = ir.to_json().unwrap();

    // Should be valid JSON.
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("Output should be valid JSON");

    // Should be an object with nodes and edges.
    assert!(parsed.is_object());
    assert!(parsed.get("nodes").is_some());
}
