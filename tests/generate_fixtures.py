"""
Generate small ONNX model fixtures for testing OxideXLA.

Run this script once to produce .onnx files in tests/models/.
These files are checked into the repo so the Rust tests can
load them without needing Python or PyTorch at test time.

Usage:
    python tests/generate_fixtures.py
"""

import os
import numpy as np

# We use the onnx library directly (no torch dependency) so that
# fixture generation is lightweight and reproducible.
try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Install onnx: pip install onnx numpy")
    raise


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_linear_model():
    """
    A single linear layer: output = MatMul(input, weight) + bias

    This is the MVP test case -- the simplest possible model that
    exercises MatMul, Add, and parameter handling.

    Input shape:  [1, 4]
    Weight shape: [4, 3]
    Bias shape:   [3]
    Output shape: [1, 3]
    """
    # Define graph inputs
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    # Create weight and bias initializers with fixed values for reproducibility
    rng = np.random.RandomState(42)
    weight_data = rng.randn(4, 3).astype(np.float32)
    bias_data = rng.randn(3).astype(np.float32)

    weight_init = numpy_helper.from_array(weight_data, name="linear0.weight")
    bias_init = numpy_helper.from_array(bias_data, name="linear0.bias")

    # MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "linear0.weight"],
        outputs=["matmul_out"],
        name="matmul_0",
    )

    # Add node (bias)
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_out", "linear0.bias"],
        outputs=["output"],
        name="add_0",
    )

    # Build graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        "linear_graph",
        [X],
        [Y],
        initializer=[weight_init, bias_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    path = os.path.join(OUTPUT_DIR, "linear.onnx")
    onnx.save(model, path)
    print(f"  Created: {path}")
    return path


def make_relu_model():
    """
    Linear + ReLU: output = ReLU(MatMul(input, weight) + bias)

    Exercises activation function mapping.

    Input shape:  [1, 4]
    Output shape: [1, 3]
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    rng = np.random.RandomState(42)
    weight_data = rng.randn(4, 3).astype(np.float32)
    bias_data = rng.randn(3).astype(np.float32)

    weight_init = numpy_helper.from_array(weight_data, name="layer0.weight")
    bias_init = numpy_helper.from_array(bias_data, name="layer0.bias")

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "layer0.weight"],
        outputs=["matmul_out"],
        name="matmul_0",
    )

    add_node = helper.make_node(
        "Add",
        inputs=["matmul_out", "layer0.bias"],
        outputs=["pre_relu"],
        name="add_0",
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["pre_relu"],
        outputs=["output"],
        name="relu_0",
    )

    graph = helper.make_graph(
        [matmul_node, add_node, relu_node],
        "relu_graph",
        [X],
        [Y],
        initializer=[weight_init, bias_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    path = os.path.join(OUTPUT_DIR, "linear_relu.onnx")
    onnx.save(model, path)
    print(f"  Created: {path}")
    return path


def make_two_layer_model():
    """
    Two-layer MLP: Linear -> ReLU -> Linear -> Softmax

    Exercises multiple parameter sets and the full forward pipeline.

    Input shape:  [1, 4]
    Hidden:       [1, 8]
    Output shape: [1, 3]
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    rng = np.random.RandomState(42)

    # Layer 0: 4 -> 8
    w0 = numpy_helper.from_array(rng.randn(4, 8).astype(np.float32), name="layer0.weight")
    b0 = numpy_helper.from_array(rng.randn(8).astype(np.float32), name="layer0.bias")

    # Layer 1: 8 -> 3
    w1 = numpy_helper.from_array(rng.randn(8, 3).astype(np.float32), name="layer1.weight")
    b1 = numpy_helper.from_array(rng.randn(3).astype(np.float32), name="layer1.bias")

    nodes = [
        helper.make_node("MatMul", ["input", "layer0.weight"], ["mm0"], name="matmul_0"),
        helper.make_node("Add", ["mm0", "layer0.bias"], ["add0"], name="add_0"),
        helper.make_node("Relu", ["add0"], ["relu0"], name="relu_0"),
        helper.make_node("MatMul", ["relu0", "layer1.weight"], ["mm1"], name="matmul_1"),
        helper.make_node("Add", ["mm1", "layer1.bias"], ["add1"], name="add_1"),
        helper.make_node(
            "Softmax", ["add1"], ["output"], name="softmax_0", axis=-1
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "two_layer_mlp",
        [X],
        [Y],
        initializer=[w0, b0, w1, b1],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    path = os.path.join(OUTPUT_DIR, "two_layer_mlp.onnx")
    onnx.save(model, path)
    print(f"  Created: {path}")
    return path


if __name__ == "__main__":
    print("Generating ONNX test fixtures:")
    make_linear_model()
    make_relu_model()
    make_two_layer_model()
    print("Done.")
