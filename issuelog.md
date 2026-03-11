# Issue Log: Transformer-class operator support

## Problem Description
Before this fix, the OxideXLA transpiler could not generate runnable JAX code for complex models like Vision Transformers or modern CNNs. The transpiler was incomplete and produced invalid Python code.

Specifically, there were three main issues:

1. Missing implementation for complex operations
Operations such as Slice, Split, Pad, Squeeze, Unsqueeze, and Cast were only stubbed. The code generator knew the operations existed in the ONNX graph but did not extract their required parameters (like padding sizes, slice start/end bounds, or target data types). As a result, the transpiler just outputted comments like `# Implementation for 'Slice' pending`.

2. Zeroed constants
Constant nodes in the ONNX graph, which hold critical values like layer normalization epsilon values, were not being parsed correctly. The tool did not know how to extract scalar values out of the ONNX TensorProto format. As a fallback, it hardcoded `0.0` for all constants. This guaranteed math errors in the generated model.

3. Invalid Python variable names
ONNX node names often use characters like slashes, dots, and hyphens (example: `/model/network.7/token_mixer/Split`). The transpiler used these exact ONNX names as Python variable names. This produced Python files that crashed immediately with SyntaxError.

## Resolution
The codebase was updated to fully support these operations.

1. Parameter mapping and structural changes
The internal graph format (JaxOp enum in `dag.rs`) was upgraded to hold all necessary operation parameters. Mapping functions were added to `reshape.rs` to correctly extract attributes like axes, steps, and target types from the ONNX graph. 

2. Tensor constant parsing
The ONNX attribute parser (`onnx_loader.rs`) was expanded to read embedded TensorProto data. `map_constant` was rewritten to interpret the raw bytes into proper float or integer values, eliminating the `0.0` fallback.

3. Complete code emission
`emit.rs` was rewritten to turn the structural data into the correct JAX NumPy function calls. For example, slicing now produces proper Python array indexing (e.g., `x[slice(0, 5, 2)]`), Pad uses `jnp.pad`, and Split uses `jnp.split`.

4. Shape inference
`shape.rs` was updated to accurately calculate how these new operations change the tensor dimensions.

5. Variable name sanitization
Added a function in `module.rs` that strips out slashes, dots, spaces, and hyphens to ensure all generated variable names are clean and valid Python identifiers. All generated Python code is now syntactically complete.
