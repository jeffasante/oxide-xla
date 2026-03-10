# Contributing to OxideXLA

We are thrilled that you're interested in contributing to OxideXLA! Our goal is to create the fastest, most reliable bridge between PyTorch (via ONNX) and pure JAX. 

We welcome contributions of all kinds: adding new operator mappings, improving shape inference, optimizing compilation speed, and expanding our test suites.

## Project Philosophy

1. **Zero Runtime Overhead**: OxideXLA is an Ahead-Of-Time (AOT) transpiler. The output must be pure JAX. We do not inject runtime libraries or stateful wrappers.
2. **Stateless Generation**: Generated code must strictly separate parameters from logic (`forward(params, inputs)`).
3. **Correctness over Features**: An operator should only be supported if it maps with 100% numerical fidelity (MSE < 10^-10) relative to its PyTorch source.

---

## Getting Started

### Prerequisites

To build and test OxideXLA, you'll need:
- **Rust** (stable, 1.70+) - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python** (3.9+)
- **JAX** - `pip install -U "jax[cpu]"`
- **PyTorch & ONNX** - `pip install torch torchvision onnx`

### Local Setup

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/jeffasante/oxide-xla.git
   cd oxide-xla
   ```
2. Build the Rust core:
   ```bash
   cargo build
   ```
3. Run the Rust unit tests:
   ```bash
   cargo test
   ```

---

## How to Contribute

### 1. Adding a New Operator Mapping

The most common contribution is adding support for a new ONNX operator. 

**Workflow:**
1. Check the ONNX specification for the operator you want to map.
2. Locate the routing table in `src/ops/mod.rs` and add an entry.
3. If it's a Neural Network op, add the mapping logic to `src/ops/nn.rs`. If it's Math, use `src/ops/math.rs`. If it modifies Shapes, use `src/ops/reshape.rs`.
4. Add the code generation template for your new operator in `src/codegen/emit.rs`.

**Example:**
If mapping `Sigmoid`, you would add:
- `JaxOp::Sigmoid` to the `JaxOp` enum in `src/graph/dag.rs`.
- The matching logic in `src/ops/nn.rs`.
- The string generation `"{} = jax.nn.sigmoid({})"` in `src/codegen/emit.rs`.

### 2. Testing Your Operator

We maintain strict numerical parity checks in Python to ensure translations are correct.

1. Write a minimal PyTorch test case in Python (e.g., using `torch.nn.Module`).
2. Export it via `torch.onnx.export`.
3. Transpile it with OxideXLA.
4. Add an integration test in the `tests/` directory ensuring that the output of the JAX function exactly matches the output of the PyTorch function given the same input array and weights.

### 3. Improving the Web Playground

If you're contributing to the WebGL playground (`web/`):
- All visual interactions occur entirely client-side.
- Do not introduce heavy frontend frameworks. Keep UI logic as Vanilla JS or lightweight modules to ensure high performance on large ONNX topologies.

---

## Pull Request Process

1. **Fork the repo** and create your branch from `master`.
2. **Format your code**: We enforce `rustfmt`. Run `cargo fmt` before committing.
3. **Pass the CI Pipeline**: Your PR must pass all checks:
   - `cargo test` (All Rust suite tests pass)
   - `cargo build --release` (Compile without warnings)
   - Python parity tests must yield an `MSE < 10^-10`.
4. **Update Documentation**: If you've mapped a new operator, make sure to add it to the table in `docs/operator_mapping.md`.
5. **Write Clear PR Descriptions**: Explain *why* you are making the change, and provide details on how it was tested.

## Code Review Guidelines

Reviewers will be looking for:
- **Immutability**: Does the generated JAX code maintain a pure, stateless functional layout?
- **Shape Logic**: Does the shape inference system (`src/graph/shape.rs`) correctly calculate dimensions before emitting code?
- **Readability**: Are complex mathematical decompositions well-commented? (e.g., BatchNormalization decomposition).

## Getting Help

If you're stuck, feel free to open a Draft PR or an Issue tagged with `question`. Provide the PyTorch model block that is failing, and the OxideXLA team or community will help diagnose the architectural mismatch.

Thank you for helping us build the fastest graph translation engine!
