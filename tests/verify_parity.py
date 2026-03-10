# OxideXLA -- Numerical Parity & Performance Validator
#
# This script performs an end-to-end comparison between PyTorch and JAX:
# 1. Runs inference in PyTorch
# 2. Exports the model to ONNX
# 3. Transpiles the ONNX model to JAX using oxide_xla
# 4. Runs inference in JAX (XLA compiled)
# 5. Compares numerical parity (MSE) and latency (speedup)

import torch
import torch.nn as nn
import onnx
import jax
import jax.numpy as jnp
import numpy as np
import time
import subprocess
import os
import sys
import importlib.util

def run_transpiler(onnx_path, output_py):
    """Call the oxide_xla binary to generate JAX code."""
    # Build if needed, or assume target/debug/oxide_xla exists.
    # We'll use cargo run --quiet -- compile as a stable way.
    cmd = ["cargo", "run", "--quiet", "--", "compile", onnx_path, "--output", output_py]
    # Ensure CWD is project root.
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"Transpilation failed for {onnx_path}")
        print(result.stderr)
        sys.exit(1)

def load_jax_function(py_path):
    """Load the generated forward(params, input) from a .py file."""
    spec = importlib.util.spec_from_file_location("jax_model", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward

def validate_model(name, torch_init_fn, input_shape):
    print(f"\n--- Validating {name} ---")
    
    # 1. Setup PyTorch model and weights
    torch_model = torch_init_fn()
    torch_model.eval()
    dummy_input = torch.randn(input_shape)
    
    # 2. Export to ONNX
    onnx_path = f"/tmp/{name}.onnx"
    torch.onnx.export(torch_model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=14)
    
    # 3. Transpile to JAX
    jax_py = f"/tmp/{name}_jax.py"
    run_transpiler(onnx_path, jax_py)
    
    # 4. Load JAX function
    jax_forward = load_jax_function(jax_py)
    
    # 5. Prepare JAX inputs and params
    # OxideXLA outputs params names based on ONNX names, 
    # so we need to map torch state_dict to the names used in ONNX.
    # Note: For simple models, ONNX names match or are predictable.
    # Our generated code uses keys like 'layer0.weight' etc.
    # We'll manually build a compatible params dict for now based on state_dict.
    params = {}
    for k, v in torch_model.state_dict().items():
        params[k] = jnp.array(v.numpy())
    
    jax_input = jnp.array(dummy_input.numpy())
    
    # 6. Numerical Comparison
    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()
    
    # JIT the jax function
    jit_jax_forward = jax.jit(jax_forward)
    
    # Warmup and Parity
    try:
        jax_out = jit_jax_forward(params, jax_input)
        mse = np.mean((torch_out - jax_out)**2)
        parity = "✓" if mse < 1e-6 else "✗"
    except Exception as e:
        print(f"JAX execution failed: {e}")
        return { "name": name, "mse": "ERROR", "speedup": "0.00x", "parity": "✗" }

    # 7. Latency Benchmark (Steady State)
    # PyTorch warmup
    for _ in range(5):
        with torch.no_grad(): _ = torch_model(dummy_input)
    
    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        with torch.no_grad(): _ = torch_model(dummy_input)
    torch_time = (time.perf_counter() - start) / iters

    # JAX warmup
    _ = jit_jax_forward(params, jax_input).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = jit_jax_forward(params, jax_input).block_until_ready()
    jax_time = (time.perf_counter() - start) / iters
    
    speedup = torch_time / jax_time
    
    print(f"  MSE:      {mse:.2e} {parity}")
    print(f"  Speedup:  {speedup:.2f}x")
    
    return {
        "name": name,
        "shape": str(input_shape),
        "torch_sum": f"{np.sum(torch_out):.5f}",
        "jax_sum": f"{np.sum(jax_out):.5f}",
        "mse": f"{mse:.2e}",
        "speedup": f"{speedup:.2f}x",
        "parity": parity
    }

# Model Factories
class MLP3Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified for ONNX export compatibility and our current op set
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 16 * 16, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        return self.fc(x)

class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 16 * 16, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

if __name__ == "__main__":
    results = []
    
    # 1. Run Tests
    results.append(validate_model("MLP-3-Layer", lambda: MLP3Layer(), (1, 64)))
    results.append(validate_model("CNN-BN-ReLU", lambda: CNNWithBN(), (1, 3, 16, 16)))

    # 2. Print Markdown Table
    print("\n\n### Validation Results Table")
    print("| Model | Shape | Torch Sum | JAX Sum | MSE | Speedup | Status |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['shape']} | {r['torch_sum']} | {r['jax_sum']} | {r['mse']} | {r['speedup']} | {r['parity']} |")
