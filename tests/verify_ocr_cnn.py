import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import os
import sys
import importlib.util

class SimpleOCR(nn.Module):
    """A standard CNN architecture often used for digit/character recognition (OCR)."""
    def __init__(self):
        super(SimpleOCR, self).__init__()
        # 1 channel (grayscale) image input, 32 features out
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # MaxPool2d(2) reduces size by half
        self.pool = nn.MaxPool2d(2, 2)
        # 28x28 image -> pool -> 14x14 -> pool -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def run_transpiler(onnx_path, output_py):
    print(f"--- Transpiling {onnx_path} to {output_py} ---")
    cmd = ["cargo", "run", "--quiet", "--", "compile", onnx_path, "--output", output_py]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"Transpilation failed:")
        print(result.stderr)
        sys.exit(1)
    print("Transpilation successful.")

def load_jax_function(py_path):
    spec = importlib.util.spec_from_file_location("jax_model", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward

def main():
    # 1. Export OCR-style CNN
    print("--- Defining and Exporting OCR CNN ---")
    model = SimpleOCR()
    model.eval()
    
    # Simulate a batch of 1 grayscale image, 28x28 pixels (MNIST style)
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = "ocr_cnn.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=14)

    # 2. Transpile using OxideXLA
    jax_py = "ocr_cnn_jax.py"
    run_transpiler(onnx_path, jax_py)

    # 3. PyTorch Inference
    print("--- Running PyTorch Inference ---")
    with torch.no_grad():
        torch_out = model(dummy_input)
        torch_prob = torch.nn.functional.softmax(torch_out, dim=1)
        top1_prob, top1_id = torch.topk(torch_prob, 1)
        print(f"PyTorch Predicted Character/Digit ID: {top1_id[0][0].item()}, Prob: {top1_prob[0][0].item():.4f}")

    # 4. JAX Inference
    print("--- Running JAX Inference ---")
    jax_forward = load_jax_function(jax_py)
    
    import onnx
    from onnx import numpy_helper
    onnx_m = onnx.load(onnx_path)
    
    params = {}
    for init in onnx_m.graph.initializer:
        params[init.name] = jnp.array(numpy_helper.to_array(init))
    
    # In some models, constants used by flatten/reshape might not be in initializers
    # but our Rust parser handles axes via read_int64_tensor directly from initializers
    
    jax_input = jnp.array(dummy_input.numpy())
    jit_forward = jax.jit(jax_forward)
    
    jax_out = jit_forward(params, jax_input)
    jax_prob = jax.nn.softmax(jax_out, axis=1)
    
    top1_idx = jnp.argmax(jax_prob, axis=1)[0]
    top1_p = jnp.max(jax_prob, axis=1)[0]
    
    print(f"JAX Predicted Character/Digit ID: {top1_idx}, Prob: {top1_p:.4f}")

    # 5. Verification
    mse = np.mean((torch_out.numpy() - np.array(jax_out))**2)
    print(f"--- Verification ---")
    print(f"MSE: {mse:.2e}")
    if top1_id[0][0].item() == int(top1_idx):
        print("Status: SUCCESS - Class match verified")
    else:
        print("Status: FAILED - Class mismatch")
        sys.exit(1)

if __name__ == "__main__":
    main()
