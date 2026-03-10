# OxideXLA -- Real World ResNet-18 Validation
#
# This script:
# 1. Downloads / Exports a real ResNet-18 model
# 2. Runs inference on a sample image (cat) using PyTorch
# 3. Transpiles the model to JAX using OxT
# 4. Runs inference on the same image using the generated JAX code
# 5. Verifies top-1 class parity

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import os
import sys
import importlib.util
from PIL import Image
import requests
from io import BytesIO

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

def get_cat_image():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    print(f"Downloading sample image from {url}...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def main():
    # 1. Export ResNet-18
    print("--- Exporting ResNet-18 ---")
    model = models.resnet18(pretrained=True)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "resnet18.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=14)

    # 2. Transpile
    jax_py = "resnet18_jax.py"
    run_transpiler(onnx_path, jax_py)

    # 3. Preprocess Image
    img = get_cat_image()
    input_tensor = preprocess(img)
    
    # 4. PyTorch Inference
    print("--- Running PyTorch Inference ---")
    with torch.no_grad():
        torch_out = model(input_tensor)
        torch_prob = torch.nn.functional.softmax(torch_out, dim=1)
        top1_prob, top1_catid = torch.topk(torch_prob, 1)
        print(f"PyTorch Top-1 ID: {top1_catid[0][0].item()}, Prob: {top1_prob[0][0].item():.4f}")

    # 5. JAX Inference
    print("--- Running JAX Inference ---")
    jax_forward = load_jax_function(jax_py)
    
    # Map weights robustly from ONNX initializers since torch.onnx.export 
    # fuses BatchNorm into Conv layers, meaning we cannot use the raw
    # PyTorch state_dict directly for JAX execution.
    import onnx
    from onnx import numpy_helper
    onnx_m = onnx.load(onnx_path)
    
    params = {}
    for init in onnx_m.graph.initializer:
        params[init.name] = jnp.array(numpy_helper.to_array(init))
    
    jax_input = jnp.array(input_tensor.numpy())
    jit_forward = jax.jit(jax_forward)
    
    jax_out = jit_forward(params, jax_input)
    jax_prob = jax.nn.softmax(jax_out, axis=1)
    
    top1_idx = jnp.argmax(jax_prob, axis=1)[0]
    top1_p = jnp.max(jax_prob, axis=1)[0]
    
    print(f"JAX Top-1 ID: {top1_idx}, Prob: {top1_p:.4f}")

    # 6. Verification
    mse = np.mean((torch_out.numpy() - np.array(jax_out))**2)
    print(f"--- Verification ---")
    print(f"MSE: {mse:.2e}")
    if top1_catid[0][0].item() == int(top1_idx):
        print("Status: SUCCESS - Class match verified")
    else:
        print("Status: FAILED - Class mismatch")

if __name__ == "__main__":
    main()
