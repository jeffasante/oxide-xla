import torch
from transformers import AutoFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import os
import sys
import importlib.util
import onnx
from onnx import numpy_helper

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
    print("--- Defining and Exporting DeiT ---")
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Load model and feature extractor
    model_name = 'facebook/deit-small-patch16-224'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    input_tensor = inputs['pixel_values']
    
    onnx_path = "deit.onnx"
    # Export the model
    print("Exporting model to ONNX...")
    torch.onnx.export(model, input_tensor, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=14)

    # Check ops
    m = onnx.load(onnx_path)
    ops = set([n.op_type for n in m.graph.node])
    print(f"ONNX ops used in DeiT: {ops}")

    # Transpile
    jax_py = "deit_jax.py"
    run_transpiler(onnx_path, jax_py)

    print("--- Running PyTorch Inference ---")
    with torch.no_grad():
        torch_out = model(**inputs).logits
        torch_prob = torch.nn.functional.softmax(torch_out, dim=1)
        top1_prob, top1_id = torch.topk(torch_prob, 1)
        print(f"PyTorch Predicted ID: {top1_id[0][0].item()}, Prob: {top1_prob[0][0].item():.4f}")

    print("--- Running JAX Inference ---")
    jax_forward = load_jax_function(jax_py)
    
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
    
    print(f"JAX Predicted ID: {top1_idx}, Prob: {top1_p:.4f}")

    print("--- Verification ---")
    mse = np.mean((torch_out.numpy() - np.array(jax_out))**2)
    print(f"MSE: {mse:.2e}")
    if top1_id[0][0].item() == int(top1_idx):
        print("Status: SUCCESS - Class match verified")
    else:
        print("Status: FAILED - Class mismatch")

if __name__ == "__main__":
    main()
