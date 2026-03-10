import torch
import torch.nn as nn
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

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # We'll use a simple fully connected layer after pooling the sequence
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # -> (batch_size, seq_len, embed_dim)
        
        # Average pooling over the sequence length (axis 1)
        pooled = torch.mean(embedded, dim=1) # -> (batch_size, embed_dim)
        
        x = self.relu(self.fc1(pooled))
        logits = self.fc2(x)
        return logits

def main():
    print("--- Defining and Exporting Text Classifier ---")
    
    vocab_size = 5000
    embed_dim = 64
    num_classes = 2 # Positive / Negative
    
    model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)
    model.eval()
    
    # Simulate a batched sequence of token IDs
    # e.g., Batch of 1 sequence, length 15
    dummy_input = torch.randint(0, vocab_size, (1, 15))
    
    onnx_path = "text_classifier.onnx"
    
    # Export the model
    print("Exporting model to ONNX...")
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input_ids'], output_names=['logits'],
                      opset_version=14)

    # Check ops
    m = onnx.load(onnx_path)
    ops = set([n.op_type for n in m.graph.node])
    print(f"ONNX ops used in TextClassifier: {ops}")

    jax_py = "text_classifier_jax.py"
    run_transpiler(onnx_path, jax_py)

    print("--- Running PyTorch Inference ---")
    with torch.no_grad():
        torch_out = model(dummy_input)
        torch_prob = torch.nn.functional.softmax(torch_out, dim=1)
        top1_prob, top1_id = torch.topk(torch_prob, 1)
        print(f"PyTorch Predicted ID: {top1_id[0][0].item()}, Prob: {top1_prob[0][0].item():.4f}")

    print("--- Running JAX Inference ---")
    jax_forward = load_jax_function(jax_py)
    
    params = {}
    for init in m.graph.initializer:
        params[init.name] = jnp.array(numpy_helper.to_array(init))
    
    jax_input = jnp.array(dummy_input.numpy())
    
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
        sys.exit(1)

if __name__ == "__main__":
    main()
