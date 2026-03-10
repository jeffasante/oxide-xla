"""
OxideXLA -- PyTorch FX to ONNX Bridge

This script traces a PyTorch nn.Module using torch.fx, exports it to
ONNX format, and optionally pipes the result directly into the OxideXLA
Rust compiler.

Usage:
    # Export to file
    python fx_export.py --model torchvision.models.resnet18 --output model.onnx

    # Pipe directly into OxideXLA
    python fx_export.py --model torchvision.models.resnet18 | oxide_xla compile - --output model.py

Requirements:
    pip install torch torchvision onnx
"""

import argparse
import sys
import importlib
import io

import torch
import torch.onnx


def resolve_model(model_path: str) -> torch.nn.Module:
    """
    Resolve a model from a dotted Python path.

    Examples:
        "torchvision.models.resnet18" -> torchvision.models.resnet18(pretrained=False)
        "torchvision.models.mobilenet_v2" -> torchvision.models.mobilenet_v2(pretrained=False)
    """
    parts = model_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Model path must be in format 'module.function', got: {model_path}"
        )

    module_path, func_name = parts
    module = importlib.import_module(module_path)
    model_fn = getattr(module, func_name)

    # Call the function to get the model instance.
    # We pass weights=None to avoid downloading pretrained weights
    # during testing. For real use, the caller can load weights separately.
    try:
        model = model_fn(weights=None)
    except TypeError:
        # Older torchvision versions use pretrained=False
        model = model_fn(pretrained=False)

    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    output_path: str = None,
) -> bytes:
    """
    Export a PyTorch model to ONNX format.

    If output_path is provided, writes to that file.
    Always returns the ONNX bytes.
    """
    dummy_input = torch.randn(*input_shape)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Static shapes for now
    )

    onnx_bytes = buffer.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(onnx_bytes)
        print(f"Exported ONNX model to: {output_path}", file=sys.stderr)

    return onnx_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch model to ONNX for OxideXLA."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Dotted path to a model constructor (e.g., torchvision.models.resnet18)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .onnx file path. If omitted, writes to stdout for piping.",
    )
    parser.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Comma-separated input tensor shape (default: 1,3,224,224)",
    )

    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(int(d) for d in args.input_shape.split(","))

    # Resolve and load the model
    print(f"Loading model: {args.model}", file=sys.stderr)
    model = resolve_model(args.model)

    # Export to ONNX
    onnx_bytes = export_to_onnx(model, input_shape, args.output)

    # If no output file, write raw bytes to stdout for piping
    if args.output is None:
        sys.stdout.buffer.write(onnx_bytes)


if __name__ == "__main__":
    main()
