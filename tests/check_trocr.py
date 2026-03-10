import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print("Loading TrOCR model...")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-tiny-handwritten")
print("Exporting TrOCR encoder to ONNX...")
encoder = model.encoder

dummy_input = torch.randn(1, 3, 384, 384)
torch.onnx.export(encoder, dummy_input, "trocr_encoder.onnx", opset_version=14)

import onnx
m = onnx.load("trocr_encoder.onnx")
ops = set([n.op_type for n in m.graph.node])
print("TrOCR Encoder Ops:", ops)
