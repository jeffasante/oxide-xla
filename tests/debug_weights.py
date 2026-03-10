import onnx
import torch
import torchvision.models as models
import numpy as np

m = onnx.load('resnet18.onnx')
onnx_inits = {t.name: t for t in m.graph.initializer}

model = models.resnet18(pretrained=True)
model.eval()
sd = {k: v.numpy() for k, v in model.state_dict().items()}

for oname in onnx_inits:
    if oname in sd:
        diff = np.abs(onnx.numpy_helper.to_array(onnx_inits[oname]) - sd[oname]).max()
        print(f"Match {oname}: diff={diff}")
    elif oname.replace("_", ".") in sd:
        diff = np.abs(onnx.numpy_helper.to_array(onnx_inits[oname]) - sd[oname.replace("_", ".")]).max()
        print(f"Match {oname}: diff={diff}")
    else:
        print(f"MISSING IN SD: {oname}, shape={onnx_inits[oname].dims}")
        t = onnx.numpy_helper.to_array(onnx_inits[oname])
        print(f"  abs max in ONNX: {np.abs(t).max()}")
