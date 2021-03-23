import torch

# Tutorials:
# https://medium.com/@fanzongshaoxing/accelerate-pytorch-model-with-tensorrt-via-onnx-d5b5164b369

def export_onnx_model(
    model, 
    input_shape,
    onnx_path,
    input_names=None,
    output_names=None,
    dynamic_axes=None
    ):

    #inputs = torch.ones(*input_shape)
    #model(inputs)
    torch.onnx.export(
        model,
        input_shape,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )