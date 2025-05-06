import os
#import onnx
import torch

def export_to_onnx(torch_model, onnx_path, device):
    dummy_frames = torch.randn(1, 64, 1, 256, 256).to(device) 
    dummy_queries = torch.randn(1, 64, 2).to(device)  # Adjust shape for your model
    torch.onnx.export(
        torch_model,
        (dummy_frames, dummy_queries),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'frames':  {0: 'batch_size', 1: 'num_frames'},
            'queries': {0: 'batch_size', 1: 'num_points'},
            'output':  {0: 'batch_size', 1: 'num_frames', 2: 'num_points'}
        },
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_log_path = "logs/run_2025-05-05-11-11"
    model_path = os.path.join(training_log_path, "myotracker.pt")
    onnx_save_path = "models/myotracker.onnx"

    model = torch.load(model_path, map_location=device)
    model.eval()

    export_to_onnx(model, onnx_save_path, device)