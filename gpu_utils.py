# gpu_utils.py
import logging
import torch

logger = logging.getLogger(__name__)

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"


def configure_gpu():
    """Configure GPU settings and return a torch.device."""
    print("== Configuring GPU ==")

    # Check if GPU is available using PyTorch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"PyTorch detected {gpu_count} GPU(s)")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # Return a torch.device;
        return torch.device("cuda:0")
    else:
        print("No GPU detected by PyTorch")
        return torch.device("cpu")


def check_pytorch_gpu():
    """Check if PyTorch can detect and use GPUs."""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"PyTorch detected {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("No GPU detected by PyTorch")
    except ImportError:
        print("PyTorch not installed, skipping PyTorch GPU check")
    except Exception as e:
        print(f"Error checking PyTorch GPU: {e}")
    return False
