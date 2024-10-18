import torch, torchvision, torchaudio

if __name__ == '__main__':
    print("Hello torch!")
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)
    print("torchaudio version:", torchaudio.__version__)
    print("torch cuda available:", torch.cuda.is_available())