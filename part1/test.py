import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device:', device)
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print('Tensor on CUDA:', x)
else:
    print('CUDA is not available. Falling back to CPU.')
    device = torch.device('cpu')
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print('Tensor on CPU:', x)
