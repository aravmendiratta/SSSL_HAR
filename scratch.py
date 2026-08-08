import torch
num_views = 6
mask = torch.zeros((4, num_views))
all_zeros = (mask.sum(dim=1) == 0)
try:
    random_idx = torch.randint(0, num_views, (all_zeros.sum(),))
    print("Success")
except Exception as e:
    print(f"Error: {e}")
