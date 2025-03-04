import torch
import matplotlib.pyplot as plt
import copy

do_visualize = True
def visualize_tensor(tensor: torch.Tensor):
    if not do_visualize:
        return
    tensor = tensor.clone()
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if tensor[0][0][0] in [0, 1]:
        plt.set_cmap('binary')
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(tensor)
    plt.show()