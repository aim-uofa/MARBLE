import torch


def capture_grad_vector(params: list[torch.nn.Parameter]) -> torch.Tensor:
    flat_chunks: list[torch.Tensor] = []
    device = None
    for param in params:
        if not param.requires_grad:
            continue
        if device is None:
            device = param.device
        if param.grad is None:
            flat_chunks.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            flat_chunks.append(param.grad.detach().reshape(-1).float().clone())

    if not flat_chunks:
        target_device = device if device is not None else torch.device("cpu")
        return torch.zeros(0, device=target_device, dtype=torch.float32)

    return torch.cat(flat_chunks, dim=0)


def clear_parameter_grads(params: list[torch.nn.Parameter]) -> None:
    for param in params:
        if param.requires_grad:
            param.grad = None


def restore_grad_vector(params: list[torch.nn.Parameter], grad_vector: torch.Tensor) -> None:
    offset = 0
    for param in params:
        if not param.requires_grad:
            continue
        numel = param.numel()
        chunk = grad_vector[offset : offset + numel]
        if chunk.numel() != numel:
            raise ValueError("Gradient vector size does not match parameter list")
        param.grad = chunk.to(device=param.device, dtype=param.dtype).view_as(param).clone()
        offset += numel

    if offset != grad_vector.numel():
        raise ValueError("Gradient vector contains extra values")


def sanitize_grad_vector(grad_vector: torch.Tensor) -> tuple[torch.Tensor, bool]:
    finite_mask = torch.isfinite(grad_vector)
    if bool(finite_mask.all().item()):
        return grad_vector, False

    sanitized = torch.where(finite_mask, grad_vector, torch.zeros_like(grad_vector))
    return sanitized, True
