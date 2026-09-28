import torch
import torch.distributed as dist


def sync_grad_vector(grad_vector: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return grad_vector

    synced = grad_vector.clone()
    dist.all_reduce(synced, op=dist.ReduceOp.SUM)
    synced /= dist.get_world_size()
    return synced


def sync_grad_vectors(grad_vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return grad_vectors
    if not grad_vectors:
        return grad_vectors
    stacked = torch.cat(grad_vectors)
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
    stacked /= dist.get_world_size()
    sizes = [g.numel() for g in grad_vectors]
    return list(stacked.split(sizes))
