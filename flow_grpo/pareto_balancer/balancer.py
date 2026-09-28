import torch

from flow_grpo.pareto_balancer.mgda import mgda_solve


def balance_reward_gradients(
    grad_vectors: list[torch.Tensor],
    normalize: bool = True,
    eps: float = 1e-12,
    weights: list[float] | None = None,
    alpha_floor: float = 0.0,
) -> dict[str, torch.Tensor | str]:
    if not grad_vectors:
        raise ValueError("balance_reward_gradients requires at least one gradient vector")

    device = grad_vectors[0].device
    dtype = grad_vectors[0].dtype
    finite_flags = [bool(torch.isfinite(grad).all().item()) for grad in grad_vectors]
    finite_mask = torch.tensor(finite_flags, device=device, dtype=torch.bool)
    raw_norms = torch.stack(
        [
            grad.norm().to(device=device, dtype=torch.float32)
            if is_finite
            else torch.zeros((), device=device, dtype=torch.float32)
            for grad, is_finite in zip(grad_vectors, finite_flags, strict=True)
        ]
    )
    valid_mask = finite_mask & (raw_norms > eps)

    if not torch.any(valid_mask):
        return {
            "combined_grad": torch.zeros_like(grad_vectors[0]),
            "alphas": torch.zeros(len(grad_vectors), device=device, dtype=torch.float32),
            "raw_norms": raw_norms,
            "valid_mask": valid_mask,
            "rescale_factor": torch.tensor(0.0, device=device, dtype=torch.float32),
            "fallback_reason": "all_zero",
        }

    valid_indices = valid_mask.nonzero(as_tuple=False).flatten().tolist()
    valid_grads = [grad_vectors[index] for index in valid_indices]

    if normalize:
        solve_grads = [grad / grad.norm().clamp_min(eps) for grad in valid_grads]
    else:
        solve_grads = valid_grads

    # Apply per-gradient weights (after normalization so they change the convex hull geometry)
    if weights is not None:
        valid_weights = [weights[index] for index in valid_indices]
        solve_grads = [w * g for w, g in zip(valid_weights, solve_grads)]

    alphas = torch.zeros(len(grad_vectors), device=device, dtype=torch.float32)

    if len(solve_grads) == 1:
        alphas[valid_indices[0]] = 1.0
        return {
            "combined_grad": valid_grads[0].clone(),
            "alphas": alphas,
            "raw_norms": raw_norms,
            "valid_mask": valid_mask,
            "rescale_factor": raw_norms[valid_indices[0]],
            "fallback_reason": "single_valid",
        }

    try:
        solve_alphas = mgda_solve(solve_grads, alpha_floor=alpha_floor).to(device=device, dtype=torch.float32)
        for local_index, global_index in enumerate(valid_indices):
            alphas[global_index] = solve_alphas[local_index]
        combined_direction = torch.stack(
            [a * g for a, g in zip(solve_alphas, solve_grads)], dim=0
        ).sum(dim=0)
        fallback_reason = "none"
    except RuntimeError:
        combined_direction = torch.stack(solve_grads, dim=0).mean(dim=0)
        alphas[valid_mask] = 1.0 / valid_mask.sum()
        fallback_reason = "solver_failed"

    if normalize:
        avg_norm = raw_norms[valid_mask].mean()
        combined_grad = combined_direction * avg_norm
        rescale_factor = avg_norm
    else:
        combined_grad = combined_direction
        rescale_factor = torch.tensor(1.0, device=device, dtype=torch.float32)

    return {
        "combined_grad": combined_grad.to(device=device, dtype=dtype),
        "alphas": alphas,
        "raw_norms": raw_norms,
        "valid_mask": valid_mask,
        "rescale_factor": rescale_factor,
        "fallback_reason": fallback_reason,
    }
