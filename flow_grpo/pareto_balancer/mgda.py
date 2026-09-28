import numpy as np
import torch
from scipy.optimize import minimize


def mgda_solve(grads: list[torch.Tensor], alpha_floor: float = 0.0) -> torch.Tensor:
    if not grads:
        raise ValueError("mgda_solve requires at least one gradient vector")

    if len(grads) == 1:
        return torch.ones(1, device=grads[0].device, dtype=torch.float32)

    grad_matrix = torch.stack([grad.reshape(-1).float() for grad in grads], dim=0)
    gram = (grad_matrix @ grad_matrix.t()).double().cpu().numpy()
    task_count = grad_matrix.shape[0]

    # Validate alpha_floor feasibility: sum of floors must be <= 1.0
    alpha_floor = max(alpha_floor, 0.0)
    if alpha_floor * task_count > 1.0:
        alpha_floor = 1.0 / task_count

    def objective(alpha: np.ndarray) -> float:
        return 0.5 * float(alpha @ gram @ alpha)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return gram @ alpha

    alpha0 = np.full(task_count, 1.0 / task_count, dtype=np.float64)
    bounds = [(alpha_floor, None)] * task_count
    constraints = [
        {
            "type": "eq",
            "fun": lambda a: a.sum() - 1.0,
            "jac": lambda a: np.ones(task_count, dtype=np.float64),
        }
    ]

    result = minimize(
        objective,
        alpha0,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-8, "maxiter": 500},
    )

    if not result.success:
        candidate = np.clip(result.x, alpha_floor, None)
        if candidate.sum() < 1e-8:
            raise RuntimeError(f"MGDA solve failed: {result.message}")
        result.x = candidate

    alphas = torch.from_numpy(np.clip(result.x, alpha_floor, None)).float()
    alphas = alphas / alphas.sum().clamp_min(1e-12)
    return alphas.to(grads[0].device)
