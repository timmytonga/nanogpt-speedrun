import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamwSNSM(Optimizer):
    """
    - Row norm for compressing step size
    - Unbiased compression of momentum term
    Current implementation
    - Row norm is performed by default on all parameters and keeping the larger dimension


    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            rank: int = 256,
            update_proj_gap: int = 200,
            proj_type: str = "svd"  # choices ['svd']
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.proj_type= proj_type
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        print(f"DEBUG: betas {betas}")

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # subset-norm for compressing adaptive step size second moment term
                if "reduce_dim" in group and "reduce_dim" not in state:
                    state["reduce_dim"] = 0 if grad.shape[0] >= grad.shape[1] else 1
                if "reduce_dim" in group:  # this means we are reducing the min dir
                    norm_dim = 1 - state["reduce_dim"]
                    update_grad = torch.sum(grad**2, dim=norm_dim, keepdim=True)
                else:
                    update_grad = grad**2

                # Projection for compressing momentum term
                if "projector" not in state:
                    state["projector"] = SVDProjector(self.rank, update_proj_gap=self.update_proj_gap, proj_type=self.proj_type)
                proj_grad = state["projector"].project(grad, state["step"])


                # State initialization
                if "exp_avg_sq" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(proj_grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(update_grad)

                # reset exp_avg state when we update
                if ("rank" in group and state["step"] > 1 and state["step"] % self.update_proj_gap == 0):
                    state["exp_avg"] = torch.zeros_like(proj_grad)

                # Now we are ready to update
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Momentum term
                if "rank" in group:
                    exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
                    orth_comp = grad - state["projector"].project_back(proj_grad)
                    numerator = state["projector"].project_back(exp_avg) + orth_comp
                else:
                    exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
                    numerator = exp_avg

                # Subset-norm step size term
                exp_avg_sq.mul_(beta2).add_(update_grad, alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(numerator, denom, value=-step_size)

                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

# svd decomposition
def get_orthogonal_matrix(weights, rank, type):
    module_params = weights

    if module_params.data.dtype != torch.float:
        float_data = False
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float()
    else:
        float_data = True
        matrix = module_params.data

    U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)

    #make the smaller matrix always to be orthogonal matrix
    if type=='right':
        B = Vh[:rank, :]
        if not float_data:
            B = B.to(original_device).type(original_type)
        return B
    elif type=='left':
        A = U[:, :rank]
        if not float_data:
            A = A.to(original_device).type(original_type)
        return A
    elif type=='full':
        A = U[:, :rank]
        B = Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
            B = B.to(original_device).type(original_type)
        return [A, B]
    else:
        raise ValueError('type should be left, right or full')


class SVDProjector:
    """
    This should be created for every parameter
    """
    def __init__(self, rank, verbose=False, update_proj_gap=200, proj_type='svd'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad: torch.Tensor, n_iter):
        if self.proj_type == 'svd':  # this is SVD
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if n_iter is not None and (self.ortho_matrix is None or n_iter % self.update_proj_gap == 0):
                    self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if n_iter is not None and (self.ortho_matrix is None or n_iter % self.update_proj_gap == 0):
                    self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'svd_full':  # this is SVD
            if n_iter is not None and (self.ortho_matrix is None or n_iter % self.update_proj_gap == 0):
                self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
        else:
            raise NotImplementedError("should not be here")

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'svd':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'svd_full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        else:
            raise NotImplementedError("should not be here")

        return full_rank_grad * self.scale
