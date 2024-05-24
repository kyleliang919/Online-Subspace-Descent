from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
from .galore_projector import GaLoreProjector
# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
    
class GaLoreLion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn
        self.init_lr = lr
        
    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(filter(lambda p: exists(p.grad), group['params'])):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values
                    
                if "step" not in state:
                    state["step"] = 0
                   
                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"], update_proj_stepsize_ratio = group["lr"]/self.init_lr, name = group["names"][i])

                if "exp_avg" not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    
                exp_avg = state['exp_avg']
                
                # weight update
                update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1)
                
                # GaLore Projection Back
                if "rank" in group:
                    update = state["projector"].project_back(update)

                # take the sign
                update = update.sign_()
                
                p.add_(update, alpha = -lr)
                state["step"] += 1
                
                # decay the momentum running average coefficient
            
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
                
                # stepweight decay
                p.data.mul_(1 - lr * wd)

        return loss
