from typing import Any

import torch.optim
from torch.optim.optimizer import ParamsT


class MyAdam(torch.optim.Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: float,
            beta1: float = 0.9, beta2: float = 0.999,
            eps: float = 1e-6,
            weight_decay: float = 0.0,
    ):
        default_params = {
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, default_params)

    def _do_weight_decay(self, param_group: dict[str, Any]):
        wd = param_group['weight_decay']
        if wd == 0:
            return

        for param in param_group['params']:
            if param.grad is not None:
                param.grad.add_(wd * param)

    def _update_states_for_param_group(self, param_group: dict[str, Any]):
        beta1 = param_group['beta1']
        beta2 = param_group['beta2']

        for param in param_group['params']:
            if param.grad is not None:
                state = self.state[param]

                if 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

                if 'm0' not in state:
                    state['m0'] = torch.zeros_like(param.grad)
                state['m0'] = beta1 * state['m0'] + (1 - beta1) * param.grad

                if 'v0' not in state:
                    state['v0'] = torch.zeros_like(param.grad)
                state['v0'] = beta2 * state['v0'] + (1 - beta2) * param.grad ** 2

                t = state['step']
                state['m'] = state['m0'] / (1 - beta1 ** t)
                state['v'] = state['v0'] / (1 - beta2 ** t)

    def _update_weights(self, param_group: dict[str, Any]):
        lr = param_group['lr']
        eps = param_group['eps']
        for param in param_group['params']:
            if param.grad is not None:
                state = self.state[param]
                m = state['m']
                v = state['v']
                update_by = - lr * m / (v + eps) ** 0.5
                param.add_(update_by)

    @torch.no_grad()
    def step(self, closure = None):
        for param_group in self.param_groups:
            self._do_weight_decay(param_group)
            self._update_states_for_param_group(param_group)
            self._update_weights(param_group)
