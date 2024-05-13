from msbench.scheduler.base_scheduler import BaseScheduler
from msbench.sparsity_mappings import SUPPORT_CONV, SUPPORT_LINEAR
import torch


class ProbMaskPruneScheduler(BaseScheduler):
    def __init__(self, custom_config_dict, prune_rate, total_iterations,
                 warmup_iteration_stamps, finetune_iteration_stamps,
                 start_prune_rate=0, no_prune_keyword='', no_prune_layer=''):
        super().__init__(custom_config_dict, no_prune_keyword, no_prune_layer)
        self.custom_config_dict = custom_config_dict
        self.process_no_prune_layers(no_prune_keyword, no_prune_layer)
        self.prune_rate = prune_rate
        self.prune_rate_target = prune_rate
        self.total_iterations = total_iterations
        self.warmup_iteration_stamps = warmup_iteration_stamps
        self.finetune_iteration_stamps = finetune_iteration_stamps
        self.start_prune_rate = start_prune_rate

    def adjust_sparsity(self, current_iterations):
        if current_iterations < self.warmup_iteration_stamps:
            self.prune_rate = 0
        elif current_iterations < self.finetune_iteration_stamps:
            self.prune_rate = self.prune_rate_target -\
                              (self.prune_rate_target - self.start_prune_rate) *\
                              (1-(current_iterations - self.warmup_iteration_stamps) /\
                              (self.finetune_iteration_stamps - self.warmup_iteration_stamps)) ** 3  # noqa E501
        else:
            self.prune_rate = self.prune_rate_target

    def adjust_tau(self, model, current_iterations):
        tau = 1 / ((1 - 0.03) * (1 - current_iterations / self.total_iterations) + 0.03)  # noqa E501
        for _, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                m.weight_fake_sparse.mask_generator.set_tau(tau)

    def _generate_mask(self, model):
        mask_dict = {}
        for name, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                mask = m.weight_fake_sparse.generate_mask(m.weight_fake_sparse.scores)  # noqa E501
                mask_dict[name] = mask
        return mask_dict

    def export_sparse_model(self, model):
        self.fix_model_subnet(model)
        mask_dict = self._generate_mask(model)
        self._apply_mask(model, mask_dict)
        self._enable_fake_sparse(model, False)

    def prune_model(self, model, current_iterations):  # noqa: F401
        self.adjust_sparsity(current_iterations)
        self.adjust_tau(model, current_iterations)
        with torch.no_grad():
            self._constrainScoreByWhole(model)

    def unfreeze_model_subnet(self, model):
        print("=> Unfreezing model subnet")
        for name, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                if name in self.no_prune_keyword + self.no_prune_layer:
                    continue
                else:
                    if hasattr(m, "scores"):
                        print(f"==> Gradient to {name}.scores")
                        m.scores.requires_grad = True

    def unfreeze_model_weights(self, model):
        print("=> Unfreezing model weights")
        for name, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                print(f"==> Gradient to {name}.weight")
                m.weight.requires_grad = True
                if hasattr(m, "bias") and m.bias is not None:
                    print(f"==> Gradient to {name}.bias")
                    m.bias.requires_grad = True

    def freeze_model_weights(self, model):
        print("=> Freezing model weights")
        for name, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                print(f"==> No gradient to {name}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    print(f"==> Setting gradient of {name}.weight to None")
                    m.weight.grad = None
                if hasattr(m, "bias") and m.bias is not None:
                    print(f"==> No gradient to {name}.bias")
                    m.bias.requires_grad = False
                    if m.bias.grad is not None:
                        print(f"==> Setting gradient of {name}.bias to None")
                        m.bias.grad = None

    def freeze_model_subnet(self, model):
        print("=> Freezing model subnet")
        for name, m in model.named_modules():
            if hasattr(m, "scores"):
                m.scores.requires_grad = False
                print(f"==> No gradient to {name}.scores")
                if m.scores.grad is not None:
                    print(f"==> Setting gradient of {name}.scores to None")
                    m.scores.grad = None

    def fix_model_subnet(self, model):
        print("=> fixing model subnet")
        for name, m in model.named_modules():
            if hasattr(m, "scores"):
                print(f"==> fixing {name} subnet")
                m.enable_fix_subnet(enabled=True)

    def _solve_v_total(self, model, total):
        k = total * (1 - self.prune_rate)
        a, b = 0, 0
        for n, m in model.named_modules():
            if hasattr(m, "scores"):
                b = max(b, m.scores.max())

        def f(v):
            s = 0
            for n, m in model.named_modules():
                if hasattr(m, "scores"):
                    s += (m.scores - v).clamp(0, 1).sum()
            return s - k

        if f(0) < 0:
            return 0
        itr = 0
        while (1):
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 20:
                break
            if obj < 0:
                b = v
            else:
                a = v
        v = max(0, v)
        return v

    def _constrainScoreByWhole(self, model):
        total = 0
        for n, m in model.named_modules():
            if hasattr(m, "scores"):
                total += m.scores.nelement()
        v = self._solve_v_total(model, total)
        for n, m in model.named_modules():
            if hasattr(m, "scores"):
                m.scores.sub_(v).clamp_(0, 1)
