import torch

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
from typing import Tuple

layers = ('down_sampling.9', 'features.2', 'features.5', 'features.8', 'up_sampling.8')

def get_feature_extractor(model):
    return create_feature_extractor(
        model, return_nodes=list(layers)
    )

def _get_num_macs(module, input_shape, output_shape, inputs):
    num_macs = 0
    num_params = ModuleProfiler.get_num_params(module)
    num_seconds = 0
    
    if isinstance(module, nn.Conv2d):
        num_macs = (
            input_shape[1] * output_shape[1] * 
            module.kernel_size[0] * module.kernel_size[1] *
            output_shape[2] * output_shape[3] // module.groups) \
            * output_shape[0]

    elif isinstance(module, nn.ConvTranspose2d):
        num_macs = (
            input_shape[1] * output_shape[1] * 
            module.kernel_size[0] * module.kernel_size[1] *
            output_shape[2] * output_shape[3] // module.groups) \
            * output_shape[0]
    
    elif isinstance(module, nn.InstanceNorm2d):
        num_macs = (
            inputs[0].numel()
        )
    
    elif isinstance(module, ConvBNReLU):
        num_params = 0
        num_macs_conv, num_params_conv, num_seconds_conv = _get_num_macs(module[0], input_shape, output_shape, inputs)
        num_macs_bn, num_params_bn, num_seconds_bn = module[1].num_macs, module[1].num_params, module[1].num_seconds
        
        num_macs += num_macs_conv + num_macs_bn
        num_seconds += num_seconds_conv + num_seconds_bn
        num_params += num_params_bn + num_params_conv
        
    
    elif isinstance(module, InvertedResidualChannels):
        num_params = 0
        for op in (*module.res_ops, *module.dw_ops, module.pw_bn):
            num_macs += op.num_macs
            num_seconds += op.num_seconds
            num_params += op.num_params
    
    else:
        num_macs = 0
        num_params = 0
        num_seconds = 0
        for m in module.children():
            num_macs += getattr(m, 'num_macs', 0)
            num_params += getattr(m, 'num_params', 0)
            num_seconds += getattr(m, 'num_seconds', 0)

    return num_macs, num_params, num_seconds


class ModuleProfiler:
    
    def __init__(
        self,
        module: nn.Module,
        height: int,
        width: int,
        batch: int = 1,
        channel: int = 3,
        device = torch.device('cpu'),
    ):
        self._module = module.eval().to(device)
        self._data = torch.randn(batch, channel, height, width, device=device)
        self._hooks = []
    
    @staticmethod
    def get_num_params(module: nn.Module):
        return sum([p.numel() for p in module.parameters()])
    
    @staticmethod
    def profiling_hook(
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor
    ):
        input_shape = inputs[0].shape
        output_shape = output.shape

        module.num_macs, module.num_params, module.num_seconds = _get_num_macs(module, input_shape, output_shape, inputs)
    
    def __call__(self):
        def add_profiling_hook(module: nn.Module):
            self._hooks.append(module.register_forward_hook(self.profiling_hook))
        
        self._module.apply(add_profiling_hook)
        with torch.inference_mode():
            _ = self._module(self._data)
        
        for hook in self._hooks:
            hook.remove()

        return self._module.num_macs, self._module.num_params

def flatten_modules(children):
    flattened = []
    for child in children:
        if len(list(child.children())) == 0:
            flattened.append(child)
        else:
            flattened += flatten_modules(child.children())
    return flattened

def create_downsampling(in_channels, out_channels):
    down_sampling = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels, out_channels[0], kernel_size=7, padding=0, bias=False),
        nn.BatchNorm2d(out_channels[0]),
        nn.ReLU(True)
    ]

    n_downsampling = 2
    for i in range(n_downsampling):
        down_sampling += [
            nn.Conv2d(out_channels[i],
                      out_channels[i + 1],
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels[i + 1]),
            nn.ReLU(True)
        ]
        
    return nn.Sequential(*down_sampling)

def create_upsampling(out_channels):
    up_sampling = []
    n_downsampling = 2
    for i in range(n_downsampling):
        up_sampling += [
            nn.ConvTranspose2d(out_channels[i],
                               out_channels[i + 1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False),
            nn.BatchNorm2d(out_channels[i + 1]),
            nn.ReLU(True)
        ]
    up_sampling += [nn.ReflectionPad2d(3)]
    up_sampling += [nn.Conv2d(out_channels[-2], out_channels[-1], kernel_size=7, padding=0)]
    up_sampling += [nn.Tanh()]
    return nn.Sequential(*up_sampling)


def teacher_pruning(teacher, cumputational_budget, channel_lower_bound=4) -> nn.Module:
    student = copy.deepcopy(teacher)
    
    scaling_factors = []
     for module in list(student.children())[0].children():
        if isinstance(module, nn.BatchNorm2d):
            scaling_factors.extend(module.weight.data.abs().clone())
            
    for inverted_res in list(student.children())[1].children():
        for conv_bn in (*inverted_res.res_ops, *inverted_res.dw_ops):
            for module in flatten_modules(conv_bn):
                if isinstance(module, nn.BatchNorm2d):
                    scaling_factors.extend(module.weight.data.abs().clone())
                    break

    for module in list(student.children())[2].children():
        if isinstance(module, nn.BatchNorm2d):
            scaling_factors.extend(module.weight.data.abs().clone())
                    
    scaling_factors = torch.tensor(scaling_factors)
    
    sorted_scaling_factors, _ = torch.sort(scaling_factors)
    l, r = 0, len(sorted_scaling_factors) - 1
    
    def create_mask(weights, cur_scaling_factor):
        mask = weights.gt(cur_scaling_factor)
        if mask.sum().item() >= channel_lower_bound:
            return mask
        
        sorted_idx = torch.argsort(weights)
        mask = torch.zeros(mask.shape, dtype=torch.bool)
        mask[sorted_idx[-channel_lower_bound:]] = True
        return mask
    
    while l < r:
        student = copy.deepcopy(teacher)
        
        mid_index = (l + r) // 2
        cur_scaling_factor = sorted_scaling_factors[mid_index]
        
        out_channels = []
        downsampling = list(student.children())[0]
        for module in downsampling:
            if isinstance(module, nn.BatchNorm2d):
                weight_copy = module.weight.data.abs().clone()
                num_channels = create_mask(weight_copy, cur_scaling_factor).sum().item()
                out_channels.append(num_channels)
        student.down_sampling = create_downsampling(3, out_channels)
        
        input_dim = out_channels[-1]
        for inverted_res in list(student.children())[1].children():
            res_channels, dw_channels = [], []
            for conv_bn in inverted_res.res_ops:
                for module in flatten_modules(conv_bn):
                    if isinstance(module, nn.BatchNorm2d):
                        weight_copy = module.weight.data.abs().clone()
                        res_channels.append(create_mask(weight_copy, cur_scaling_factor).sum().item())
            for conv_bn in inverted_res.dw_ops:
                for module in flatten_modules(conv_bn):
                    if isinstance(module, nn.BatchNorm2d):
                        weight_copy = module.weight.data.abs().clone()
                        dw_channels.append(create_mask(weight_copy, cur_scaling_factor).sum().item())
            inverted_res.input_dim = input_dim
            inverted_res.dw_channels, inverted_res.res_channels = dw_channels, res_channels
            inverted_res.res_ops, inverted_res.dw_ops, inverted_res.pw_bn = inverted_res._build()
        
        out_channels = [input_dim]
        upsampling = list(student.children())[2]
        for module in upsampling:
            if isinstance(module, nn.BatchNorm2d):
                weight_copy = module.weight.data.abs().clone()
                num_channels = create_mask(weight_copy, cur_scaling_factor).sum().item()
                out_channels.append(num_channels)
        out_channels.append(3)
        student.up_sampling = create_upsampling(out_channels)
        
        # print(student)
        profiler = ModuleProfiler(student, 256, 256)
        student_num_macs, student_num_params = profiler()
        student_cumputational_budget = student_num_macs
        
        print(cumputational_budget, student_cumputational_budget, f'{student_cumputational_budget/cumputational_budget * 10}%')
        
        if student_cumputational_budget > cumputational_budget:
            l = mid_index + 1
        else:
            r = mid_index
    return student