from distutils.version import LooseVersion

from torch import nn

from .utils import NNComputeModuleProfile, ProfilerResult, is_compute_layer
from .vision_hooks import *

logger = logging.getLogger(__name__)


def prRed(skk): print("\033[91m{}\033[00m".format(skk))


def prGreen(skk): print("\033[92m{}\033[00m".format(skk))


def prYellow(skk): print("\033[93m{}\033[00m".format(skk))


if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logger.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=torch.__version__))

default_dtype = torch.float64

register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    # Conv2dSamePadding: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,

}

if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    register_hooks.update({
        nn.SyncBatchNorm: count_bn
    })


def profile(model: nn.Module, inputs, custom_ops=None, verbose=True):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('num_act', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('num_dp', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('data_reuse', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('weight_reuse', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('dim_dp', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('input_dim', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('output_dim', torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(**inputs)

    per_compute_layer_complexity = []

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        total_num_act, total_num_dp = 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params, m_num_act, m_num_dp, m_weight_reuse, m_input_reuse, m_dim_dp \
                    = m.total_ops.item(), m.total_params.item(), m.num_act.item(), m.num_dp.item(), \
                      m.weight_reuse.item(), m.input_reuse.item(), m.dim_dp.item()
            else:
                m_ops, m_params, m_num_act, m_num_dp = dfs_count(m, prefix=prefix + "\t")
                m_weight_reuse = 0
                m_input_reuse = 0
                m_dim_dp = 0
            total_ops += m_ops
            total_params += m_params
            total_num_act += m_num_act
            total_num_dp += m_num_dp
            module_name = m._get_name()
            # print(prefix, module_name, '(ops:', m_ops, 'params:', m_params, 'act:', m_num_act, 'dp:', m_num_dp,')')

            if is_compute_layer(m):
                per_compute_layer_complexity.append([
                    module_name,
                    m_ops,
                    m_params,
                    m_num_act,
                    m_num_dp,
                    m_dim_dp,
                    NNComputeModuleProfile(m, m.input_dim, m.output_dim)
                ])

        return total_ops, total_params, total_num_act, total_num_dp

    total_ops, total_params, total_num_act, total_num_dp = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop('total_params')
        m._buffers.pop('num_act')
        m._buffers.pop('num_dp')
        m._buffers.pop('data_reuse')
        m._buffers.pop('weight_reuse')
        m._buffers.pop('dim_dp')
        m._buffers.pop('input_dim')
        m._buffers.pop('output_dim')

    return total_ops, total_params, total_num_act, total_num_dp, per_compute_layer_complexity


class Profiler:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    def profile(self):
        macs, params, num_act, num_dp, per_compute_layer_complexity = profile(self.model, inputs=self.inputs)
        return ProfilerResult(macs, params, num_act, num_dp, per_compute_layer_complexity)


def profile_compute_layers(model, inputs, custom_ops=None, verbose=False):
    profiling_results = profile(model, inputs, custom_ops=custom_ops, verbose=verbose)
    total_params = profiling_results[1]

    input_details = [p[-1] for p in profiling_results[-1]]

    ret = {}
    count = 0

    num_ones = 0
    num_ele = 0
    for idx, (n, m) in enumerate(model.named_modules()):
        if idx == 0:
            continue

        if not is_compute_layer(m):
            continue

        setattr(input_details[count], 'param_proportion', input_details[count].num_param / total_params)
        ret[n + '.weight'] = input_details[count]

        num_ones += input_details[count].mask.sum()
        num_ele += input_details[count].mask.numel()

        count += 1

    model_sparsity = 1 - 1.0 * num_ones.item() / num_ele
    return ret, model_sparsity