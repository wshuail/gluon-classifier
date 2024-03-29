from .shufflenet import shufflenet0_5_g3
from .shufflenetv2 import shufflenetv2_0_5

_models = {
    'shufflenet0.5_g3': shufflenet0_5_g3,
    'shufflenetv2_0.5': shufflenetv2_0_5
}


def get_model(name, num_classes, **kwargs):
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](num_classes, **kwargs)
    return net
