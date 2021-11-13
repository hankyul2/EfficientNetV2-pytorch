dependencies = ['torch', 'python-box', 'numpy']

from src.efficientnet_v2 import get_efficientnet_v2


def efficientnet_v2_s(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_s', pretrained, **kwargs)


def efficientnet_v2_m(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_m', pretrained, **kwargs)


def efficientnet_v2_l(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_l', pretrained, **kwargs)


def efficientnet_v2_s_in21k(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_s_in21k', pretrained, **kwargs)


def efficientnet_v2_m_in21k(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_m_in21k', pretrained, **kwargs)


def efficientnet_v2_l_in21k(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_l_in21k', pretrained, **kwargs)


def efficientnet_v2_xl_in21k(pretrained=False, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_xl_in21k', pretrained)
