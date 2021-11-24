dependencies = ['torch', 'numpy']

from efficientnetv2 import get_efficientnet_v2


def efficientnet_v2_s(pretrained=False, nclass=1000, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_s', pretrained, nclass, **kwargs)


def efficientnet_v2_m(pretrained=False, nclass=1000, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_m', pretrained, nclass, **kwargs)


def efficientnet_v2_l(pretrained=False, nclass=1000, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_l', pretrained, nclass, **kwargs)


def efficientnet_v2_s_in21k(pretrained=False, nclass=21843, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_s_in21k', pretrained, nclass, **kwargs)


def efficientnet_v2_m_in21k(pretrained=False, nclass=21843, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_m_in21k', pretrained, nclass, **kwargs)


def efficientnet_v2_l_in21k(pretrained=False, nclass=21843, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_l_in21k', pretrained, nclass, **kwargs)


def efficientnet_v2_xl_in21k(pretrained=False, nclass=21843, **kwargs):
    return get_efficientnet_v2('efficientnet_v2_xl_in21k', pretrained, nclass)
