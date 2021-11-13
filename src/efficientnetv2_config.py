def get_efficientnet_v2_hyperparam(model_name):
    from box import Box
    # train_size, eval_size, dropout, randaug, mixup
    if 'efficientnet_v2_s' in model_name:
        end = 300, 384, 0.2, 10, 0
    elif 'efficientnet_v2_m' in model_name:
        end = 384, 480, 0.3, 15, 0.2
    elif 'efficientnet_v2_l' in model_name:
        end = 384, 480, 0.4, 20, 0.5
    elif 'efficientnet_v2_xl' in model_name:
        end = 384, 512, 0.4, 20, 0.5
    return Box({"init_train_size": 128, "init_dropout": 0.1, "init_randaug": 5, "init_mixup": 0,
             "end_train_size": end[0], "end_dropout": end[2], "end_randaug": end[3], "end_mixup": end[4], "eval_size": end[1]})


def get_efficientnet_v2_structure(model_name):
    if 'efficientnet_v2_s' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
    elif 'efficientnet_v2_m' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]
    elif 'efficientnet_v2_l' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 7, False, True),
            (4, 3, 2, 64, 96, 7, False, True),
            (4, 3, 2, 96, 192, 10, True, False),
            (6, 3, 1, 192, 224, 19, True, False),
            (6, 3, 2, 224, 384, 25, True, False),
            (6, 3, 1, 384, 640, 7, True, False),
        ]
    elif 'efficientnet_v2_xl' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 8, False, True),
            (4, 3, 2, 64, 96, 8, False, True),
            (4, 3, 2, 96, 192, 16, True, False),
            (6, 3, 1, 192, 256, 24, True, False),
            (6, 3, 2, 256, 512, 32, True, False),
            (6, 3, 1, 512, 640, 8, True, False),
        ]
