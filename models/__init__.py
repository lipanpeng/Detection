from models.yolov3.models import Darknet


def get_model(name, net_cfg=None, cfg=None):
    if name == 'yolov3':
        model = Darknet(net_cfg, cfg)
    else:
        raise RuntimeError('Not implemented')

    return model