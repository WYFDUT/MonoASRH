from lib.models.MonoASRH import MonoASRH


def build_model(cfg,mean_size):
    if cfg['type'] == 'MonoASRH':
        return MonoASRH(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
