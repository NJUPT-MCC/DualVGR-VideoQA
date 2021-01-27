import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
    pass