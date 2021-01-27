import torch

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=1, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=1, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=2)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=2)
    cov1 = torch.bmm(emb1, torch.transpose(emb1,1,2))
    cov2 = torch.bmm(emb2, torch.transpose(emb2,1,2))
    cost = torch.mean((cov1-cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim):
    bs = emb1.size(0)
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.bmm(emb1, torch.transpose(emb1,1,2))
    K2 = torch.bmm(emb2, torch.transpose(emb2,1,2))
    RK1 = torch.bmm(R.expand_as(K1), K1)
    RK2 = torch.bmm(R.expand_as(K2), K2)
    ans = torch.bmm(RK1, RK2)
    HSIC = 0
    for index in range(bs):
        HSIC += torch.trace(ans[index])
    return HSIC


