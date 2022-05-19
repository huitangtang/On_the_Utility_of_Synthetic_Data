import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import contextlib

# NOTE hyper-parameters we use in VAT
# n_power: a number of power iteration for approximation of r_vadv
# XI: a small float for the approx. of the finite difference method
# epsilon: the value for how much deviate from original data point X


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, G, params):
        super(VAT, self).__init__()
        self.G = G
        self.n_power = params.ip
        self.XI = params.xi
        self.epsilon = params.eps_vat

    def forward(self, X, logit):
        vat_loss = virtual_adversarial_loss(X, logit, self.G, self.n_power,
                                            self.XI, self.epsilon)
        return vat_loss  # already averaged


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1, 1)
    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    # print(torch.norm(d.view(d.size(0), -1), dim=1))
    return d


def generate_virtual_adversarial_perturbation(x, logit, G, n_power, XI,
                                              epsilon):
    d = torch.randn_like(x)
    with _disable_tracking_bn_stats(G):
        for _ in range(n_power):
            d = XI * get_normalized_vector(d).requires_grad_()
            logit_m = G(x + d)[0]
            dist = kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, G, n_power, XI, epsilon):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, G,
                                                       n_power, XI, epsilon) #logit or detached logit??? | generate_virtual_adversarial_perturbation several times ???
    with _disable_tracking_bn_stats(G):
        logit_p = logit.detach() #detach or not???
        logit_m = G(x + r_vadv)[0]
        loss = kl_divergence_with_logit(logit_p, logit_m)
        #loss = kl_divergence_with_logit(logit, logit_m)

    return loss
    
    
    