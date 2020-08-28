from torch import nn, Tensor

from .layers import Linear
from . import RegressionModel, RaoBRegressionModel, ClassificationModel
from .. import prior

__all__ = ('LinearNealNormal', 'LinearPrior', 'DenseNet', 'RaoBDenseNet', 'ClassificationDenseNet')

def LinearNealNormal(in_dim: int, out_dim: int, std_w: float, std_b: float) -> nn.Module:
    return Linear(prior.Normal((out_dim, in_dim), 0., std_w/in_dim**.5),
                  prior.Normal((out_dim,), 0., std_b))


def LinearPrior(in_dim, out_dim, prior_w=prior.Normal, loc_w=0., std_w=1.,
                     prior_b=prior.Normal, loc_b=0., std_b=1., scaling_fn=None):
    if scaling_fn is None:
        def scaling_fn(std, dim):
            return std/dim**0.5
    return Linear(prior_w((out_dim, in_dim), loc_w, scaling_fn(std_w, in_dim)),
                  prior_b((out_dim,), 0., std_b))


def DenseNet(in_features, out_features, width, noise_std=1.,
             prior_w=prior.Normal, loc_w=0., std_w=2**.5,
             prior_b=prior.Normal, loc_b=0., std_b=1.,
             scaling_fn=None):
    return RegressionModel(
        nn.Sequential(
            LinearPrior(in_features, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn),
            nn.ReLU(),
            LinearPrior(width, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn),
            nn.ReLU(),
            LinearPrior(width, out_features, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn)
        ), noise_std)


def ClassificationDenseNet(in_features, out_features, width, softmax_temp=1.,
                           prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                           prior_b=prior.Normal, loc_b=0., std_b=1.,
                           scaling_fn=None):
    return ClassificationModel(
        nn.Sequential(
            LinearPrior(in_features, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn),
            nn.ReLU(),
            LinearPrior(width, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn),
            nn.ReLU(),
            LinearPrior(width, out_features, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn)
        ), softmax_temp)


def RaoBDenseNet(x_train: Tensor, y_train: Tensor, width: int,
                 noise_std: float=1.,
                 prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                 prior_b=prior.Normal, loc_b=0., std_b=1.,
                 scaling_fn=None) -> nn.Module:
    in_features = x_train.size(-1)
    return RaoBRegressionModel(
        x_train, y_train, noise_std,
        last_layer_std=(2/width)**.5,
        net=nn.Sequential(
            LinearPrior(in_features, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn),
            nn.ReLU(),
            LinearPrior(width, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn),
            nn.ReLU()))
