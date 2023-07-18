import os
import numpy as np
import h5py
import arviz
import json
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
import bnn_priors

from bnn_priors import exp_utils
from train_bnn import get_patches, patches2prior

from typing import Dict, Iterable, Tuple

def set_prior_scale(model, dataloader, scale_prior):
    data_sample = dataloader.dataset[0][0].unsqueeze(0)
    alphas = get_patches(model, data_sample, scale_prior=scale_prior)
    alphas = patches2prior(model, alphas)
    model.set_prior_scale(alphas)

def evaluate_model(model: bnn_priors.models.AbstractModel,
                   dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                   samples: Dict[str, torch.Tensor]):
    if hasattr(dataloader.dataset, "tensors"):
        labels = dataloader.dataset.tensors[1].cpu()
    elif hasattr(dataloader.dataset, "targets"):
        labels = torch.tensor(dataloader.dataset.targets).cpu()
    else:
        raise ValueError("I cannot find the labels in the dataloader.")
    N, *_ = labels.shape
    E = exp_utils._n_samples_dict(samples)
    
    log_likelihoods = torch.zeros((E, N), dtype=torch.float64, device='cpu')
    log_priors = torch.zeros((E, N), dtype=torch.float64, device='cpu')
    potentials = torch.zeros((E, N), dtype=torch.float64, device='cpu')

    device = next(iter(model.parameters())).device

    for sample_i, sample in enumerate(exp_utils.sample_iter(samples)):
        with torch.no_grad():
            model.load_state_dict(sample)
            i = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                likelihood_dist = model(batch_x)
                log_likelihood = likelihood_dist.log_prob(batch_y)
                log_prior = model.log_prior()
                potential = - (log_likelihood * (N / batch_x.shape[0]) + log_prior)
                
                next_i = i+len(batch_x)
                log_likelihoods[sample_i, i:next_i] = log_likelihood.cpu().detach()
                log_priors[sample_i, i:next_i] = log_prior.cpu().detach()
                potentials[sample_i, i:next_i] = potential.cpu().detach()
                i = next_i
    
    return log_likelihoods.numpy(), log_priors.numpy(), potentials.numpy()
    
def main():
    skip_first = 50
    batch_size = 128
    temps = [0.001, 0.01, 0.1, 1]
    exp_name = "exp_cifar10_depth20_width3_lr0.01_warmup45_cycles60_scale0"
    device = torch.device("cuda:0")  # can be CUDA instead
    
    sample_files = glob(f"/data2/users/yn621/cold-posterior-cnn/results/{exp_name}/*/samples.pt")
    # sample_files = [
    #     "/data2/users/yn621/cold-posterior-cnn/results/exp_cifar10_depth20_width3_lr0.01_warmup45_cycles60_scale0/1/samples.pt",
    #     "/data2/users/yn621/cold-posterior-cnn/results/exp_cifar10_depth20_width3_lr0.01_warmup45_cycles60_scale0/5/samples.pt",
    # ]
    
    lls = {str(temp) : [] for temp in temps}
    lps = {str(temp) : [] for temp in temps}
    ps = {str(temp) : [] for temp in temps}
    for sample_file in tqdm(sample_files):
        run_dir = Path(os.path.dirname(sample_file))
        samples = exp_utils.load_samples(sample_file, idx=np.s_[skip_first:])
        with h5py.File(f"{run_dir}/metrics.h5", "r") as metrics_file:
            exp_utils.reject_samples_(samples, metrics_file)
        del samples["steps"]
        del samples["timestamps"]
        for s in samples.items ():
            assert len(s)>0, f"we have less than {skip_first} samples"
            
        with open(os.path.join(run_dir, "config.json"), "r") as f:
            config = json.load(f)
            data = exp_utils.get_data(config["data"], "cpu")
            batch_size = min(batch_size, len(data.norm.test))
            dataloader = torch.utils.data.DataLoader(data.norm.test, batch_size=batch_size)
            model = exp_utils.get_model(x_train=data.norm.train_X, y_train=data.norm.train_y,
                                **{k: v for k, v in config.items() if k in set((
                                    "model",
                                    "width", "depth", "weight_prior", "weight_loc", "weight_scale",
                                    "bias_prior", "bias_loc", "bias_scale", "batchnorm",
                                    "weight_prior_params", "bias_prior_params"))})
            set_prior_scale(model, dataloader, scale_prior=int(config["scale_prior"]))
            model = model.to(device)
            model.eval()

            log_likelihoods, log_priors, potentials = evaluate_model(model, dataloader, samples)
            
        lls[str(config["temperature"])].extend(log_likelihoods[np.newaxis, :])
        lps[str(config["temperature"])].extend(log_priors[np.newaxis, :])
        ps[str(config["temperature"])].extend(potentials[np.newaxis, :])
    
    for temp in lls:
        lls[temp] = np.stack(lls[temp])
    for temp in lps:
        lps[temp] = np.stack(lps[temp])
    for temp in ps:
        ps[temp] = np.stack(ps[temp])
    
    # print log likelihood estimation
    print("\nRhat diagnostic for functions of log-likelihood")
    for temp in lls:
        data = arviz.convert_to_dataset(lls[temp])
        rhat = arviz.rhat(data)
        print(f"temperature {temp}: {rhat.mean()}")
        
    print("\nRhat diagnostic for functions of log-prior")
    for temp in lps:
        data = arviz.convert_to_dataset(lps[temp])
        rhat = arviz.rhat(data)
        print(f"temperature {temp}: {rhat.mean()}")
        
    print("\nRhat diagnostic for functions of potential")
    for temp in ps:
        data = arviz.convert_to_dataset(ps[temp])
        rhat = arviz.rhat(data)
        print(f"temperature {temp}: {rhat.mean()}")
    
if __name__ == "__main__":
    main()