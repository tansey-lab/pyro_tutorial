import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.constraints import positive
from pyro.infer import HMC, MCMC


def model(data=None, n_obs=None):
    if data is None and n_obs is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        n_obs = data.shape[0]

    mu = pyro.sample("mu", dist.Gamma(1.0, 1.0))

    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def guide(data):
    mu_loc = pyro.param("mu_loc", torch.tensor(0.0))
    mu_scale = pyro.param("mu_scale", torch.tensor(1.0), constraint=positive)

    # Why cant we use normal?
    pyro.sample("mu", dist.LogNormal(mu_loc, mu_scale))


def generate_data(mu_truth, rng):
    return rng.normal(mu_truth, 1, size=100)


def main():
    rng = np.random.default_rng(0)
    data = torch.Tensor(generate_data(4.2, rng))
    mu_truth = data.mean().detach().numpy().item()

    print("The mean of our generated dataset is:")
    print(data.mean().detach().numpy().item())

    hmc_kernel = HMC(model)
    mcmc = MCMC(hmc_kernel, warmup_steps=100, num_samples=200)
    mcmc.run(data=data)
    mcmc_mean = mcmc.get_samples()["mu"].mean()
    print("mcmc_mean for mu: ", mcmc_mean, "mu_truth: ", mu_truth)


if __name__ == "__main__":
    main()