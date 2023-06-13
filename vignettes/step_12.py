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

    # Let's make our prior alot more complicated!
    alpha = pyro.sample(
        "alpha", dist.InverseGamma(torch.Tensor([1.0]), torch.Tensor([1.0]))
    )

    a = pyro.sample("a", dist.Gamma(alpha, 1.0))
    b = pyro.sample("b", dist.Gamma(1.0, 1.0))

    mu = pyro.sample("mu", dist.Gamma(a, b))
    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def generate_data(mu_truth, rng):
    return rng.normal(mu_truth, 1, size=10)


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
