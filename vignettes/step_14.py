import numpy as np
import pyro
import pyro.distributions as dist
import pyro.util
import torch
import tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


def model(data=None, n_obs=None):
    if data is None and n_obs is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        n_obs = data.shape[0]

    mu = pyro.sample("mu", dist.Gamma(1.0, 1.0))

    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def generate_data(mu_truth, rng):
    return rng.normal(mu_truth, 1, size=100)


def main():
    pyro.util.set_rng_seed(0)
    rng = np.random.default_rng(0)
    mu_truth = 4.2

    data = generate_data(mu_truth, rng)

    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)

    autoguide = AutoNormal(model)

    svi = SVI(model, autoguide, optimizer, loss=Trace_ELBO())

    for step in tqdm.trange(1000):
        svi.step(torch.tensor(data, dtype=torch.float32))

    mu_loc = pyro.param("AutoNormal.locs.mu").item()
    mu_scale = pyro.param("AutoNormal.scales.mu").item()
    print("mu_truth: ", mu_truth, "mu_loc: ", mu_loc, "mu_scale: ", mu_scale)


if __name__ == "__main__":
    main()
