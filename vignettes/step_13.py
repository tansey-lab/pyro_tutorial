import numpy as np
import pyro
import pyro.distributions as dist
import torch
import tqdm
from pyro.distributions.constraints import positive
from pyro.infer import SVI, Trace_ELBO
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


# Note that the function signature of the guide function should match the function signature
# of the model function
def guide(data=None, n_obs=None):
    mu_loc = pyro.param("mu_loc", torch.tensor(0.0))
    mu_scale = pyro.param("mu_scale", torch.tensor(1.0), constraint=positive)

    # Why cant we use normal?
    pyro.sample("mu", dist.LogNormal(mu_loc, mu_scale))


def generate_data(mu_truth, rng):
    return rng.normal(mu_truth, 1, size=10)


def main():
    pyro.util.set_rng_seed(0)
    rng = np.random.default_rng(0)
    mu_truth = 4.2

    data = generate_data(mu_truth, rng)

    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)

    manual_guide_svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in tqdm.trange(100):
        manual_guide_svi.step(torch.tensor(data, dtype=torch.float32))

    mu_loc = pyro.param("mu_loc").item()
    mu_scale = pyro.param("mu_scale").item()

    # We need to exp the mu_loc parameter since it is for a LogNormal distribution
    print("mu_truth: ", mu_truth, "mu_loc: ", np.exp(mu_loc), "mu_scale: ", mu_scale)


if __name__ == "__main__":
    main()
