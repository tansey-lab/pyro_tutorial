import pyro
import pyro.distributions as dist
import torch
import numpy as np

from pyro import poutine

PRIOR_BELIEF_IN_MU = 0.0


def model(data=None, n_obs=None):
    if data is None and n_obs is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        n_obs = data.shape[0]

    mu = pyro.param("mu", torch.tensor(PRIOR_BELIEF_IN_MU))

    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def main():
    good_data = torch.Tensor(
        np.random.default_rng(42).normal(PRIOR_BELIEF_IN_MU, 1, size=1000)
    )
    bad_data = torch.Tensor(
        np.random.default_rng(42).normal(PRIOR_BELIEF_IN_MU + 1, 1, size=1000)
    )

    good_trace = poutine.trace(model).get_trace(data=good_data, n_obs=10)
    bad_trace = poutine.trace(model).get_trace(data=bad_data, n_obs=10)

    print(
        "Good trace log prob sum:",
        -1 * good_trace.log_prob_sum().detach().numpy().item(),
    )
    print(
        "Bad trace log prob sum:", -1 * bad_trace.log_prob_sum().detach().numpy().item()
    )

    even_worse_data = torch.Tensor(
        np.random.default_rng(42).normal(PRIOR_BELIEF_IN_MU + 2, 1, size=1000)
    )
    even_worse_trace = poutine.trace(model).get_trace(data=even_worse_data, n_obs=10)

    print(
        "Even worse trace log prob sum:",
        -1 * even_worse_trace.log_prob_sum().detach().numpy().item(),
    )


if __name__ == "__main__":
    main()
