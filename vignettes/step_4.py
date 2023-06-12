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
    one_trace = poutine.trace(model).get_trace(data=None, n_obs=10)

    # .nodes???
    sampled_y_vector = one_trace.nodes["y"]["value"].detach().numpy()

    print("Our sampled vector is:")
    print(sampled_y_vector)
    print("Mean of our sampled vector is:")
    print(np.mean(sampled_y_vector))

    print("The log_prob_sum of this sample is:")
    print(one_trace.log_prob_sum().detach().numpy().item())


if __name__ == "__main__":
    main()
