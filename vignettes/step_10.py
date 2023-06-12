import pyro
import pyro.distributions as dist
import torch
import numpy as np

from pyro import poutine
import torch.distributions


def model(data=None, n_obs=None):
    if data is None and n_obs is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        n_obs = data.shape[0]

    mu = pyro.sample("mu", dist.Normal(0, 0.3))

    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def generate_data(mu_truth, rng):
    return rng.normal(mu_truth, 1, size=10)


def main():
    learning_rate = 0.001
    mu_truth = 4.2
    np.random.default_rng(0)

    generated_data = torch.tensor(generate_data(mu_truth, np.random.default_rng()))
    print(generated_data.mean())

    mu_param = torch.tensor(0.0, requires_grad=True)

    for i in range(10):
        conditioned_model = poutine.condition(model, {"mu": mu_param})
        trace = poutine.trace(conditioned_model).get_trace(data=generated_data)
        loss = trace.log_prob_sum()
        loss.backward()
        gradient = mu_param.grad.detach()
        print(
            "loss:",
            loss.detach().numpy().item(),
            "mu:",
            mu_param.detach().numpy().item(),
            "gradient:",
            gradient.numpy().item(),
        )
        new_mu = mu_param.clone().detach() + (mu_param.grad.detach() * learning_rate)
        mu_param = torch.tensor(new_mu, requires_grad=True)


if __name__ == "__main__":
    main()
