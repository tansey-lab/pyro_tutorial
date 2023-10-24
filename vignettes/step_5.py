import pyro
import pyro.distributions as dist
import torch

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

    print("The nodes in our model are:")
    for node in one_trace.nodes:
        print(node)

    pyro.render_model(
        model,
        model_args=(
            None,
            10,
        ),
        filename="step_5_model.png",
        render_params=True,
        render_distributions=True,
    )


if __name__ == "__main__":
    main()
