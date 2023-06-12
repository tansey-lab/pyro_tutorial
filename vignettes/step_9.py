import pyro
import pyro.distributions as dist
import torch

from pyro import poutine
import torch.distributions

def model(data=None, n_obs=None):
    if data is None and n_obs is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        n_obs = data.shape[0]

    mu = pyro.sample("mu", dist.Normal(0, 1))

    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def main():
    mu_param = torch.tensor(0., requires_grad=True)

    conditioned_model = poutine.condition(model, {'mu': mu_param})

    one_trace_from_conditioned_model = poutine.trace(conditioned_model).get_trace(data=None, n_obs=10)
    total_loss = 0.
    for k, site in one_trace_from_conditioned_model.nodes.items():
        if site["type"] == "sample":
            total_loss = total_loss + site["fn"].log_prob(site["value"]).sum()

    total_loss.backward()
    print(mu_param.grad)



if __name__ == "__main__":
    main()
