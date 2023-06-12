import pyro
import pyro.distributions as dist
import torch


def model(data=None, n_obs=None):
    if data is None and n_obs is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        n_obs = data.shape[0]

    mu = pyro.param("mu", torch.tensor(0.0))
    with pyro.plate("N", n_obs):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def main():
    one_sample = model(data=None, n_obs=10)

    one_sample_as_numbers = one_sample.detach().numpy()

    print("One sample (of 10):")
    print(one_sample_as_numbers)

    one_observed_sample = model(data=torch.tensor([1.0, 2.0, 3.0]), n_obs=None)

    one_observed_sample_as_numbers = one_observed_sample.detach().numpy()

    print("One observed sample (of 3):")
    print(one_observed_sample_as_numbers)


if __name__ == "__main__":
    main()
