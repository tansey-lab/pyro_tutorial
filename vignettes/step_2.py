import pyro
import pyro.distributions as dist
import torch


def model(data=None):
    mu = pyro.param("mu", torch.tensor(0.))
    y = pyro.sample("y", dist.Normal(mu, 1), obs=data)
    return y


def main():
    one_sample = model(torch.tensor([4.2]))
    # RuntimeWarning: trying to observe a value outside of inference at y

    one_sample_as_a_number = one_sample.detach().numpy().item()

    print(one_sample_as_a_number)


if __name__ == "__main__":
    main()