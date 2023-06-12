import pyro
import pyro.distributions as dist
import torch


def model():
    mu = pyro.param("mu", torch.tensor(0.0))
    y = pyro.sample("y", dist.Normal(mu, 1))
    return y


def main():
    one_sample = model()

    # one_sample is a torch.Tensor of length 1
    # We need to call some methods to get the
    # value out as a regular python number
    one_sample_as_a_number = one_sample.detach().numpy().item()

    print(one_sample_as_a_number)

    print([model().detach().numpy().item() for _ in range(10)])


if __name__ == "__main__":
    main()
