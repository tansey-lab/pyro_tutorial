import pyro.distributions as dist
import pyro
import pyro.util
from pyro import poutine


def model(data=None, N=None, J=None):
    if data is None and N is None and J is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        N = data.shape[0]
        J = data.shape[1]

    with pyro.plate("J", J):
        mu = pyro.sample("mu", dist.Gamma(1., 1.))
        with pyro.plate("N", N):
            y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def model_v2(data=None, N=None, J=None):
    if data is None and N is None and J is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        N = data.shape[0]
        J = data.shape[1]

    with pyro.plate("J", J):
        mu = pyro.sample("mu", dist.Gamma(1., 1.))

    with pyro.plate("N", N, dim=-2):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


def main():
    pyro.util.set_rng_seed(0)
    print(model(data=None, N=10, J=3))
    trace1 = poutine.trace(model).get_trace(data=None, N=10, J=3)

    pyro.util.set_rng_seed(0)
    print(model(data=None, N=10, J=3))
    trace2 = poutine.trace(model_v2).get_trace(data=None, N=10, J=3)


if __name__ == "__main__":
    main()