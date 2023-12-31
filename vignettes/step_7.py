import pyro
import pyro.distributions as dist

from pyro import poutine


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
    pyro.render_model(
        model,
        model_args=(
            None,
            10,
        ),
        filename="step_7_model.png",
    )

    one_trace = poutine.trace(model).get_trace(data=None, n_obs=10)

    sampled_y_vector = one_trace.nodes["y"]["value"].detach().numpy()
    sampled_mu_value = one_trace.nodes["mu"]["value"].detach().numpy()

    print("Sampled mu is:")
    print(sampled_mu_value)
    print("Sampled y is:")
    print(sampled_y_vector)

    log_prob_sum = one_trace.log_prob_sum()

    print("The log_prob_sum of this sample is:")
    print(log_prob_sum.detach().numpy().item())


if __name__ == "__main__":
    main()
