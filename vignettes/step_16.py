import numpy as np
import pyro
import pyro.util
import torch
import tqdm
from pyro import poutine
from pyro.distributions import Gamma, Dirichlet
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from torch.distributions import biject_to

def model(expression_truth=None, N=None, J=None, K=None):
    if expression_truth is not None:
        N = expression_truth.shape[0]
        K = expression_truth.shape[1]
        J = expression_truth.shape[2]

    with pyro.plate("J", J):
        with pyro.plate("K", K):
            alpha = pyro.sample("alpha", Gamma(1, 1))

    with pyro.plate("N", N):
        sampled = pyro.sample("obs", Dirichlet(alpha).to_event(1), obs=expression_truth)

    return sampled


def generate_data(n_genes, n_cell_types, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    alpha = rng.gamma(1, 10, size=(n_cell_types, n_genes))

    result = np.empty((n_samples, n_cell_types, n_genes))

    for k in range(n_cell_types):
        alpha_k = alpha[k, :]
        expression_k = rng.dirichlet(alpha_k, size=n_samples)
        for i in range(n_samples):
            result[i, k, :] = expression_k[i, :]

    return alpha, result


def main():
    pyro.util.set_rng_seed(0)
    rng = np.random.default_rng(42)

    alpha_truth, data = generate_data(n_genes=20, n_cell_types=3, n_samples=10, rng=rng)

    data = torch.Tensor(data)

    optimizer = Adam(optim_args={"lr": 0.05})
    guide = AutoNormal(model)

    trace = poutine.trace(model).get_trace(data)
    trace.compute_log_prob()
    print("---------- Tensor Shapes ------------")
    print(trace.format_shapes())

    print(trace.log_prob_sum())

    pyro.render_model(model, model_args=(data,), filename="model.png")

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in tqdm.trange(1000):  # Consider running for more steps.
        loss = svi.step(data)

    alpha_loc = pyro.param("AutoNormal.locs.alpha")
    alpha_scale = pyro.param("AutoNormal.scales.alpha")

    # We might be surprised to see alpha_loc is negative sometimes! This is impossible for a Gamma distribution.

    # While its true gamma distributions have support only for positive numbers, we used an AutoNormal guide, which means
    # our variational family is unconstrained over all parameters. In order to make the autoguide line up with our model
    # pyro automatically applies transformations so the support of the parameters match the support of the distributions.
    # We can transform the parameters back to the original space by using the biject_to function and the support of the distribution
    # we used in our model, this gives us a function that will transform the parameters back to the original space.
    alpha_loc_transformed_back = biject_to(Gamma.support)(alpha_loc)

    create_imshow_comparison_of_numpy_arrays(
        alpha_truth, alpha_loc_transformed_back.detach().numpy(), alpha_scale.detach().numpy(), "step_16.png"
    )


def create_imshow_comparison_of_numpy_arrays(
    arr_truth, arr_estimate, arr_variance, fn: str
):
    """
    create 2 subplots each with an imshow
    # add colorbar
    # save imshow to fn
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # add titles to each subplot
    ax1.title.set_text("Truth")
    ax2.title.set_text("Estimate Mean")
    ax3.title.set_text("Estimate Var")

    im1 = ax1.imshow(arr_truth)
    im2 = ax2.imshow(arr_estimate)
    im3 = ax3.imshow(arr_variance)
    # add colorbar
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar3 = ax2.figure.colorbar(im3, ax=ax3)
    # save imshow to fn
    fig.savefig(fn)


if __name__ == "__main__":
    main()
