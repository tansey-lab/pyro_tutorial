import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.util
import seaborn as sns
import torch
import tqdm
from pyro import poutine
from pyro.distributions.transforms import OrderedTransform
from pyro.infer import SVI, TraceEnum_ELBO, infer_discrete
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.metrics import confusion_matrix


def plot_histogram(arr1, arr2, output_fn: str):
    """
    Create a histogram for the two arrays overlapping in different colors and save to file
    """
    plt.figure(figsize=(10, 10))
    sns.histplot(arr1, color="blue", label="observed")
    sns.histplot(arr2, color="orange", label="predicted")
    plt.legend()
    plt.savefig(output_fn)


def plot_confusion_matrix(arr_observed, arr_predicted, output_fn: str):
    """
    Create a confusion matrix for the two provided arrays using matplotlib and save to a file.
    """
    cm = confusion_matrix(arr_observed, arr_predicted)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(output_fn)


def generate_data(n_samples, K, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    scale = rng.lognormal(0, 2)
    weights = rng.dirichlet(2 * np.ones(K))
    locs = rng.normal(0, 10, size=K)

    result = np.empty(n_samples)
    assignments = np.empty(n_samples)

    for n in tqdm.trange(n_samples):
        z = rng.choice(np.arange(K), p=weights)

        result[n] = rng.normal(locs[z], scale)
        assignments[n] = z

    return result, scale, weights, locs, assignments


def model(K, data=None, n_obs=None):
    if data is not None:
        n_obs = len(data)
    # Global variables.
    weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample("scale", dist.LogNormal(0.0, 2.0))

    locs = pyro.sample("locs", dist.Normal(0.0, 10.0).expand([K]).to_event(1))

    ordered_transform = OrderedTransform()

    locs = ordered_transform(locs)

    with pyro.plate("data", n_obs):
        # Local variables.
        assignment = pyro.sample(
            "assignment", dist.Categorical(weights), infer={"enumerate": "parallel"}
        )
        return pyro.sample("obs", dist.Normal(locs[assignment], scale), obs=data)


def main():
    pyro.util.set_rng_seed(0)
    rng = np.random.default_rng(66)

    data_np, scale_truth, weights_truth, locs_truth, assignments_truth = generate_data(
        100, 2, rng
    )

    data = torch.tensor(data_np, dtype=torch.float32)

    K = 2  # Fixed number of components.

    trace = poutine.trace(poutine.enum(model, first_available_dim=-2)).get_trace(K)
    trace.compute_log_prob()
    print("---------- Tensor Shapes ------------")
    print(trace.format_shapes())

    optim = pyro.optim.Adam({"lr": 0.1})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    guide = AutoNormal(poutine.block(model, hide=["assignment"]))

    svi = SVI(model, guide, optim, elbo)

    losses = []
    for _ in tqdm.trange(200):
        loss = svi.step(K, data)
        losses.append(loss)

    print(
        "locs truth",
        locs_truth,
        "SVI locs",
        pyro.param("AutoNormal.locs.locs").detach().numpy(),
    )

    guide_trace = poutine.trace(guide).get_trace(K, data)  # record the globals
    trained_model = poutine.replay(model, trace=guide_trace)  # replay the globals

    def classifier(K, data):
        inferred_model = infer_discrete(
            trained_model, temperature=0, first_available_dim=-2
        )  # avoid conflict with data plate
        trace = poutine.trace(inferred_model).get_trace(K, data)
        return trace.nodes["assignment"]["value"]

    plot_confusion_matrix(
        assignments_truth, classifier(K, data), "confusion_matrix.png"
    )

    sample = trained_model(K=2, data=None, n_obs=100)

    plot_histogram(data_np, sample.detach().numpy(), "true_vs_sampled_distribution.png")


if __name__ == "__main__":
    main()
