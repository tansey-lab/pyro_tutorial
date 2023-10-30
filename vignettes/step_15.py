import numpy as np
import pyro
import pyro.distributions as dist
import pyro.util
from pyro import poutine


# This is the kind of indexing that Pyro nested plates DO NOT use
def the_indexing_we_know_and_love():
    result = np.zeros((10, 3))
    for i in range(10):
        for j in range(3):
            # In pyro world this line would be result[j, i] (indexing from right)
            result[i, j] = i + j
    print(result)


# model and model_v2 show two different ways of nesting plates
def model_nested_with_blocks(data=None, N=None, G=None):
    if data is None and N is None and G is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        N = data.shape[0]
        G = data.shape[1]

    with pyro.plate("G", G):
        mu = pyro.sample("mu", dist.Gamma(1.0, 1.0))
        with pyro.plate("N", N):
            y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


# This is the alternate way of nesting plates
# We explicitly use the dim argument to tell Pyro which plate is the outer and which is the inner
def model_with_explicit_plate_dimensions(data=None, N=None, G=None):
    if data is None and N is None and G is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        N = data.shape[0]
        G = data.shape[1]

    with pyro.plate("G", G):
        mu = pyro.sample("mu", dist.Gamma(1.0, 1.0))

    with pyro.plate("N", N, dim=-2):
        y = pyro.sample("y", dist.Normal(mu, 1), obs=data)

    return y


# In this model we define the plates as variables so we can use them multiple times.
# We also show having per cell AND per gene parameters
def model_with_reusable_plates(data=None, N=None, G=None):
    if data is None and N is None and G is None:
        raise ValueError("Someone has gotta tell us how many observations there are")

    if data is not None:
        N = data.shape[0]
        G = data.shape[1]

    # In this case where we don't use explicit nesting of plates, we need to tell Pyro
    # the order of the plates. This is what the `dim` argument does here.
    # As we know in python negative numbers index from the right size. So dim=-1 means
    # the last dimension, and -2 means the second from last dimension.
    # so defining gene_plate with dim -2 and cell_plate with dim -1 means that the
    # expression matrix will be of shape (N, G)

    # In `model_nested_with_blocks` we didn't need dim because the nesting of the with blocks essentially
    # gives pyro enough information to figure this out.
    gene_plate = pyro.plate("G", G, dim=-1)
    cell_plate = pyro.plate("N", N, dim=-2)

    with gene_plate:
        mu = pyro.sample("mu", dist.Gamma(1.0, 1.0))

    with cell_plate:
        nu = pyro.sample("nu", dist.Gamma(1.0, 1.0))

    with cell_plate, gene_plate:
        y = pyro.sample("y", dist.Normal(mu, nu), obs=data)

    return y


def main():
    the_indexing_we_know_and_love()
    pyro.render_model(
        model_nested_with_blocks,
        model_args=(
            None,
            10,
            3
        ),
        filename="step_15_model.png",
    )

    pyro.render_model(
        model_with_explicit_plate_dimensions,
        model_args=(
            None,
            10,
            3
        ),
        filename="step_15_model_with_explicit_plate_dimensions.png",
    )

    pyro.render_model(
        model_with_reusable_plates,
        model_args=(
            None,
            10,
            3
        ),
        filename="step_15_model_with_reusable_plates.png",
    )

    pyro.util.set_rng_seed(0)
    print("Shape of model_nested_with_blocks output:")
    print(model_nested_with_blocks(data=None, N=10, G=3).shape)
    trace1 = poutine.trace(model_nested_with_blocks).get_trace(data=None, N=10, G=3)

    pyro.util.set_rng_seed(0)
    print("Shape of model_with_explicit_plate_dimensions output:")
    print(model_with_explicit_plate_dimensions(data=None, N=10, G=3).shape)
    trace2 = poutine.trace(model_with_explicit_plate_dimensions).get_trace(data=None, N=10, G=3)

    pyro.util.set_rng_seed(0)
    print("Shape of model_with_reusable_plates output:")
    print(model_with_reusable_plates(data=None, N=10, G=3).shape)
    trace3 = poutine.trace(model_with_reusable_plates).get_trace(data=None, N=10, G=3)


if __name__ == "__main__":
    main()
