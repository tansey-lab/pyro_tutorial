# 0 to Pyro in 60 Seconds

This guide shows, in a series of vignettes that are monotonically increasing in complexity, how to use the
pyro probabilistic programming language.


## Step 1

Introducing our first random variable, using the `pyro.sample` primitive

[step_1.py](./vignettes/step_1.py)

## Step 2

Introducing the concept of observed values, via the `obs=` argument to `pyro.sample`

[step_2.py](./vignettes/step_2.py)

## Step 3

Introducing `pyro.plate`, a primitive for marking conditionally independent variables

[step_3.py](./vignettes/step_3.py)

## Step 4

Introducing `poutine.trace`, our first "effect handler"

[step_4.py](./vignettes/step_4.py)

## Step 5

Introducing the internal structure of `poutine.trace`, and how that structure can be used to calculate the 
log probability sum and also create a graphical model plate diagram

[step_5.py](./vignettes/step_5.py)

## Step 6

Combining observed values with calculating the log probability sum of our distribution

[step_6.py](./vignettes/step_6.py)

## Step 7

Demonstrating two separate random variables in one model, and how we can sample the individual values of each
using `poutine.trace`

[step_7.py](./vignettes/step_7.py)

## Step 8

Using `poutine.condition` to condition on the value of one random variable before we sample from our posterior,
and how this affects the log probability sum

[step_8.py](./vignettes/step_8.py)

## Step 9

Demonstrating how to look at the gradient at a given random variable in our model

[step_9.py](./vignettes/step_9.py)

## Step 10

Using this gradient to write a simple gradient descent algorithm to the find the MAP estimate of a random variable
in our model

[step_10.py](./vignettes/step_10.py)

## Step 11

Showing how the gradient descent algorithm laid out in step 10 fails 
if we have much more complicated, deeply nested hierarchical priors

[step_11.py](./vignettes/step_11.py)

## Step 12

Since we cannot optimize our loss function, the next best thing to try is an MCMC algorithm to sample from our posterior

[step_12.py](./vignettes/step_12.py)

## Step 13

The MCMC sampler works great but it is maybe too slow, so the next best thing to try is SVI. 

For the sake of simplicity we return to the simple model from step 10 and show how to run SVI on it using a manually
constructed guide

[step_13.py](./vignettes/step_13.py)

## Step 14

Demonstrating how to use the `AutoNormal` class to automatically construct a guide function for us, and we 
see the results are the same as our manually constructed guide.

[step_14.py](./vignettes/step_14.py)

## Step 15

Demonstrating nested plates, and the peculiarity of indexing into nested plates in pyro. We show two equivalent
ways to nest plates in pyro.

[step_15.py](./vignettes/step_15.py)

## ⚠️ Required Reading ⚠️

It will be very difficult to understand the next vignette unless you
understand the concept of "batch" and "event" dimensions.

This concept is explained very well by this blog post:

https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/

(you only need to read up to the "Other Scenarios" section)

## Step 16

Demonstrating the difference between batch and event dimensions with an example model that has to deal with both.

[step_16.py](./vignettes/step_16.py)

## Step 17

Introducing a model with a discrete latent variable, and how we can use TraceEnum_ELBO to marginalize over it,
and infer_discrete to create a classifier from our trained model.

[step_17.py](./vignettes/step_17.py)

## Step 18

Demonstrating that some local optima are better than others, and how our model from step_17 is sensitive to its
initialization values

[step_18.py](./vignettes/step_18.py)

## Step 19

Demonstrating how to plot the ELBO loss curve for a model.

Inspecting this curve can give you insight into if you need to run for more SVI steps and if your model is
converging.

[step_19.py](./vignettes/step_18.py)

# Other Resources

## Lecture from David Blei on Variational Inference

https://youtu.be/DaqNNLidswA

## MiniPyro

A contributor has made a completely stripped down implementation of the pyro framework 
that is only a few hundred lines of code and pretty closely emulates what the full version of pyro does

Reading through this implementation can give you deep insight into how pyro itself works and I highly recommend it
for advanced users:

https://github.com/pyro-ppl/pyro/blob/727aff741e105715840bfdafee5bfeda7e8b65e8/pyro/contrib/minipyro.py#L15
