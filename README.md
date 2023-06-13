# 0 to Pyro in 60 Seconds

This guide shows, in a series of vignettes that are monotonically increasing in complexity, how to use the
pyro probabilistic programming language.


## Step 1

Introducing our first random variable, using the `pyro.sample` primitive

[./vignettes/step_1.py](step_1.py)

## Step 2

Introducing the concept of observed values, via the `obs=` argument to `pyro.sample`

[./vignettes/step_2.py](step_2.py)

## Step 3

Introducing `pyro.plate`, a primitive for marking conditionally independent variables

[./vignettes/step_3.py](step_3.py)

## Step 4

Introducing `poutine.trace`, a tool for calculating the log probability sum of 
an observation from our posterior distribution

[./vignettes/step_4.py](step_4.py)

## Step 5

Introducing the internal structure of `poutine.trace`, and how that structure can be used to calculate the 
log probability and also create a graphical model plate diagram

[./vignettes/step_5.py](step_5.py)

## Step 6

Combining observed values with calculating the log probability sum of our distribution

[./vignettes/step_6.py](step_6.py)

## Step 7

Demonstrating two separate random variables in one model, and how we can sample the individual values of each
using `poutine.trace`

[./vignettes/step_7.py](step_7.py)

## Step 8

Using `poutine.condition` to condition on the value of one random variable before we sample from our posterior,
and how this affects the log probability sum

[./vignettes/step_8.py](step_8.py)

## Step 9

Demonstrating how to look at the gradient at a given random variable in our model

[./vignettes/step_9.py](step_9.py)

## Step 10

Using this gradient to write a simple gradient descent algorithm to the find the MAP estimate of a random variable
in our model

[./vignettes/step_10.py](step_10.py)

## Step 11

Showing how the process laid out in step 10 fails if we have non-conjugate priors

[./vignettes/step_11.py](step_11.py)

## Step 12

Since we cannot optimize our loss function, the next best thing to try is an MCMC algorithm to sample from our posterior

[./vignettes/step_12.py](step_12.py)

## Step 13

Since our MCMC sampler is too slow, the next best thing to try is SVI. This step demonstrated SVI using a manually
constructed guide

[./vignettes/step_13.py](step_13.py)

## Step 14

Demonstrating how to use the `AutoNormal` class to automatically construct a guide function for us, and we 
see the results are the same as our manually constructed guide.

[./vignettes/step_14.py](step_14.py)

## Step 15

Demonstrating nested plates, and the peculiarity of indexing into nested plates in pyro

[./vignettes/step_15.py](step_15.py)

## Step 16

Demonstrating the difference between batch and event dimensions with an example model that has to deal with both

[./vignettes/step_16.py](step_16.py)

## Step 17

Introducing a model with a discrete latent variable, and how we can use TraceEnum_ELBO to marginalize over it,
and infer_discrete to create a classifier from our trained model.

[./vignettes/step_17.py](step_17.py)

## Step 18

Demonstrating that some local optima are better than others, and how our model from step_17 is sensitive to its
initialization values

[./vignettes/step_18.py](step_18.py)

