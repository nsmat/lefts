# Installing lefts

```
pip install lefts
```

# What does Lefts do?

### Lefts commands transform machine learning models

To lefts, a machine learning model is a set of functions. We will introduce the full set of functions in the following section, but for now it suffices to focus on these:
- fit: which optimises model parameters over a training data set.
- predict: which generates predictions on test data.


Each Lefts command is a functor that transforms a model by acting on those functions to yield a new, transformed model. For example, if we have a Lefts command T, we can create a new model by using it to transform each function.

```
● ┌─ Model ─────────────┐              ┌─ Model ────────────────┐                                                                                                                                                                   
  │ .fit:     fitter    │  ────T────►  │ .fit:     T(fitter)    │                                                                                                                                                                   
  │ .predict: predictor │              │ .predict: T(predictor) │                                                                                                                                                                   
  └─────────────────────┘              └────────────────────────┘
 ```

The transformed model has the same interface, we can keep applying more lefts transformations to it to build up increasingly complex behaviour.

### Train, test, validation splits

Lefts 


### Model labels


# Understanding Lefts Models


