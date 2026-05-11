# Pomap: composable machine-learning model transformations

Pomap is a very simple domain specific language for building complex machine learning workflows from simple ones. Starting with your favourite machine learning models, you can use Pomap commands to:
- Build complex ensembles.
- Build complex cross validation and hyper-parametrisation procedures.
- Create dependencies between models.
- Any combination of the above.

Without making subsequent model evaluation, storage, or experimentation any more complex than it was with the original model. This implementation is built on top of the excellent Polars DataFrame library.


# Commands
Pomap has four commands:
- Lift: trains multiple copies of a model across different subsets of data.
- Ensemble: Takes a set of models and makes them evaluate as one.
- LearnsFrom: Allows a model to learn its hyperparameters from another.
- Feeds (not implemented): Allows the output of one model to be used as a feature by another.

## Examples
TODO: move these to an API specification? Or just link to a tutorial?
### Lift

### Ensemble

### LearnsFrom

### Feeds

# Model management
### Accessing models
TODO: describe labels here. I want to revisit this first since they are still a bit clunky

### Manipulating models
TODO: not implemented - idea is you should be able to easily delete nodes etc.

### Saving models
TODO: Not implemented

# Models

Pomap can operate on any model that is defined by:
- a fit method, which maps from training data into the model parameters
- a predict method, which maps from model parameters and test data into predictions.

A Pomap command creates a new model by transforming these functions into a new .fit and .predict. Because this new model also has a .fit and .predict, it can be transformed with further Pomap commands.

### Pomap trees

TODO: write this 


### Conventions

Pomap imposes some constraints on model interfaces.
- All hyperparameters are passed as arguments to the fit method.
- We expect that data is passed to fit and predict as Polars dataframes.
- The predict method returns an iterable, with the order of predictions matching the order on the input training data frame.

