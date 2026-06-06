# Pomap: composable machine-learning model transformations

Pomap is a very simple domain specific language for building complex machine learning workflows from simple ones. Starting with your favourite machine learning models, you can use Pomap commands to:
- Build complex ensembles.
- Build complex cross validation and hyper-parametrisation procedures.
- Allow a model to create features or targets for another model.
- And any creative combination of the above.

Without making subsequent model evaluation, storage, or experimentation any more complex than it was with the original model. This implementation is built on top of the excellent Polars DataFrame library.


# Commands
Pomap has five commands:
- Lift: trains multiple copies of a model across different subsets of data.
- Ensemble: Takes a set of models and makes them evaluate as one.
- LearnsFrom: Allows a model to learn its hyperparameters from another.
- Feeds: Allows the output of one model to be used as a feature or target by another.
- Split: Trains a model on a given train/test/validation split.

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

