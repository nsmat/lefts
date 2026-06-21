# Commands

Lefts has six commands. Each takes a `Model` (or a model constructor) and returns a new `Model` with the same `.fit` / `.predict` interface.

| Command | Short description |
|---------|------------------|
| [`leaf`](leaf.md) | Entry point — wraps a model factory |
| [`lift`](lift.md) | One model → family of models over a list of values |
| [`ensemble`](ensemble.md) | Many models → one combined model |
| [`tune`](tune.md) | Source model learns hyperparameters for consumer model |
| [`feed`](feed.md) | Source predictions become features/targets for consumer |
| [`split`](split.md) | Fixed train / test / validation filters |

Because every command returns a `Model`, they compose freely:

```
ensemble
  └── lift
        └── tune
              ├── source: leaf
              └── consumer: leaf
```
