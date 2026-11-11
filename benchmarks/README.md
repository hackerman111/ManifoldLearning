# Benchmarks and experiment results

- [`experiments/`](experiments/README.md) contains experiment modules and their
  protocols. The modules remain importable as `experiments.*`; run them with
  `python -m experiments.<module>` from the repository root.
- The scripts directly in this directory contain focused benchmark and audit
  drivers.
- [`results/`](results/README.md) stores generated benchmark outputs and frozen
  verification campaigns.

The live estimator and solver implementation remains in [`../ADP/`](../ADP/).
