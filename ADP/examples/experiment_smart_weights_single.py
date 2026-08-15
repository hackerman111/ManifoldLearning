if __package__:
    from .smart_weights_grid import make_experiment, run_file
else:
    from smart_weights_grid import make_experiment, run_file

experiment = make_experiment("single", 1)

if __name__ == "__main__":
    raise SystemExit(run_file(__file__))
