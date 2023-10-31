import numpy as np

def create_sequence(runner):
    # ----------- Get seq -----------#
    # For editing timesteps
    if runner.args.n_train_step != 0:
        seq_train = np.linspace(0, 1, runner.args.n_train_step) * runner.args.t_0
        seq_train = seq_train[seq_train >= runner.t_edit]
        seq_train = [int(s + 1e-6) for s in list(seq_train)]
        print("Uniform skip type")

    else:
        seq_train = list(range(runner.t_edit, runner.args.t_0))
        print("No skip")

    seq_train_next = [-1] + list(seq_train[:-1])

    # For sampling
    seq_test = np.linspace(0, 1, runner.args.n_test_step) * runner.args.t_0
    seq_test_edit = seq_test[seq_test >= runner.t_edit]
    seq_test_edit = [int(s + 1e-6) for s in list(seq_test_edit)]
    seq_test = [int(s + 1e-6) for s in list(seq_test)]
    seq_test_next = [-1] + list(seq_test[:-1])

    print(f"seq_train: {seq_train}")
    print(f"seq_test: {seq_test}")
    print(f"seq_test_edit: {seq_test_edit}")

    return seq_train, seq_train_next, seq_test, seq_test_next, seq_test_edit