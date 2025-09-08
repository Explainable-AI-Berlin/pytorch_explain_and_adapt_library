import sys
import pdb
import traceback
import torch
import signal


def register_pdb_hook():
    previous_hook = sys.excepthook

    def custom_exception_hook(exctype, value, tb):
        if previous_hook is not None:
            previous_hook(exctype, value, tb)
        # Print the exception traceback
        traceback.print_exception(exctype, value, tb)

        # Start the debugger in post-mortem mode
        pdb.post_mortem(tb)

    print("THESIS DEBUG: Registered pdb hook")
    sys.excepthook = custom_exception_hook


def start_cuda_memory_recording(max_entries=100_000):
    torch.cuda.memory._record_memory_history(max_entries=max_entries)


def stop_cuda_memory_recording():
    torch.cuda.memory._record_memory_history(enabled=None)


def dump_cuda_snapshot(snapshot_name="/tmp/cuda_snapshot.pickle"):
    print(f"THESIS DEBUG: Dumping CUDA snapshot {snapshot_name}...")
    torch.cuda.memory._dump_snapshot(snapshot_name)


def register_cuda_snapshot_hook(snapshot_name: str = "/tmp/cuda_snapshot.pickle"):
    previous_hook = sys.excepthook

    def custom_exception_hook(exctype, value, tb):
        print("THESIS DEBUG: Exception caught, dumping CUDA snapshot")
        dump_cuda_snapshot(snapshot_name)
        if previous_hook is not None:
            previous_hook(exctype, value, tb)

    print("THESIS DEBUG: Registered CUDA snapshot hook")
    sys.excepthook = custom_exception_hook


USR1_SIGNAL_RECEIVED = False


def setup_usr1_signal_handler():
    # See https://services.criann.fr/en/services/hpc/mesonet-project/guide/signals-sent-by-slurm/
    def handle_sigusr1(signum, frame):
        global USR1_SIGNAL_RECEIVED
        print("THESIS DEBUG: Received SIGUSR1 signal.")
        USR1_SIGNAL_RECEIVED = True

    print("THESIS DEBUG: Setting up SIGUSR1 signal handler")
    signal.signal(signal.SIGUSR1, handle_sigusr1)


def usr1_signal_received():
    global USR1_SIGNAL_RECEIVED
    return USR1_SIGNAL_RECEIVED
