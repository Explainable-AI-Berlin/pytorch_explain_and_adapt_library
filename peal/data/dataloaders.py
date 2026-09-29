"""Dataloader construction and mixing for PEAL training loops.

``get_dataloader`` and ``create_dataloaders_from_datasource`` turn a config
plus a dataset path, dataset tuple or dataloader tuple into the
train/val/test dataloaders used by trainers and adaptors. ``DataloaderMixer``
is the training-side wrapper that samples batches from several dataloaders
with priorities (e.g. original data plus counterfactuals appended by CFKD),
supports concatenated class-balanced batches and epochs of a fixed number of
steps; ``DataStack`` and ``create_class_ordered_batch`` build small
class-ordered batches for visualizations.
"""

import copy
import os

import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.multiprocessing
from peal.log import get_logger

_log = get_logger(__name__)


torch.multiprocessing.set_sharing_strategy("file_system")

from peal.data.dataset_factory import get_datasets
from peal.global_utils import load_yaml_config
from peal.training.interfaces import TrainingConfig, TaskConfig


class DataStack:
    """
    Per-class FIFO buffer of samples drawn from a dataset or dataloader.

    ``data[c]`` holds ``[X, y]`` pairs of class ``c``. The stack is refilled
    from the source whenever some class runs empty, so ``pop(c)`` always
    returns a sample of class ``c``; it is used to assemble class-ordered
    batches for visualizations.

    Parameters
    ----------
    datasource : torch.utils.data.Dataset or DataLoader or DataloaderMixer
        Where samples come from. A dataset is walked sequentially and
        cyclically via ``current_idx``; a dataloader is drawn from with
        ``next(iter(...))``.
    num_classes : int
        Number of classes, i.e. number of per-class lists.
    transform : callable, optional
        Transform temporarily swapped into ``dataset.transform`` while the
        stack is being filled.
    """

    def __init__(self, datasource, num_classes, transform=None):
        """Create the per-class lists and fill them once."""
        self.datasource = datasource
        if isinstance(datasource, torch.utils.data.Dataset):
            self.dataset = datasource
            self.current_idx = 0

        else:
            self.dataset = datasource.dataset

        self.num_classes = num_classes
        self.data = []
        for idx in range(num_classes):
            self.data.append([])

        self.transform = transform
        self.fill_stack()

    def fill_stack(self):
        """
        Draw samples from the source until every class list is non-empty.

        With hints or indices enabled the label is a tuple and its first
        element is used as the class index.
        """
        if not self.transform is None:
            data_transform = self.dataset.transform
            self.dataset.transform = self.transform

        while np.min(list(map(lambda x: len(x), self.data))) == 0:
            if isinstance(self.datasource, torch.utils.data.Dataset):
                X, y = self.dataset.__getitem__(self.current_idx)
                if (
                    hasattr(self.dataset, "hints_enabled")
                    and self.dataset.hints_enabled
                    or hasattr(self.dataset, "idx_enabled")
                    and self.dataset.idx_enabled
                ):
                    y_index = y[0]

                else:
                    y_index = y

                self.data[int(y_index)].append([X, y])
                self.current_idx = (self.current_idx + 1) % self.dataset.__len__()

            else:
                X, y = next(iter(self.datasource))
                if isinstance(y, int):
                    X, y = X

                if (
                    isinstance(y, list)
                    or hasattr(self.dataset, "hints_enabled")
                    and self.dataset.hints_enabled
                    or hasattr(self.dataset, "idx_enabled")
                    and self.dataset.idx_enabled
                ):
                    for i in range(X.shape[0]):
                        try:
                            y_out = tuple([y_elem[i] for y_elem in y])

                        except Exception:
                            raise

                        self.data[int(y[0][i])].append([X[i], y_out])

                else:
                    for i in range(X.shape[0]):
                        self.data[int(y[i])].append([X[i], int(y[i])])

        if not self.transform is None:
            self.dataset.transform = data_transform

    def pop(self, class_idx):
        """
        Remove and return the oldest buffered sample of one class.

        Parameters
        ----------
        class_idx : int
            Class whose sample is popped.

        Returns
        -------
        list
            ``[X, y]`` of the sample; the stack is refilled afterwards.
        """
        sample = self.data[class_idx].pop(0)
        self.fill_stack()
        return sample

    def reset(self):
        """Empty all class lists, reset a ``DataloaderMixer`` source and refill."""
        self.data = []
        for idx in range(self.num_classes):
            self.data.append([])

        # TODO if no mixer is used here this will not work
        if isinstance(self.datasource, DataloaderMixer):
            self.datasource.reset()

        self.fill_stack()


class DataIterator:
    """
    Iterator over a ``DataloaderMixer`` yielding a fixed number of batches.

    One pass yields ``dataloader.train_config.steps_per_epoch`` batches
    obtained from ``dataloader.sample()``, independent of the underlying
    dataset sizes.

    Parameters
    ----------
    dataloader : DataloaderMixer
        The mixer to draw batches from.
    """

    def __init__(self, dataloader):
        """Store the mixer and start the step counter at 0."""
        self.dataloader = dataloader
        # member variable to keep track of current index
        self._index = 0

    def __next__(self):
        """Return the next mixed batch or raise ``StopIteration`` after the epoch."""
        if self._index < self.dataloader.train_config.steps_per_epoch:
            self._index += 1
            return self.dataloader.sample()

        # End of Iteration
        raise StopIteration


class DataloaderMixer(DataLoader):
    """
    Dataloader that samples batches from several dataloaders.

    Starts with one dataloader; ``append`` adds more (e.g. counterfactual
    datasets during CFKD) with priorities proportional to dataset size. In
    the default mode every ``sample()`` picks one dataloader by multinomial
    draw over ``priorities``; with ``train_config.concatenate_batches`` one
    batch from each dataloader is concatenated instead (used for class
    balancing). Exhausted iterators are restarted transparently, and one
    epoch is ``train_config.steps_per_epoch`` batches. Only the attribute
    interface of ``DataLoader`` is reused; ``DataLoader.__init__`` is not
    called.

    Parameters
    ----------
    train_config : TrainingConfig
        Provides ``steps_per_epoch`` and optionally ``concatenate_batches``.
    initial_dataloader : DataLoader
        First member; its ``batch_size`` and ``dataset`` are adopted.
    return_src : bool, optional
        If ``True`` (and batches are not concatenated) ``sample`` returns
        ``(batch, dataloader_index)``.

    Attributes
    ----------
    dataloaders : list of DataLoader
    iterators : list
        One live iterator per dataloader.
    priorities : numpy.ndarray or None
        Sampling probabilities, ``None`` while only one dataloader exists.
    hints_enabled, class_balancing_enabled : bool
    """

    def __init__(self, train_config, initial_dataloader, return_src=False):
        """Register the first dataloader (``DataLoader.__init__`` is skipped)."""
        self.train_config = train_config
        self.dataloaders = [initial_dataloader]
        self.batch_size = initial_dataloader.batch_size
        self.priorities = None
        self.dataset = initial_dataloader.dataset  # TODO kind of hacky
        self.iterators = [iter(self.dataloaders[0])]
        self.return_src_internal = return_src
        self.hints_enabled = False
        self.class_balancing_enabled = False

    def update_dataset(self, dataset):
        """Re-point ``self.dataset`` at the first member's dataset (arg ignored)."""
        self.dataset = self.dataloaders[0].dataset

    def __getstate__(self):
        """Pickle everything except the live iterators."""
        return {
            "train_config": self.train_config,
            "dataloaders": self.dataloaders,
            "batch_size": self.batch_size,
            "priorities": self.priorities,
            "dataset": self.dataset,
            "return_src_internal": self.return_src_internal,
            "hints_enabled": self.hints_enabled,
            "class_balancing_enabled": self.class_balancing_enabled,
        }

    def __setstate__(self, state):
        """Restore the pickled attributes and rebuild the iterators via ``reset``."""
        self.train_config = state["train_config"]
        self.dataloaders = state["dataloaders"]
        self.batch_size = state["batch_size"]
        self.priorities = state["priorities"]
        self.dataset = state["dataset"]
        self.return_src_internal = state["return_src_internal"]
        self.hints_enabled = state["hints_enabled"]
        self.class_balancing_enabled = state["class_balancing_enabled"]
        self.iterators = [None for it in range(len(self.dataloaders))]
        self.reset()

    @property
    def return_src(self):
        """Whether ``sample`` also returns the source index (off when concatenating)."""
        if (
            hasattr(self.train_config, "concatenate_batches")
            and self.train_config.concatenate_batches
        ):
            return False

        else:
            return self.return_src_internal

    def append(self, dataloader, priority=1, weight_added_dataloader=None):
        """
        Add another dataloader and recompute the sampling priorities.

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader to add.
        priority : float, optional
            Multiplier on the new dataloader's dataset size before the
            priorities are normalized. Defaults to 1.
        weight_added_dataloader : float, optional
            If given, priorities become ``[1 - w, w]`` regardless of sizes
            (only meaningful with exactly two dataloaders).

        Notes
        -----
        With ``train_config.concatenate_batches`` all members are reset to
        half the batch size so concatenated batches keep the original size.
        """
        self.dataloaders.append(dataloader)
        self.iterators.append(iter(self.dataloaders[-1]))
        if weight_added_dataloader is None:
            self.priorities = np.zeros(len(self.dataloaders))
            for i in range(len(self.dataloaders)):
                self.priorities[i] = self.dataloaders[i].dataset.__len__()

            self.priorities[-1] *= priority
            self.priorities = self.priorities / self.priorities.sum()

        else:
            self.priorities = np.array(
                [1 - weight_added_dataloader, weight_added_dataloader]
            )

        if (
            hasattr(self.train_config, "concatenate_batches")
            and self.train_config.concatenate_batches
        ):
            self.reset(batch_size=self.batch_size // 2)

    def __iter__(self):
        """Return a ``DataIterator`` over ``steps_per_epoch`` mixed batches."""
        return DataIterator(self)

    # def return_iter(self, dataloader):
    #     """Recursively reset dataloader and return fresh iterator.

    #     Args:
    #         dataloader: The dataloader to reset (can be DataloaderMixer or DataLoader)

    #     Returns:
    #         A fresh iterator from the reset dataloader
    #     """
    #     if isinstance(dataloader, DataloaderMixer):
    #         # Recursively reset all nested dataloaders
    #         for nested_dl in dataloader.dataloaders:
    #             if isinstance(nested_dl, DataloaderMixer):
    #                 nested_dl.reset()
    #         # Reset this dataloader's iterators
    #         dataloader.reset()
    #     return iter(dataloader)

    def sample(self):
        """
        Draw one batch.

        In priority mode one dataloader is chosen by multinomial draw and
        its next batch returned; in concatenation mode the next batch of
        every dataloader is fetched and the tensors (also inside nested
        lists/tuples) are concatenated along dim 0. Exhausted iterators are
        restarted once; a dataloader that is still empty afterwards raises
        ``StopIteration``.

        Returns
        -------
        object
            The batch as produced by the member dataloader(s), or
            ``(batch, source_index)`` when ``return_src`` is set.
        """
        if (
            not hasattr(self.train_config, "concatenate_batches")
            or not self.train_config.concatenate_batches
            or len(self.dataloaders) == 1
        ):
            if not self.priorities is None:
                idx = int(np.random.multinomial(1, self.priorities).argmax())

            else:
                idx = 0

            item = next(self.iterators[idx], "STOP")
            if isinstance(item, str) and item == "STOP":
                self.iterators[idx] = iter(self.dataloaders[idx])
                item = next(self.iterators[idx], "STOP")
                # If still STOP after reset, dataloader is empty - raise StopIteration
                if isinstance(item, str) and item == "STOP":
                    raise StopIteration

            if self.return_src:
                item = (item, idx)

        else:
            subitems = []
            for idx in range(len(self.iterators)):
                item = next(self.iterators[idx], "STOP")
                if isinstance(item, str) and item == "STOP":
                    self.iterators[idx] = iter(self.dataloaders[idx])
                    item = next(self.iterators[idx], "STOP")
                    # If still STOP after reset, dataloader is empty - raise StopIteration
                    if isinstance(item, str) and item == "STOP":
                        raise StopIteration

                subitems.append(item)

            if not subitems:
                # No iterators at all: every constituent dataloader was dropped as empty
                # (class-balanced training splits the counterfactual set per class, and a
                # ~40-row set whose "false" verdicts all land in one class leaves the other
                # side empty). The two branches above already signal exhaustion with
                # StopIteration; do the same here instead of IndexError on subitems[0].
                raise StopIteration

            item = subitems[0]

            for subitem in subitems[1:]:
                for i in range(len(item)):
                    if isinstance(item[i], list) or isinstance(item[i], tuple):
                        for j in range(len(item[i])):
                            item[i][j] = torch.cat([item[i][j], subitem[i][j]], dim=0)

                    else:
                        item[i] = torch.cat([item[i], subitem[i]], dim=0)

            if self.return_src:
                item = (item, 0)

        return item

    def reset(self, batch_size=None):
        """
        Rebuild every member dataloader and its iterator.

        Parameters
        ----------
        batch_size : int, optional
            New batch size for the rebuilt ``DataLoader`` objects; nested
            mixers are reset recursively with the same value.
        """
        for i in range(len(self.dataloaders)):
            if isinstance(self.dataloaders[i], DataloaderMixer):
                self.dataloaders[i].reset(batch_size)

            else:
                new_batch_size = (
                    batch_size if batch_size else self.dataloaders[i].batch_size
                )
                self.dataloaders[i] = DataLoader(
                    self.dataloaders[i].dataset, batch_size=new_batch_size
                )

            self.iterators[i] = iter(self.dataloaders[i])

    def remove_empty_dataloaders(self):
        """Recursively remove dataloaders with zero-length datasets.

        This method iterates through all nested dataloaders and removes any
        that have no data (len(dataset) == 0), including nested DataloaderMixers
        that become empty after recursive cleaning.
        """
        # First, recursively clean nested DataloaderMixers
        for dataloader in self.dataloaders:
            if isinstance(dataloader, DataloaderMixer):
                dataloader.remove_empty_dataloaders()

        # Track indices to remove (iterate backwards to avoid index shift)
        indices_to_remove = []
        for i in range(len(self.dataloaders) - 1, -1, -1):
            dataloader = self.dataloaders[i]
            # Check if dataset is empty
            if len(dataloader.dataset) == 0:
                indices_to_remove.append(i)

        # Remove empty dataloaders and their iterators
        for idx in indices_to_remove:
            _log.info(
                "%s", f"Removing empty dataloader at index {idx} (dataset size: 0)"
            )
            del self.dataloaders[idx]
            del self.iterators[idx]

        # Recalculate priorities if needed
        if len(indices_to_remove) > 0 and self.priorities is not None:
            if len(self.dataloaders) > 0:
                self.priorities = np.zeros(len(self.dataloaders))
                for i in range(len(self.dataloaders)):
                    self.priorities[i] = self.dataloaders[i].dataset.__len__()
                self.priorities = self.priorities / self.priorities.sum()
            else:
                self.priorities = None

    def __len__(self):
        """Total number of samples over all member datasets."""
        length = 0
        for dataloader in self.dataloaders:
            length += len(dataloader.dataset)

        return length

    def enable_hints(self):
        """Enable hint outputs on all member datasets and reset the iterators."""
        for dataloader in self.dataloaders:
            if isinstance(dataloader, DataloaderMixer):
                dataloader.enable_hints()

            else:
                dataloader.dataset.enable_hints()

        self.reset()
        self.hints_enabled = True

    def disable_hints(self):
        """Disable hint outputs on all member datasets and reset the iterators."""
        for dataloader in self.dataloaders:
            if isinstance(dataloader, DataloaderMixer):
                dataloader.disable_hints()

            else:
                dataloader.dataset.disable_hints()

        self.reset()
        self.hints_enabled = False

    def enable_idx(self):
        """Make member datasets return sample indices and reset the iterators."""
        for dataloader in self.dataloaders:
            if isinstance(dataloader, DataloaderMixer):
                dataloader.enable_idx()

            else:
                dataloader.dataset.enable_idx()

        self.reset()
        self.idx_enabled = True

    def disable_idx(self):
        """Stop member datasets returning sample indices and reset the iterators."""
        for dataloader in self.dataloaders:
            if isinstance(dataloader, DataloaderMixer):
                dataloader.disable_idx()

            else:
                dataloader.dataset.disable_idx()

        self.reset()
        self.idx_enabled = False

    def enable_class_balancing(self):
        """
        Replace every plain member dataloader by a per-class concatenating mixer.

        For each class ``i`` in ``dataset.output_size`` a deep copy of the
        dataloader restricted to that class is created; the copies are
        wrapped in a nested ``DataloaderMixer`` with ``concatenate_batches``
        and ``steps_per_epoch = 200`` so each batch holds an equal share of
        every class. No-op if already enabled.
        """
        if not self.class_balancing_enabled:
            for idx, dataloader in enumerate(self.dataloaders):
                if isinstance(dataloader, DataloaderMixer):
                    dataloader.enable_class_balancing()

                else:
                    new_dataloaders = []

                    for i in range(dataloader.dataset.output_size):
                        # TODO this is a hacky way to get the class restriction
                        # if i not in [248, 269]:
                        #     continue
                        # print(f"using {i}")

                        dataloader_copy = copy.deepcopy(dataloader)
                        dataloader_copy.dataset.enable_class_restriction(i)
                        new_dataloaders.append(dataloader_copy)
                    new_config = copy.deepcopy(self.train_config)
                    new_config.steps_per_epoch = 200
                    new_config.concatenate_batches = True
                    self.dataloaders[idx] = DataloaderMixer(
                        new_config, new_dataloaders[0]
                    )
                    for i in range(1, len(new_dataloaders)):
                        self.dataloaders[idx].append(new_dataloaders[i])

            self.reset()
            self.class_balancing_enabled = True

    def disable_class_balancing(self):
        """
        Undo ``enable_class_balancing``.

        Keeps only the first plain member dataloader found (with its class
        restriction removed) and drops the priorities. No-op if not enabled.
        """
        if self.class_balancing_enabled:
            for idx, dataloader in enumerate(self.dataloaders):
                if not isinstance(dataloader, DataloaderMixer):
                    dataloader.dataset.disable_class_rectriction()
                    self.dataloaders = [dataloader]
                    self.priorities = None
                    break

                dataloader.disable_class_balancing()

            self.reset()
            self.class_balancing_enabled = False


class WeightedDataloaderList:
    """
    A list of ``DataLoader`` objects with associated sampling weights.

    Parameters
    ----------
    dataloaders : list of DataLoader
        Members; each is asserted to be a ``torch.utils.data.DataLoader``.
    weights : torch.Tensor, optional
        Initial weights. Defaults to uniform ``1 / len(dataloaders)``.
    """

    def __init__(self, dataloaders, weights=None):
        """Validate the members and set uniform weights if none are given."""
        for dataloader in dataloaders:
            assert isinstance(dataloader, torch.utils.data.DataLoader), (
                str(dataloader) + " is not dataloader!"
            )

        self.dataloaders = dataloaders
        if not weights is None:
            self.weights = weights

        else:
            self.weights = torch.ones([len(self.dataloaders)]) / len(self.dataloaders)

    def append(self, dataloader):
        """Add a dataloader, halving the existing weights and giving it weight 0.5."""
        assert isinstance(dataloader, torch.utils.data.DataLoader), (
            str(dataloader) + " is not dataloader!"
        )
        self.dataloaders.append(dataloader)
        self.weights *= 0.5
        self.weights = torch.cat([self.weights, torch.tensor([0.5])])


def resolve_num_workers(training_config=None):
    """Worker processes for the loaders ``get_dataloader`` builds.

    The default stays 0, i.e. the behaviour before 2026-09-25: images are
    decoded and resized on the training process between GPU steps, which the
    generator code measured at ~15 % GPU utilisation. Raise it per config
    (``training.num_workers``) or globally (``$PEAL_NUM_WORKERS``); the config
    wins when it is non-zero. It is not raised by default because a dataset that
    lazily attaches a CUDA model (``calculate_outlier_score``) cannot be forked
    into workers, and that has to be checked per dataset first.

    Parameters
    ----------
    training_config : TrainingConfig, optional
        Read for its ``num_workers`` field.

    Returns
    -------
    int
        Number of worker processes, ``0`` for in-process loading.
    """
    configured = getattr(training_config, "num_workers", None)
    if configured:
        return int(configured)
    env = os.environ.get("PEAL_NUM_WORKERS")
    return int(env) if env else 0


def get_dataloader(
    dataset,
    training_config=None,
    mode="train",
    task_config=None,
    batch_size=None,
    steps_per_epoch=None,
):
    """
    Wrap a PEAL dataset in a ``DataLoader`` (and a mixer for training).

    Parameters
    ----------
    dataset : PealDataset
        Dataset to load; ``dataset.task_config`` is set to ``task_config``
        and a ``task_config.class_restriction`` is enabled on it.
    training_config : TrainingConfig, optional
        Supplies ``<mode>_batch_size`` when ``batch_size`` is ``None`` and
        ``steps_per_epoch`` for the mixer. Required if ``batch_size`` is
        ``None``.
    mode : str, optional
        ``"train"`` (shuffled) or ``"val"``/``"test"`` (in order).
    task_config : TaskConfig, optional
        Task config attached to the dataset.
    batch_size : int, optional
        Explicit batch size overriding the training config.
    steps_per_epoch : int, optional
        If given (or set on ``training_config``) and ``mode == "train"``,
        the loader is wrapped in a ``DataloaderMixer``.

    Returns
    -------
    DataLoader or DataloaderMixer
        Single-process (``num_workers=0``) loader.
    """
    assert (
        not training_config is None or not batch_size is None
    ), "the batch size has to be given!"

    dataset.task_config = task_config
    if task_config is not None and task_config.class_restriction is not None:
        dataset.enable_class_restriction(task_config.class_restriction)

    num_workers = resolve_num_workers(training_config)
    loader_kwargs = dict(
        num_workers=num_workers,
        shuffle=bool(mode == "train"),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["pin_memory"] = torch.cuda.is_available()

    if batch_size is None:
        dataloader = DataLoader(
            dataset,
            batch_size=getattr(training_config, mode + "_batch_size"),
            **loader_kwargs,
        )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            **loader_kwargs,
        )

    if mode == "train" and (
        not steps_per_epoch is None
        or (not training_config is None and not training_config.steps_per_epoch is None)
    ):
        dataloader = DataloaderMixer(training_config, dataloader)

    return dataloader


def create_class_ordered_batch(dataset, config):
    """
    Build a small batch with two consecutive samples of every class.

    Parameters
    ----------
    dataset : PealDataset or DataLoader
        Source handed to ``DataStack``.
    config : object
        Config whose ``task.output_size`` (or ``data.output_size``) gives
        the number of classes.

    Returns
    -------
    tuple of torch.Tensor
        ``(X, y)`` with ``2 * output_size`` samples ordered by class.
    """
    if "output_size" in config.task.keys():
        output_size = config.task.output_size

    else:
        output_size = config.data.output_size

    datastack = DataStack(dataset, output_size)

    test_X = []
    test_y = []
    for i in range(output_size):
        test_X.append(datastack.pop(i)[0])
        test_X.append(datastack.pop(i)[0])
        test_y.append(i)
        test_y.append(i)

    test_X = torch.stack(test_X)
    test_y = torch.tensor(test_y)

    return test_X, test_y


def create_dataloaders_from_datasource(
    config,
    datasource=None,
    enable_hints=False,
    test_config=None,
):
    """
    Create the train/val/test dataloaders for a config from a datasource.

    Parameters
    ----------
    config : object
        Config with a ``data`` (or ``data_config``) section and optionally
        ``training``, ``task``, ``predictor`` (dict with ``training`` and
        ``task`` yaml paths, used when no ``training`` section exists) and
        ``transition_restrictions`` (its first entry becomes a class
        restriction on the train and val sets).
    datasource : str or tuple or list, optional
        A dataset root passed to ``get_datasets``, a tuple of two or three
        ``Dataset`` objects, or a tuple of two or three ``DataLoader``
        objects returned as-is. Defaults to ``data_config.dataset_path``.
    enable_hints : bool, optional
        Call ``enable_hints`` on the training dataset.
    test_config : object, optional
        Forwarded to ``get_datasets`` for a differently configured test set.

    Returns
    -------
    tuple
        ``(train_dataloader, val_dataloader, test_dataloader)``; a member is
        ``None`` when its dataset is empty. Training loaders are
        ``DataloaderMixer`` instances when ``steps_per_epoch`` is set.

    Notes
    -----
    As a side effect ``config.data`` is replaced by the training dataset's
    config when that dataset is not ``multiclass``. An unrecognized
    ``datasource`` prints a message and calls ``quit()``.
    """
    data_config = config.data if "data" in dir(config) else config.data_config
    if (isinstance(datasource, tuple) or isinstance(datasource, list)) and isinstance(
        datasource[0], DataLoader
    ):
        train_dataloader = datasource[0]

        val_dataloader = datasource[1]
        if len(datasource) == 2:
            test_dataloader = val_dataloader

        else:
            test_dataloader = datasource[2]

    else:
        if datasource is None:
            datasource = data_config.dataset_path
        if isinstance(datasource, str):
            dataset_train, dataset_val, dataset_test = get_datasets(
                config=data_config,
                base_dir=datasource,
                test_config=test_config,
            )
        elif isinstance(datasource[0], torch.utils.data.Dataset):
            if len(datasource) == 2:
                dataset_train, dataset_val = datasource
                dataset_test = dataset_val

            else:
                dataset_train, dataset_val, dataset_test = datasource

        else:
            _log.info("%s", "datasource is not a valid input!")
            quit()

        if enable_hints:
            dataset_train.enable_hints()
        # this is hacky needs to be done properly
        training_config = config.training if "training" in dir(config) else None
        task_config = config.task if "task" in dir(config) else None
        if "predictor" in dir(config) and training_config is None:
            training_config = load_yaml_config(
                config.predictor["training"], config_model=TrainingConfig
            )
            task_config = load_yaml_config(
                config.predictor["task"], config_model=TaskConfig
            )
            if config.transition_restrictions is not None:
                training_config.class_restriction = config.transition_restrictions[0]
        if len(dataset_train) > 0:
            if training_config is not None and isinstance(training_config, dict):
                batch_size = (
                    training_config["train_batch_size"]
                    if isinstance(training_config, dict)
                    else training_config.batch_size
                )
                step_per_epoch = training_config["steps_per_epoch"]
            else:
                batch_size = None
                step_per_epoch = None
            if "transition_restrictions" in dir(config):
                if config.transition_restrictions is not None:
                    _log.info(
                        "%s",
                        f"enabling class restriction training set{config.transition_restrictions[0]}",
                    )
                    dataset_train.enable_class_restriction(
                        config.transition_restrictions[0]
                    )
                    training_config.class_restriction = config.transition_restrictions[
                        0
                    ]

            train_dataloader = get_dataloader(
                dataset=dataset_train,
                training_config=training_config,
                mode="train",
                task_config=task_config,
                batch_size=batch_size,
                steps_per_epoch=step_per_epoch,
            )

        else:
            train_dataloader = None

        if len(dataset_val) > 0:
            if training_config is not None and isinstance(training_config, dict):
                batch_size = (
                    training_config["val_batch_size"]
                    if isinstance(training_config, dict)
                    else training_config.batch_size
                )
            else:
                batch_size = None
            # if training_config is None and config.transition_restrictions is not None:
            if "transition_restrictions" in dir(config):
                if config.transition_restrictions is not None:
                    _log.info(
                        "%s",
                        f"enabling class restriction validation set{config.transition_restrictions[0]}",
                    )
                    dataset_val.enable_class_restriction(
                        config.transition_restrictions[0]
                    )
                    training_config.class_restriction = config.transition_restrictions[
                        0
                    ]
                    _log.info("%s", "restriction enabled")
            step_per_epoch = None
            val_dataloader = get_dataloader(
                dataset=dataset_val,
                training_config=training_config,
                batch_size=batch_size,
                steps_per_epoch=step_per_epoch,
                mode="val",
                task_config=task_config,
            )
            # print(len(val_dataloader))
        else:
            val_dataloader = None

        if len(dataset_test) > 0:
            if training_config is not None and isinstance(training_config, dict):
                batch_size = (
                    training_config["test_batch_size"]
                    if isinstance(training_config, dict)
                    else training_config.batch_size
                )
            else:
                batch_size = None
            test_dataloader = get_dataloader(
                dataset=dataset_test,
                training_config=training_config,
                batch_size=batch_size,
                steps_per_epoch=step_per_epoch,
                mode="test",
                task_config=task_config,
            )

        else:
            test_dataloader = None

    # TODO this seems quite hacky and could cause problems when combining multiclass dataset with SegmentationMask teacher
    if (
        not train_dataloader is None
        and "config" in train_dataloader.dataset.__dict__.keys()
        and train_dataloader.dataset.config.output_type != "multiclass"
    ):
        # TODO sanity check or warning
        if "data" in dir(config):
            config.data = train_dataloader.dataset.config

    # TODO deal with other datasets
    return train_dataloader, val_dataloader, test_dataloader
