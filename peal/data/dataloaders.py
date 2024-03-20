import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from peal.data.dataset_factory import get_datasets


class DataStack:
    """
    This class is used to create a stack of data for each class.
    """

    def __init__(self, datasource, num_classes, transform=None):
        """
        This function is used to initialize the DataStack class.

        Args:
            datasource (_type_): _description_
            num_classes (_type_): _description_
        """
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
        This function is used to fill the stack with data.
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
                    or self.dataset.idx_enabled
                ):
                    y_index = y[0]

                else:
                    y_index = y

                self.data[int(y_index)].append([X, y])
                self.current_idx = (self.current_idx + 1) % self.dataset.__len__()

            else:
                X, y = next(iter(self.datasource))
                if (
                    hasattr(self.dataset, "hints_enabled")
                    and self.dataset.hints_enabled
                    or self.dataset.idx_enabled
                ):
                    for i in range(X.shape[0]):
                        y_out = tuple([y_elem[i] for y_elem in y])
                        self.data[int(y[0][i])].append([X[i], y_out])

                else:
                    for i in range(X.shape[0]):
                        self.data[int(y[i])].append([X[i], int(y[i])])

        if not self.transform is None:
            self.dataset.transform = data_transform

    def pop(self, class_idx):
        """
        This function is used to pop a sample from the stack.

        Args:
            class_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        sample = self.data[class_idx].pop(0)
        self.fill_stack()
        return sample

    def reset(self):
        """
        This function is used to reset the stack.
        """
        self.data = []
        for idx in range(self.num_classes):
            self.data.append([])

        # TODO if no mixer is used here this will not work
        if isinstance(self.datasource, DataloaderMixer):
            self.datasource.reset()

        self.fill_stack()


class DataIterator:
    """
    This class is used to iterate over the data in a dataset.
    """

    def __init__(self, dataloader):
        """
        _summary_

        Args:
            dataloader (_type_): _description_
        """
        self.dataloader = dataloader
        # member variable to keep track of current index
        self._index = 0

    def __next__(self):
        """
        _summary_

        Raises:
            StopIteration: _description_

        Returns:
            _type_: _description_
        """
        if self._index < self.dataloader.train_config.steps_per_epoch:
            self._index += 1
            return self.dataloader.sample()

        # End of Iteration
        raise StopIteration


class DataloaderMixer(DataLoader):
    """
    _summary_

    Args:
        DataLoader (_type_): _description_
    """

    def __init__(self, train_config, initial_dataloader, return_src=False):
        """
        _summary_

        Args:
            train_config (_type_): _description_
            initial_dataloader (_type_): _description_
        """
        self.train_config = train_config
        self.dataloaders = [initial_dataloader]
        self.batch_size = initial_dataloader.batch_size
        self.priorities = None
        self.dataset = initial_dataloader.dataset  # TODO kind of hacky
        self.iterators = [iter(self.dataloaders[0])]
        self.return_src = return_src

    def append(self, dataloader, priority=1, mixing_ratio=None):
        """
        _summary_

        Args:
            dataloader (_type_): _description_
            priority (int, optional): _description_. Defaults to 1.
        """
        self.dataloaders.append(dataloader)
        self.iterators.append(iter(self.dataloaders[-1]))
        if mixing_ratio is None:
            self.priorities = np.zeros(len(self.dataloaders))
            for i in range(len(self.dataloaders)):
                self.priorities[i] = self.dataloaders[i].dataset.__len__()

            self.priorities[-1] *= priority
            self.priorities = self.priorities / self.priorities.sum()

        else:
            self.priorities = np.array([1 - mixing_ratio, mixing_ratio])

    def __iter__(self):
        return DataIterator(self)

    def sample(self):
        if not self.priorities is None:
            idx = int(np.random.multinomial(1, self.priorities).argmax())

        else:
            idx = 0

        item = next(self.iterators[idx], "STOP")
        if isinstance(item, str) and item == "STOP":
            self.iterators[idx] = iter(self.dataloaders[idx])
            item = next(self.iterators[idx])

        if self.return_src:
            item = (item, idx)

        return item

    def reset(self):
        for i in range(len(self.dataloaders)):
            self.dataloaders[i] = DataLoader(
                self.dataloaders[i].dataset, batch_size=self.dataloaders[i].batch_size
            )
            self.iterators[i] = iter(self.dataloaders[i])

    def __len__(self):
        length = 0
        for dataloader in self.dataloaders:
            length += len(dataloader.dataset)

        return length


def get_dataloader(
    dataset,
    training_config=None,
    mode="train",
    task_config=None,
    batch_size=None,
    steps_per_epoch=None,
):
    assert (
        not training_config is None or not batch_size is None
    ), "the batch size has to be given!"
    dataset.task_config = task_config
    if batch_size is None:
        dataloader = DataLoader(
            dataset,
            batch_size=getattr(training_config, mode + "_batch_size"),
            num_workers=8,
            shuffle=True,
        )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
        )

    if mode == "train" and (
        not steps_per_epoch is None
        or (not training_config is None and not training_config.steps_per_epoch is None)
    ):
        dataloader = DataloaderMixer(training_config, dataloader)

    return dataloader


def create_class_ordered_batch(dataset, config):
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
    This function creates the dataloaders from a given datasource.
    """
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
            datasource = config.data.dataset_path

        if isinstance(datasource, str):
            dataset_train, dataset_val, dataset_test = get_datasets(
                config=config.data,
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
            print("datasource is not a valid input!")
            quit()

        """
        if hasattr(config, "architecture") and isinstance(
            config.architecture, VAEConfig
        ):
            dataset_train = VAEDatasetWrapper(dataset_train)
            dataset_val = VAEDatasetWrapper(dataset_val)
            dataset_test = VAEDatasetWrapper(dataset_test)

        # TODO reintegrate normalizing flows
        if "n_bits" in config.architecture.keys():
            dataset_train = GlowDatasetWrapper(
                dataset_train, config.architecture.n_bits
            )
            dataset_val = GlowDatasetWrapper(
                dataset_val, config.architecture.n_bits
            )
            dataset_test = GlowDatasetWrapper(
                dataset_test, config.architecture.n_bits
            )
        
        # TODO reintegrate hints
        """
        if enable_hints:
            dataset_train.enable_hints()

        if len(dataset_train) > 0:
            train_dataloader = get_dataloader(
                dataset=dataset_train,
                training_config=config.training,
                mode="train",
                task_config=config.task,
            )

        else:
            train_dataloader = None

        if len(dataset_val) > 0:
            val_dataloader = get_dataloader(
                dataset=dataset_val,
                training_config=config.training,
                mode="val",
                task_config=config.task,
            )

        else:
            val_dataloader = None

        if len(dataset_test) > 0:
            test_dataloader = get_dataloader(
                dataset=dataset_test,
                training_config=config.training,
                mode="test",
                task_config=config.task,
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
        config.data = train_dataloader.dataset.config

    # TODO deal with other datasets
    """for dataloader in [train_dataloader, val_dataloader, test_dataloader]:
        if not isinstance(dataloader.dataset, PealDataset):
            dataloader.dataset = wrap_dataset(dataloader.dataset, config.data)"""

    return train_dataloader, val_dataloader, test_dataloader
