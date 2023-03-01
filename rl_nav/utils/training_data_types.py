import abc


class TrainingDataType(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        pass


class GeneratedTrainingData(TrainingDataType):
    def __init__(self, num_steps: int):
        self._num_steps = num_steps
        print('YAY')

    def __next__(self):
        pass


class FromFileTrainingData(TrainingDataType):
    def __init__(self, file_path: str):
        self._file_path = file_path

    def __next__(self):
        pass # THIS WILL NEED TO CHANGE
