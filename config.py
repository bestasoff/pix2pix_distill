import dataclasses

@dataclasses.dataclass
class DataConfig:
    dataroot: str = 'cityscapes_dataset'
    direction: str = 'BtoA'

@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 8
    lr: float = 0.0002
    epoch: int = 200
    reconstruction_lambda: float = 100.0
    distillation_lambda: float = 1.0
    generator_lambda: float = 1.0