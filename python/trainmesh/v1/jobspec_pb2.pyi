from buf.validate import validate_pb2 as _validate_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobSpec(_message.Message):
    __slots__ = ("version", "cluster", "run", "execution", "training", "planner")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    TRAINING_FIELD_NUMBER: _ClassVar[int]
    PLANNER_FIELD_NUMBER: _ClassVar[int]
    version: int
    cluster: Cluster
    run: Run
    execution: Execution
    training: Training
    planner: _struct_pb2.Struct
    def __init__(self, version: _Optional[int] = ..., cluster: _Optional[_Union[Cluster, _Mapping]] = ..., run: _Optional[_Union[Run, _Mapping]] = ..., execution: _Optional[_Union[Execution, _Mapping]] = ..., training: _Optional[_Union[Training, _Mapping]] = ..., planner: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Cluster(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class Run(_message.Message):
    __slots__ = ("name", "seed", "resume", "results_path")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    RESUME_FIELD_NUMBER: _ClassVar[int]
    RESULTS_PATH_FIELD_NUMBER: _ClassVar[int]
    name: str
    seed: int
    resume: str
    results_path: str
    def __init__(self, name: _Optional[str] = ..., seed: _Optional[int] = ..., resume: _Optional[str] = ..., results_path: _Optional[str] = ...) -> None: ...

class Execution(_message.Message):
    __slots__ = ("strategy", "workers", "processes_per_worker", "cpu_threads_per_rank", "backend", "rendezvous")
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    PROCESSES_PER_WORKER_FIELD_NUMBER: _ClassVar[int]
    CPU_THREADS_PER_RANK_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    RENDEZVOUS_FIELD_NUMBER: _ClassVar[int]
    strategy: str
    workers: int
    processes_per_worker: int
    cpu_threads_per_rank: int
    backend: Backend
    rendezvous: Rendezvous
    def __init__(self, strategy: _Optional[str] = ..., workers: _Optional[int] = ..., processes_per_worker: _Optional[int] = ..., cpu_threads_per_rank: _Optional[int] = ..., backend: _Optional[_Union[Backend, _Mapping]] = ..., rendezvous: _Optional[_Union[Rendezvous, _Mapping]] = ...) -> None: ...

class Backend(_message.Message):
    __slots__ = ("device", "communication")
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    COMMUNICATION_FIELD_NUMBER: _ClassVar[int]
    device: str
    communication: str
    def __init__(self, device: _Optional[str] = ..., communication: _Optional[str] = ...) -> None: ...

class Rendezvous(_message.Message):
    __slots__ = ("backend", "endpoint")
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    backend: str
    endpoint: str
    def __init__(self, backend: _Optional[str] = ..., endpoint: _Optional[str] = ...) -> None: ...

class Training(_message.Message):
    __slots__ = ("model", "dataset", "optimizer", "batch", "epochs", "precision", "checkpoint", "validation")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_FIELD_NUMBER: _ClassVar[int]
    model: Model
    dataset: Dataset
    optimizer: Optimizer
    batch: Batch
    epochs: int
    precision: str
    checkpoint: Checkpoint
    validation: Validation
    def __init__(self, model: _Optional[_Union[Model, _Mapping]] = ..., dataset: _Optional[_Union[Dataset, _Mapping]] = ..., optimizer: _Optional[_Union[Optimizer, _Mapping]] = ..., batch: _Optional[_Union[Batch, _Mapping]] = ..., epochs: _Optional[int] = ..., precision: _Optional[str] = ..., checkpoint: _Optional[_Union[Checkpoint, _Mapping]] = ..., validation: _Optional[_Union[Validation, _Mapping]] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("type", "input_size", "hidden_size", "output_size")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_SIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
    type: str
    input_size: int
    hidden_size: int
    output_size: int
    def __init__(self, type: _Optional[str] = ..., input_size: _Optional[int] = ..., hidden_size: _Optional[int] = ..., output_size: _Optional[int] = ...) -> None: ...

class Dataset(_message.Message):
    __slots__ = ("type", "samples", "input_size", "output_size")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    INPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
    type: str
    samples: int
    input_size: int
    output_size: int
    def __init__(self, type: _Optional[str] = ..., samples: _Optional[int] = ..., input_size: _Optional[int] = ..., output_size: _Optional[int] = ...) -> None: ...

class Optimizer(_message.Message):
    __slots__ = ("name", "learning_rate")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    learning_rate: float
    def __init__(self, name: _Optional[str] = ..., learning_rate: _Optional[float] = ...) -> None: ...

class Batch(_message.Message):
    __slots__ = ("per_rank", "gradient_accumulation_steps")
    PER_RANK_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_ACCUMULATION_STEPS_FIELD_NUMBER: _ClassVar[int]
    per_rank: int
    gradient_accumulation_steps: int
    def __init__(self, per_rank: _Optional[int] = ..., gradient_accumulation_steps: _Optional[int] = ...) -> None: ...

class Checkpoint(_message.Message):
    __slots__ = ("enabled", "directory")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    directory: str
    def __init__(self, enabled: _Optional[bool] = ..., directory: _Optional[str] = ...) -> None: ...

class Validation(_message.Message):
    __slots__ = ("samples",)
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    samples: int
    def __init__(self, samples: _Optional[int] = ...) -> None: ...
