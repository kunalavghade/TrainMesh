import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Capabilities(_message.Message):
    __slots__ = ("hostname", "os", "arch", "logical_cpus", "memory_bytes", "runtime_version", "torch_version", "device", "devices")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    ARCH_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_CPUS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    os: str
    arch: str
    logical_cpus: int
    memory_bytes: int
    runtime_version: str
    torch_version: str
    device: str
    devices: _containers.RepeatedCompositeFieldContainer[DeviceProfile]
    def __init__(self, hostname: _Optional[str] = ..., os: _Optional[str] = ..., arch: _Optional[str] = ..., logical_cpus: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., runtime_version: _Optional[str] = ..., torch_version: _Optional[str] = ..., device: _Optional[str] = ..., devices: _Optional[_Iterable[_Union[DeviceProfile, _Mapping]]] = ...) -> None: ...

class DeviceProfile(_message.Message):
    __slots__ = ("kind", "samples_per_second")
    KIND_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    kind: str
    samples_per_second: float
    def __init__(self, kind: _Optional[str] = ..., samples_per_second: _Optional[float] = ...) -> None: ...

class JoinRequest(_message.Message):
    __slots__ = ("token", "csr_der")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    CSR_DER_FIELD_NUMBER: _ClassVar[int]
    token: str
    csr_der: bytes
    def __init__(self, token: _Optional[str] = ..., csr_der: _Optional[bytes] = ...) -> None: ...

class JoinResponse(_message.Message):
    __slots__ = ("node_id", "certificate_der", "ca_certificate_der", "heartbeat_interval_ms")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CERTIFICATE_DER_FIELD_NUMBER: _ClassVar[int]
    CA_CERTIFICATE_DER_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    certificate_der: bytes
    ca_certificate_der: bytes
    heartbeat_interval_ms: int
    def __init__(self, node_id: _Optional[str] = ..., certificate_der: _Optional[bytes] = ..., ca_certificate_der: _Optional[bytes] = ..., heartbeat_interval_ms: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("capabilities", "address", "interface")
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    INTERFACE_FIELD_NUMBER: _ClassVar[int]
    capabilities: Capabilities
    address: str
    interface: str
    def __init__(self, capabilities: _Optional[_Union[Capabilities, _Mapping]] = ..., address: _Optional[str] = ..., interface: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("state", "directives", "heartbeat_interval_ms")
    STATE_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVES_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    state: str
    directives: _containers.RepeatedCompositeFieldContainer[Directive]
    heartbeat_interval_ms: int
    def __init__(self, state: _Optional[str] = ..., directives: _Optional[_Iterable[_Union[Directive, _Mapping]]] = ..., heartbeat_interval_ms: _Optional[int] = ...) -> None: ...

class Directive(_message.Message):
    __slots__ = ("start", "cancel")
    START_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    start: StartAssignment
    cancel: CancelAssignment
    def __init__(self, start: _Optional[_Union[StartAssignment, _Mapping]] = ..., cancel: _Optional[_Union[CancelAssignment, _Mapping]] = ...) -> None: ...

class StartAssignment(_message.Message):
    __slots__ = ("job_id", "node_rank", "machines", "ranks_per_machine", "master_addr", "master_port", "config_yaml")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_RANK_FIELD_NUMBER: _ClassVar[int]
    MACHINES_FIELD_NUMBER: _ClassVar[int]
    RANKS_PER_MACHINE_FIELD_NUMBER: _ClassVar[int]
    MASTER_ADDR_FIELD_NUMBER: _ClassVar[int]
    MASTER_PORT_FIELD_NUMBER: _ClassVar[int]
    CONFIG_YAML_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    node_rank: int
    machines: int
    ranks_per_machine: int
    master_addr: str
    master_port: int
    config_yaml: bytes
    def __init__(self, job_id: _Optional[str] = ..., node_rank: _Optional[int] = ..., machines: _Optional[int] = ..., ranks_per_machine: _Optional[int] = ..., master_addr: _Optional[str] = ..., master_port: _Optional[int] = ..., config_yaml: _Optional[bytes] = ...) -> None: ...

class CancelAssignment(_message.Message):
    __slots__ = ("job_id", "reason")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    reason: str
    def __init__(self, job_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class ReportAssignmentRequest(_message.Message):
    __slots__ = ("job_id", "state", "message", "result_json")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: str
    message: str
    result_json: bytes
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[str] = ..., message: _Optional[str] = ..., result_json: _Optional[bytes] = ...) -> None: ...

class ReportAssignmentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CreateTokenRequest(_message.Message):
    __slots__ = ("ttl_seconds", "uses")
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    USES_FIELD_NUMBER: _ClassVar[int]
    ttl_seconds: int
    uses: int
    def __init__(self, ttl_seconds: _Optional[int] = ..., uses: _Optional[int] = ...) -> None: ...

class CreateTokenResponse(_message.Message):
    __slots__ = ("join_string", "expires_at")
    JOIN_STRING_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    join_string: str
    expires_at: _timestamp_pb2.Timestamp
    def __init__(self, join_string: _Optional[str] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("id", "state", "address", "interface", "capabilities", "joined_at", "last_heartbeat", "current_job_id", "island")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    INTERFACE_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    JOINED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    CURRENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ISLAND_FIELD_NUMBER: _ClassVar[int]
    id: str
    state: str
    address: str
    interface: str
    capabilities: Capabilities
    joined_at: _timestamp_pb2.Timestamp
    last_heartbeat: _timestamp_pb2.Timestamp
    current_job_id: str
    island: str
    def __init__(self, id: _Optional[str] = ..., state: _Optional[str] = ..., address: _Optional[str] = ..., interface: _Optional[str] = ..., capabilities: _Optional[_Union[Capabilities, _Mapping]] = ..., joined_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., last_heartbeat: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., current_job_id: _Optional[str] = ..., island: _Optional[str] = ...) -> None: ...

class ListNodesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListNodesResponse(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class SubmitJobRequest(_message.Message):
    __slots__ = ("config_yaml",)
    CONFIG_YAML_FIELD_NUMBER: _ClassVar[int]
    config_yaml: bytes
    def __init__(self, config_yaml: _Optional[bytes] = ...) -> None: ...

class SubmitJobResponse(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class Assignment(_message.Message):
    __slots__ = ("node_id", "node_rank", "state", "message")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_RANK_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    node_rank: int
    state: str
    message: str
    def __init__(self, node_id: _Optional[str] = ..., node_rank: _Optional[int] = ..., state: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("id", "state", "machines", "ranks_per_machine", "master_addr", "master_port", "assignments", "result_json", "error", "created_at", "finished_at", "rank_batch_sizes", "communication")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MACHINES_FIELD_NUMBER: _ClassVar[int]
    RANKS_PER_MACHINE_FIELD_NUMBER: _ClassVar[int]
    MASTER_ADDR_FIELD_NUMBER: _ClassVar[int]
    MASTER_PORT_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    RANK_BATCH_SIZES_FIELD_NUMBER: _ClassVar[int]
    COMMUNICATION_FIELD_NUMBER: _ClassVar[int]
    id: str
    state: str
    machines: int
    ranks_per_machine: int
    master_addr: str
    master_port: int
    assignments: _containers.RepeatedCompositeFieldContainer[Assignment]
    result_json: bytes
    error: str
    created_at: _timestamp_pb2.Timestamp
    finished_at: _timestamp_pb2.Timestamp
    rank_batch_sizes: _containers.RepeatedScalarFieldContainer[int]
    communication: str
    def __init__(self, id: _Optional[str] = ..., state: _Optional[str] = ..., machines: _Optional[int] = ..., ranks_per_machine: _Optional[int] = ..., master_addr: _Optional[str] = ..., master_port: _Optional[int] = ..., assignments: _Optional[_Iterable[_Union[Assignment, _Mapping]]] = ..., result_json: _Optional[bytes] = ..., error: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., rank_batch_sizes: _Optional[_Iterable[int]] = ..., communication: _Optional[str] = ...) -> None: ...

class GetJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class GetJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class CancelJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class CancelJobResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
