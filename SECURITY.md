# Security policy

TrainMesh currently exposes no network listener and does not support remote worker enrollment or remote process launch. Do not expose any future coordinator or agent service until authenticated TLS transport, token expiry/revocation, safe process constraints, and audit logging are implemented.

Checkpoint files use Python serialization and should only be resumed from trusted local sources. A SHA-256 value in adjacent metadata detects accidental corruption; it does not authenticate the checkpoint against a malicious replacement.

Report security issues privately to the repository maintainers before public disclosure.
