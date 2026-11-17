//! Typed errors. Callers branch on the variant: a lost quorum or a divergence ends the job,
//! while a failed peer is handled inside the group by electing a new root.

/// Everything that can go wrong in tmcomm.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("tls: {0}")]
    Tls(String),
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("config: {0}")]
    Config(String),
    #[error("round {round}: {alive} island(s) reachable, quorum is {quorum}")]
    QuorumLost {
        round: u64,
        alive: usize,
        quorum: usize,
    },
    #[error("round {round}: island {island} holds a different model state (digest mismatch)")]
    Diverged { round: u64, island: u32 },
    #[error("round {round}: this island was dropped by the group ({reason})")]
    Excluded { round: u64, reason: String },
    #[error("aborted by the root in round {round}: {reason}")]
    Aborted { round: u64, reason: String },
}

impl From<rustls::Error> for Error {
    fn from(err: rustls::Error) -> Self {
        Error::Tls(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
