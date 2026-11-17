//! tmcomm: communication between TrainMesh islands.
//!
//! Inside an island, ranks synchronise every step with PyTorch collectives over a fast link.
//! Between islands (slow, possibly WAN links) DiLoCo synchronises rarely: every H inner steps
//! the island leaders average their pseudo-gradients with [`group::OuterGroup`]. This crate makes
//! that exchange small (bf16/int8 with error feedback, [`codec`]), authenticated (mTLS with the
//! cluster CA, [`tls`]) and tolerant of island loss (quorum and root failover).

pub mod codec;
pub mod error;
pub mod group;
pub mod tls;
pub mod wire;

#[cfg(feature = "python")]
mod python;

pub use codec::{Codec, Encoded, ErrorFeedback};
pub use error::{Error, Result};
pub use group::{GroupConfig, OuterGroup, Peer, RoundReport};
pub use tls::TlsFiles;
