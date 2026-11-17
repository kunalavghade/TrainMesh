//! Length-checked binary frames between island leaders (little endian).
//!
//! ```text
//! frame   = magic "TMC1" | kind u8 | body
//! Hello        = island u32
//! Contribution = round u64 | island u32 | digest [32] | payload
//! Result       = round u64 | n u32 | members u32*n | m u32 | alive u32*m | payload
//! Abort        = round u64 | len u32 | reason utf8
//! payload = codec u8 | len u64 | scales u32 | data u64 | scales f32* | data bytes
//! ```
//!
//! Every count is checked against `max_values` (the model size) before anything is allocated,
//! so a peer cannot make the reader allocate arbitrary memory.

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::codec::{check_shape, Codec, Encoded};
use crate::error::{Error, Result};

pub const MAGIC: [u8; 4] = *b"TMC1";
pub const DIGEST_LEN: usize = 32;
const MAX_ISLANDS: u32 = 4096;
const MAX_REASON: u32 = 4096;

const KIND_HELLO: u8 = 1;
const KIND_CONTRIBUTION: u8 = 2;
const KIND_RESULT: u8 = 3;
const KIND_ABORT: u8 = 4;

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    Hello {
        island: u32,
    },
    Contribution {
        round: u64,
        island: u32,
        digest: [u8; DIGEST_LEN],
        payload: Encoded,
    },
    Result {
        round: u64,
        members: Vec<u32>,
        alive: Vec<u32>,
        payload: Encoded,
    },
    Abort {
        round: u64,
        reason: String,
    },
}

/// Writes one frame and returns its size in bytes.
pub async fn write_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    message: &Message,
) -> Result<u64> {
    let mut head = Vec::with_capacity(128);
    head.extend_from_slice(&MAGIC);
    let payload = match message {
        Message::Hello { island } => {
            head.push(KIND_HELLO);
            head.extend_from_slice(&island.to_le_bytes());
            None
        }
        Message::Contribution {
            round,
            island,
            digest,
            payload,
        } => {
            head.push(KIND_CONTRIBUTION);
            head.extend_from_slice(&round.to_le_bytes());
            head.extend_from_slice(&island.to_le_bytes());
            head.extend_from_slice(digest);
            Some(payload)
        }
        Message::Result {
            round,
            members,
            alive,
            payload,
        } => {
            head.push(KIND_RESULT);
            head.extend_from_slice(&round.to_le_bytes());
            for list in [members, alive] {
                head.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for island in list {
                    head.extend_from_slice(&island.to_le_bytes());
                }
            }
            Some(payload)
        }
        Message::Abort { round, reason } => {
            head.push(KIND_ABORT);
            head.extend_from_slice(&round.to_le_bytes());
            let bytes = reason.as_bytes();
            let len = bytes.len().min(MAX_REASON as usize);
            head.extend_from_slice(&(len as u32).to_le_bytes());
            head.extend_from_slice(&bytes[..len]);
            None
        }
    };
    let mut size = head.len() as u64;
    if let Some(payload) = payload {
        head.push(payload.codec as u8);
        head.extend_from_slice(&(payload.len as u64).to_le_bytes());
        head.extend_from_slice(&(payload.scales.len() as u32).to_le_bytes());
        head.extend_from_slice(&(payload.data.len() as u64).to_le_bytes());
        for scale in &payload.scales {
            head.extend_from_slice(&scale.to_le_bytes());
        }
        size = (head.len() + payload.data.len()) as u64;
        writer.write_all(&head).await?;
        writer.write_all(&payload.data).await?;
    } else {
        writer.write_all(&head).await?;
    }
    writer.flush().await?;
    Ok(size)
}

/// Reads one frame; payloads longer than `max_values` are refused before allocation.
pub async fn read_message<R: AsyncRead + Unpin>(
    reader: &mut R,
    max_values: usize,
) -> Result<(Message, u64)> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).await?;
    if magic != MAGIC {
        return Err(Error::Protocol(format!("bad magic {magic:?}")));
    }
    let kind = reader.read_u8().await?;
    let mut size = 5u64;
    let message = match kind {
        KIND_HELLO => {
            size += 4;
            Message::Hello {
                island: reader.read_u32_le().await?,
            }
        }
        KIND_CONTRIBUTION => {
            let round = reader.read_u64_le().await?;
            let island = reader.read_u32_le().await?;
            let mut digest = [0u8; DIGEST_LEN];
            reader.read_exact(&mut digest).await?;
            let (payload, bytes) = read_payload(reader, max_values).await?;
            size += 12 + DIGEST_LEN as u64 + bytes;
            Message::Contribution {
                round,
                island,
                digest,
                payload,
            }
        }
        KIND_RESULT => {
            let round = reader.read_u64_le().await?;
            let members = read_islands(reader).await?;
            let alive = read_islands(reader).await?;
            let (payload, bytes) = read_payload(reader, max_values).await?;
            size += 16 + 4 * (members.len() + alive.len()) as u64 + bytes;
            Message::Result {
                round,
                members,
                alive,
                payload,
            }
        }
        KIND_ABORT => {
            let round = reader.read_u64_le().await?;
            let len = reader.read_u32_le().await?;
            if len > MAX_REASON {
                return Err(Error::Protocol(format!("abort reason of {len} bytes")));
            }
            let mut bytes = vec![0u8; len as usize];
            reader.read_exact(&mut bytes).await?;
            size += 12 + u64::from(len);
            Message::Abort {
                round,
                reason: String::from_utf8_lossy(&bytes).into_owned(),
            }
        }
        other => return Err(Error::Protocol(format!("unknown frame kind {other}"))),
    };
    Ok((message, size))
}

async fn read_islands<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Vec<u32>> {
    let count = reader.read_u32_le().await?;
    if count > MAX_ISLANDS {
        return Err(Error::Protocol(format!("{count} islands in one frame")));
    }
    let mut islands = Vec::with_capacity(count as usize);
    for _ in 0..count {
        islands.push(reader.read_u32_le().await?);
    }
    Ok(islands)
}

async fn read_payload<R: AsyncRead + Unpin>(
    reader: &mut R,
    max_values: usize,
) -> Result<(Encoded, u64)> {
    let codec = Codec::from_u8(reader.read_u8().await?)?;
    let len = reader.read_u64_le().await?;
    let scale_count = reader.read_u32_le().await?;
    let data_len = reader.read_u64_le().await?;
    if len > max_values as u64 {
        return Err(Error::Protocol(format!(
            "payload of {len} values exceeds the model size {max_values}"
        )));
    }
    let len = len as usize;
    let (want_scales, want_data) = codec.layout(len);
    if scale_count as usize != want_scales || data_len as usize != want_data {
        return Err(Error::Protocol(format!(
            "{codec:?} payload of {len} values must have {want_scales} scales and {want_data} bytes"
        )));
    }
    let mut scales = Vec::with_capacity(want_scales);
    for _ in 0..want_scales {
        scales.push(f32::from_le_bytes(
            reader.read_u32_le().await?.to_le_bytes(),
        ));
    }
    let mut data = vec![0u8; want_data];
    reader.read_exact(&mut data).await?;
    let payload = Encoded {
        codec,
        len,
        scales,
        data,
    };
    check_shape(&payload, len)?;
    let bytes = 21 + 4 * want_scales as u64 + want_data as u64;
    Ok((payload, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode;

    async fn round_trip(message: Message, max_values: usize) -> Result<Message> {
        let mut buffer = Vec::new();
        let written = write_message(&mut buffer, &message).await?;
        assert_eq!(written, buffer.len() as u64);
        let (decoded, read) = read_message(&mut buffer.as_slice(), max_values).await?;
        assert_eq!(read, written);
        Ok(decoded)
    }

    #[tokio::test]
    async fn every_message_round_trips() {
        let payload = encode(Codec::Int8, &[1.0, -2.0, 3.5]).unwrap();
        for message in [
            Message::Hello { island: 7 },
            Message::Contribution {
                round: 3,
                island: 2,
                digest: [9; DIGEST_LEN],
                payload: payload.clone(),
            },
            Message::Result {
                round: 3,
                members: vec![0, 2],
                alive: vec![0, 1, 2],
                payload,
            },
            Message::Abort {
                round: 4,
                reason: "diverged".into(),
            },
        ] {
            assert_eq!(round_trip(message.clone(), 3).await.unwrap(), message);
        }
    }

    #[tokio::test]
    async fn oversized_payload_is_refused() {
        let message = Message::Contribution {
            round: 0,
            island: 0,
            digest: [0; DIGEST_LEN],
            payload: encode(Codec::F32, &[0.0; 10]).unwrap(),
        };
        assert!(matches!(
            round_trip(message, 9).await,
            Err(Error::Protocol(_))
        ));
    }

    #[tokio::test]
    async fn bad_magic_is_refused() {
        let bytes = b"XXXX\x01\x00\x00\x00\x00".to_vec();
        assert!(matches!(
            read_message(&mut bytes.as_slice(), 1).await,
            Err(Error::Protocol(_))
        ));
    }

    #[tokio::test]
    async fn inconsistent_lengths_are_refused() {
        let mut buffer = Vec::new();
        write_message(
            &mut buffer,
            &Message::Contribution {
                round: 0,
                island: 0,
                digest: [0; DIGEST_LEN],
                payload: encode(Codec::Bf16, &[1.0, 2.0]).unwrap(),
            },
        )
        .await
        .unwrap();
        // Corrupt the data length field (last 8 bytes before the 4 data bytes).
        let at = buffer.len() - 4 - 8;
        buffer[at] = 99;
        assert!(matches!(
            read_message(&mut buffer.as_slice(), 2).await,
            Err(Error::Protocol(_))
        ));
    }
}
