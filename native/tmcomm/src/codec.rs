//! Lossy codecs for pseudo-gradients, plus error feedback so the loss does not accumulate.
//!
//! | codec | bytes/value | error |
//! |-------|-------------|-------|
//! | f32   | 4           | none |
//! | bf16  | 2           | relative 2^-9 (round to nearest even) |
//! | int8  | 1 (+4 per 1024-value chunk) | at most scale/2, scale = chunk absmax / 127 |

use crate::error::{Error, Result};

/// Values per int8 chunk; each chunk carries its own scale so one large value does not flatten
/// the rest of the vector.
pub const INT8_CHUNK: usize = 1024;
const INT8_MAX: f32 = 127.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    F32 = 0,
    Bf16 = 1,
    Int8 = 2,
}

impl Codec {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "" | "none" | "fp32" | "f32" => Ok(Codec::F32),
            "bf16" => Ok(Codec::Bf16),
            "int8" => Ok(Codec::Int8),
            other => Err(Error::Config(format!(
                "unknown codec {other:?} (use none, bf16 or int8)"
            ))),
        }
    }

    pub fn from_u8(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Codec::F32),
            1 => Ok(Codec::Bf16),
            2 => Ok(Codec::Int8),
            other => Err(Error::Protocol(format!("unknown codec code {other}"))),
        }
    }

    /// Scale count and data bytes a payload of `len` values must have.
    pub fn layout(self, len: usize) -> (usize, usize) {
        match self {
            Codec::F32 => (0, len * 4),
            Codec::Bf16 => (0, len * 2),
            Codec::Int8 => (len.div_ceil(INT8_CHUNK), len),
        }
    }
}

/// An encoded vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    pub codec: Codec,
    pub len: usize,
    pub scales: Vec<f32>,
    pub data: Vec<u8>,
}

impl Encoded {
    /// Bytes on the wire for scales and data (headers excluded).
    pub fn payload_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len()
    }
}

pub fn encode(codec: Codec, values: &[f32]) -> Result<Encoded> {
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(Error::Protocol(format!(
            "value {index} is not finite ({}); refusing to send it",
            values[index]
        )));
    }
    let (scales, data) = match codec {
        Codec::F32 => (
            Vec::new(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        Codec::Bf16 => (
            Vec::new(),
            values
                .iter()
                .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                .collect(),
        ),
        Codec::Int8 => encode_int8(values),
    };
    Ok(Encoded {
        codec,
        len: values.len(),
        scales,
        data,
    })
}

/// Overwrites `out` with the decoded values.
pub fn decode_into(encoded: &Encoded, out: &mut [f32]) -> Result<()> {
    out.fill(0.0);
    decode_add(encoded, out)
}

/// Adds the decoded values to `out`.
pub fn decode_add(encoded: &Encoded, out: &mut [f32]) -> Result<()> {
    check_shape(encoded, out.len())?;
    match encoded.codec {
        Codec::F32 => {
            for (slot, bytes) in out.iter_mut().zip(encoded.data.chunks_exact(4)) {
                *slot += f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        Codec::Bf16 => {
            for (slot, bytes) in out.iter_mut().zip(encoded.data.chunks_exact(2)) {
                *slot += bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
        }
        Codec::Int8 => {
            for ((chunk, data), scale) in out
                .chunks_mut(INT8_CHUNK)
                .zip(encoded.data.chunks(INT8_CHUNK))
                .zip(&encoded.scales)
            {
                for (slot, byte) in chunk.iter_mut().zip(data) {
                    *slot += f32::from(*byte as i8) * scale;
                }
            }
        }
    }
    Ok(())
}

pub fn check_shape(encoded: &Encoded, len: usize) -> Result<()> {
    let (scales, data) = encoded.codec.layout(encoded.len);
    if encoded.len != len || encoded.scales.len() != scales || encoded.data.len() != data {
        return Err(Error::Protocol(format!(
            "payload has {} values, {} scales, {} bytes; expected {len} values, {scales} scales, {data} bytes",
            encoded.len,
            encoded.scales.len(),
            encoded.data.len()
        )));
    }
    Ok(())
}

/// Error feedback: what compression dropped this round is added to the next round's input, so
/// the error stays bounded instead of accumulating over rounds.
#[derive(Debug, Default)]
pub struct ErrorFeedback {
    residual: Vec<f32>,
}

impl ErrorFeedback {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compress(&mut self, codec: Codec, values: &[f32]) -> Result<Encoded> {
        if codec == Codec::F32 {
            return encode(codec, values);
        }
        if self.residual.len() != values.len() {
            self.residual = vec![0.0; values.len()];
        }
        let corrected: Vec<f32> = values
            .iter()
            .zip(&self.residual)
            .map(|(v, r)| v + r)
            .collect();
        let encoded = encode(codec, &corrected)?;
        decode_into(&encoded, &mut self.residual)?;
        for (residual, wanted) in self.residual.iter_mut().zip(&corrected) {
            *residual = wanted - *residual;
        }
        Ok(encoded)
    }

    /// Largest absolute value still carried over (for tests and diagnostics).
    pub fn max_residual(&self) -> f32 {
        self.residual.iter().fold(0.0, |m, r| m.max(r.abs()))
    }
}

fn encode_int8(values: &[f32]) -> (Vec<f32>, Vec<u8>) {
    let mut scales = Vec::with_capacity(values.len().div_ceil(INT8_CHUNK));
    let mut data = Vec::with_capacity(values.len());
    for chunk in values.chunks(INT8_CHUNK) {
        let absmax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = absmax / INT8_MAX;
        scales.push(scale);
        for v in chunk {
            let q = if scale > 0.0 {
                (v / scale).round().clamp(-INT8_MAX, INT8_MAX) as i8
            } else {
                0
            };
            data.push(q as u8);
        }
    }
    (scales, data)
}

/// Round to nearest even on the upper 16 bits.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn bf16_to_f32(half: u16) -> f32 {
    f32::from_bits(u32::from(half) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32))
            .collect()
    }

    #[test]
    fn f32_round_trips_exactly() {
        let values = ramp(3000);
        let mut out = vec![0.0; values.len()];
        decode_into(&encode(Codec::F32, &values).unwrap(), &mut out).unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn bf16_error_is_within_half_an_ulp() {
        let values = ramp(3000);
        let encoded = encode(Codec::Bf16, &values).unwrap();
        assert_eq!(encoded.payload_bytes(), 6000);
        let mut out = vec![0.0; values.len()];
        decode_into(&encoded, &mut out).unwrap();
        for (got, want) in out.iter().zip(&values) {
            assert!(
                (got - want).abs() <= want.abs() * 2f32.powi(-8),
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn int8_error_is_within_half_a_step_per_chunk() {
        let values = ramp(3000);
        let encoded = encode(Codec::Int8, &values).unwrap();
        assert_eq!(encoded.scales.len(), 3);
        assert_eq!(encoded.payload_bytes(), 3000 + 3 * 4);
        let mut out = vec![0.0; values.len()];
        decode_into(&encoded, &mut out).unwrap();
        for (index, (got, want)) in out.iter().zip(&values).enumerate() {
            let scale = encoded.scales[index / INT8_CHUNK];
            assert!(
                (got - want).abs() <= scale * 0.5 + 1e-6,
                "{index}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn zero_chunk_encodes_to_zero() {
        let mut out = vec![1.0; 10];
        decode_into(&encode(Codec::Int8, &[0.0; 10]).unwrap(), &mut out).unwrap();
        assert_eq!(out, vec![0.0; 10]);
    }

    #[test]
    fn non_finite_values_are_refused() {
        assert!(matches!(
            encode(Codec::Int8, &[1.0, f32::NAN]),
            Err(Error::Protocol(_))
        ));
    }

    #[test]
    fn wrong_shape_is_refused() {
        let encoded = encode(Codec::Bf16, &[1.0, 2.0]).unwrap();
        let mut out = vec![0.0; 3];
        assert!(matches!(
            decode_into(&encoded, &mut out),
            Err(Error::Protocol(_))
        ));
    }

    #[test]
    fn error_feedback_makes_the_running_mean_exact() {
        // A value int8 cannot represent next to a large one in the same chunk: without error
        // feedback it rounds to 0 every round; with it, the decoded values average out to it.
        let mut values = vec![0.0f32; 16];
        values[0] = 100.0;
        values[1] = 0.3; // scale = 100/127 ≈ 0.787, so 0.3 rounds to 0
        let mut feedback = ErrorFeedback::new();
        let rounds = 1000;
        let mut sum = vec![0.0f64; values.len()];
        for _ in 0..rounds {
            let encoded = feedback.compress(Codec::Int8, &values).unwrap();
            let mut out = vec![0.0; values.len()];
            decode_into(&encoded, &mut out).unwrap();
            for (s, o) in sum.iter_mut().zip(&out) {
                *s += f64::from(*o);
            }
        }
        let mean = sum[1] / f64::from(rounds);
        assert!((mean - 0.3).abs() < 1e-3, "mean of decoded 0.3 = {mean}");
        assert!(feedback.max_residual() <= 100.0 / 127.0);
    }

    #[test]
    fn codec_names_parse() {
        assert_eq!(Codec::parse("int8").unwrap(), Codec::Int8);
        assert_eq!(Codec::parse("none").unwrap(), Codec::F32);
        assert!(Codec::parse("fp8").is_err());
    }
}
