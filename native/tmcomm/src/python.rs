//! Python module `tmcomm` (built by maturin with the `python` feature).
//!
//! ```python
//! group = tmcomm.OuterGroup(island=0, peers=[(0, "10.0.0.1:29600", "node-a"), ...],
//!                           listen="0.0.0.0:29600", quorum=2, max_values=n,
//!                           codec="int8", tls=("cert.pem", "key.pem", "ca.pem"))
//! report = group.all_reduce_mean(round, flat_float32_numpy_array, sha256_digest)
//! ```
//! The array is replaced in place by the mean. The GIL is released while waiting on the network.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{Codec, Error, GroupConfig, OuterGroup, Peer, TlsFiles};

create_exception!(
    tmcomm,
    TmcommError,
    PyRuntimeError,
    "Inter-island communication failed."
);
create_exception!(
    tmcomm,
    QuorumLostError,
    TmcommError,
    "Fewer islands than the quorum remain."
);
create_exception!(
    tmcomm,
    DivergedError,
    TmcommError,
    "An island's model state differs."
);
create_exception!(
    tmcomm,
    ExcludedError,
    TmcommError,
    "This island was dropped by the group."
);

fn to_py(err: Error) -> PyErr {
    let message = err.to_string();
    match err {
        Error::QuorumLost { .. } => QuorumLostError::new_err(message),
        Error::Diverged { .. } => DivergedError::new_err(message),
        Error::Excluded { .. } => ExcludedError::new_err(message),
        Error::Config(_) => PyValueError::new_err(message),
        _ => TmcommError::new_err(message),
    }
}

#[pyclass(name = "OuterGroup", module = "tmcomm")]
struct PyOuterGroup {
    // Python classes must be Sync; the group's TLS streams are only Send.
    inner: Mutex<Option<OuterGroup>>,
}

impl PyOuterGroup {
    fn lock(&self) -> PyResult<MutexGuard<'_, Option<OuterGroup>>> {
        self.inner
            .lock()
            .map_err(|_| TmcommError::new_err("a previous call panicked"))
    }
}

#[pymethods]
impl PyOuterGroup {
    #[new]
    #[pyo3(signature = (
        island, peers, listen, quorum, max_values, codec = "int8", tls = None,
        round_timeout = 60.0, startup_timeout = 300.0, connect_timeout = 10.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        island: u32,
        peers: Vec<(u32, String, String)>,
        listen: String,
        quorum: usize,
        max_values: usize,
        codec: &str,
        tls: Option<(PathBuf, PathBuf, PathBuf)>,
        round_timeout: f64,
        startup_timeout: f64,
        connect_timeout: f64,
    ) -> PyResult<Self> {
        let config = GroupConfig {
            island,
            peers: peers
                .into_iter()
                .map(|(island, address, name)| Peer {
                    island,
                    address,
                    name,
                })
                .collect(),
            listen,
            tls: tls.map(|(cert, key, ca)| TlsFiles { cert, key, ca }),
            quorum,
            codec: Codec::parse(codec).map_err(to_py)?,
            max_values,
            round_timeout: seconds(round_timeout)?,
            startup_timeout: seconds(startup_timeout)?,
            connect_timeout: seconds(connect_timeout)?,
        };
        let group = py
            .allow_threads(|| OuterGroup::new(config))
            .map_err(to_py)?;
        Ok(Self {
            inner: Mutex::new(Some(group)),
        })
    }

    /// Replaces `values` (a writable, C-contiguous float32 buffer) with the group mean and
    /// returns the round report as a dict.
    fn all_reduce_mean<'py>(
        &self,
        py: Python<'py>,
        round: u64,
        values: PyBuffer<f32>,
        digest: &[u8],
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut guard = self.lock()?;
        let group = guard
            .as_mut()
            .ok_or_else(|| TmcommError::new_err("the group is closed"))?;
        if values.readonly() || !values.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "values must be a writable, C-contiguous float32 buffer",
            ));
        }
        let digest: [u8; 32] = digest
            .try_into()
            .map_err(|_| PyValueError::new_err("digest must be 32 bytes (sha256)"))?;
        let mut data = values.to_vec(py)?;
        let report = py
            .allow_threads(|| group.all_reduce_mean(round, &mut data, digest))
            .map_err(to_py)?;
        values.copy_from_slice(py, &data)?;

        let dict = PyDict::new(py);
        dict.set_item("round", report.round)?;
        dict.set_item("root", report.root)?;
        dict.set_item("members", report.members)?;
        dict.set_item("alive", report.alive)?;
        dict.set_item("bytes_sent", report.bytes_sent)?;
        dict.set_item("bytes_received", report.bytes_received)?;
        dict.set_item("seconds", report.seconds)?;
        Ok(dict)
    }

    #[getter]
    fn alive(&self) -> PyResult<Vec<u32>> {
        self.lock()?
            .as_ref()
            .map(OuterGroup::alive)
            .ok_or_else(|| TmcommError::new_err("the group is closed"))
    }

    /// Closes the listener and every connection.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(group) = self.lock()?.take() {
            py.allow_threads(|| drop(group));
        }
        Ok(())
    }
}

fn seconds(value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .map_err(|_| PyValueError::new_err(format!("{value} is not a valid number of seconds")))
}

#[pymodule]
fn tmcomm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyOuterGroup>()?;
    m.add("TmcommError", py.get_type::<TmcommError>())?;
    m.add("QuorumLostError", py.get_type::<QuorumLostError>())?;
    m.add("DivergedError", py.get_type::<DivergedError>())?;
    m.add("ExcludedError", py.get_type::<ExcludedError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
