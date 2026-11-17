//! Mutual TLS 1.3 between island leaders with the cluster CA that `meshd` runs.
//!
//! Node certificates carry the node ID as a DNS name and both server and client auth, so a
//! leader can dial another leader by node ID (server name check) and the accepting leader can
//! check that the dialer's certificate names the island it claims to be.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use tokio_rustls::{TlsAcceptor, TlsConnector};

use crate::error::{Error, Result};

/// PEM files written by `tm-agent` from its node identity.
#[derive(Clone, Debug)]
pub struct TlsFiles {
    pub cert: PathBuf,
    pub key: PathBuf,
    pub ca: PathBuf,
}

/// Ready-to-use TLS endpoints for one node.
#[derive(Clone)]
pub struct Endpoints {
    pub acceptor: TlsAcceptor,
    pub connector: TlsConnector,
}

impl Endpoints {
    pub fn load(files: &TlsFiles) -> Result<Self> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut roots = RootCertStore::empty();
        for cert in read_certs(&files.ca)? {
            roots.add(cert)?;
        }
        let roots = Arc::new(roots);
        let chain = read_certs(&files.cert)?;
        let key = PrivateKeyDer::from_pem_file(&files.key)
            .map_err(|e| Error::Tls(format!("reading key {}: {e}", files.key.display())))?;

        let verifier = WebPkiClientVerifier::builder_with_provider(roots.clone(), provider.clone())
            .build()
            .map_err(|e| Error::Tls(format!("client verifier: {e}")))?;
        let server = ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_client_cert_verifier(verifier)
            .with_single_cert(chain.clone(), key.clone_key())?;
        let client = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_root_certificates(roots)
            .with_client_auth_cert(chain, key)?;
        Ok(Self {
            acceptor: TlsAcceptor::from(Arc::new(server)),
            connector: TlsConnector::from(Arc::new(client)),
        })
    }
}

/// The server name a leader is dialled by: its node ID.
pub fn server_name(name: &str) -> Result<ServerName<'static>> {
    ServerName::try_from(name.to_owned())
        .map_err(|e| Error::Config(format!("peer name {name:?} is not a valid DNS name: {e}")))
}

/// Checks that a verified client certificate was issued to `name`.
pub fn check_peer_name(cert: &CertificateDer<'_>, name: &str) -> Result<()> {
    let end_entity = webpki::EndEntityCert::try_from(cert)
        .map_err(|e| Error::Tls(format!("parsing peer certificate: {e}")))?;
    end_entity
        .verify_is_valid_for_subject_name(&server_name(name)?)
        .map_err(|e| Error::Tls(format!("peer certificate is not for {name}: {e}")))
}

fn read_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let certs = CertificateDer::pem_file_iter(path)
        .map_err(|e| Error::Tls(format!("reading {}: {e}", path.display())))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::Tls(format!("parsing {}: {e}", path.display())))?;
    if certs.is_empty() {
        return Err(Error::Tls(format!("no certificate in {}", path.display())));
    }
    Ok(certs)
}
