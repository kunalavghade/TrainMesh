//! Real leaders over loopback TCP, each on its own thread, as in a job.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use rcgen::{BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair};
use tmcomm::{Codec, Error, GroupConfig, OuterGroup, Peer, RoundReport, TlsFiles};

const SAME: [u8; 32] = [7; 32];
static DIR_COUNTER: AtomicU32 = AtomicU32::new(0);

type IslandOutcome = Result<(Vec<f32>, Vec<RoundReport>), Error>;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn peers(count: u32) -> Vec<Peer> {
    (0..count)
        .map(|island| Peer {
            island,
            address: format!("127.0.0.1:{}", free_port()),
            name: format!("node-{island}"),
        })
        .collect()
}

fn config(island: u32, peers: &[Peer], quorum: usize, codec: Codec) -> GroupConfig {
    GroupConfig {
        island,
        peers: peers.to_vec(),
        listen: peers[island as usize].address.clone(),
        tls: None,
        quorum,
        codec,
        max_values: 4096,
        round_timeout: Duration::from_secs(5),
        startup_timeout: Duration::from_secs(10),
        connect_timeout: Duration::from_secs(2),
    }
}

/// Runs `rounds` rounds on every island in parallel; island i contributes i+1 everywhere.
fn run_islands(configs: Vec<GroupConfig>, rounds: u64, len: usize) -> Vec<IslandOutcome> {
    let handles: Vec<_> = configs
        .into_iter()
        .map(|config| {
            thread::spawn(move || {
                let island = config.island;
                let mut group = OuterGroup::new(config)?;
                let mut reports = Vec::new();
                let mut values = vec![0.0f32; len];
                for round in 0..rounds {
                    values.iter_mut().for_each(|v| *v = (island + 1) as f32);
                    reports.push(group.all_reduce_mean(round, &mut values, SAME)?);
                }
                Ok((values, reports))
            })
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

#[test]
fn three_islands_agree_on_the_exact_mean() {
    let peers = peers(3);
    let configs = (0..3).map(|i| config(i, &peers, 2, Codec::F32)).collect();
    for outcome in run_islands(configs, 3, 3000) {
        let (values, reports) = outcome.unwrap();
        assert!(values.iter().all(|v| *v == 2.0), "mean of 1, 2, 3");
        for report in reports {
            assert_eq!(report.members, vec![0, 1, 2]);
            assert_eq!(report.root, 0);
        }
    }
}

#[test]
fn int8_mean_is_identical_on_every_island() {
    let peers = peers(3);
    let configs = (0..3).map(|i| config(i, &peers, 2, Codec::Int8)).collect();
    let results: Vec<_> = run_islands(configs, 2, 2500)
        .into_iter()
        .map(|o| o.unwrap())
        .collect();
    for (values, reports) in &results {
        assert_eq!(
            values, &results[0].0,
            "islands must apply bit-identical updates"
        );
        assert!(values.iter().all(|v| (v - 2.0).abs() < 0.02));
        let member = reports
            .iter()
            .find(|r| r.root != 0 || r.bytes_sent > 0)
            .unwrap();
        assert!(member.bytes_sent > 0);
    }
    // A member sends ~1 byte per value (+ scales and headers), not 4.
    let (_, member_reports) = &results[1];
    assert!(
        member_reports[0].bytes_sent < 2600,
        "{}",
        member_reports[0].bytes_sent
    );
}

#[test]
fn losing_the_root_elects_the_next_island() {
    let peers = peers(3);
    let handles: Vec<_> = (0..3u32)
        .map(|island| {
            let config = config(island, &peers, 2, Codec::F32);
            thread::spawn(move || -> Result<Vec<RoundReport>, Error> {
                let mut group = OuterGroup::new(config)?;
                let mut reports = Vec::new();
                for round in 0..3 {
                    if island == 0 && round == 1 {
                        drop(group); // the root's machine goes away after round 0
                        return Ok(reports);
                    }
                    let mut values = vec![(island + 1) as f32; 100];
                    reports.push(group.all_reduce_mean(round, &mut values, SAME)?);
                    if round > 0 {
                        assert!(values.iter().all(|v| *v == 2.5), "mean of 2 and 3");
                    }
                }
                Ok(reports)
            })
        })
        .collect();
    let outcomes: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().unwrap().unwrap())
        .collect();
    for reports in &outcomes[1..] {
        assert_eq!(reports[0].members, vec![0, 1, 2]);
        assert_eq!(reports[1].root, 1);
        assert_eq!(reports[1].members, vec![1, 2]);
        assert_eq!(reports[2].alive, vec![1, 2]);
    }
}

#[test]
fn losing_quorum_stops_the_survivor() {
    let peers = peers(3);
    let handles: Vec<_> = (0..3u32)
        .map(|island| {
            let config = config(island, &peers, 2, Codec::F32);
            thread::spawn(move || -> Result<(), Error> {
                let mut group = OuterGroup::new(config)?;
                let mut values = vec![1.0f32; 10];
                group.all_reduce_mean(0, &mut values, SAME)?;
                if island < 2 {
                    return Ok(()); // islands 0 and 1 disappear
                }
                group.all_reduce_mean(1, &mut values, SAME)?;
                Ok(())
            })
        })
        .collect();
    let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(outcomes[0].is_ok() && outcomes[1].is_ok());
    assert!(
        matches!(outcomes[2], Err(Error::QuorumLost { round: 1, .. })),
        "{:?}",
        outcomes[2]
    );
}

#[test]
fn a_diverged_island_stops_the_round() {
    let peers = peers(2);
    let handles: Vec<_> = (0..2u32)
        .map(|island| {
            let config = config(island, &peers, 2, Codec::F32);
            thread::spawn(move || {
                let mut group = OuterGroup::new(config)?;
                let mut values = vec![1.0f32; 10];
                let digest = [island as u8; 32];
                group.all_reduce_mean(0, &mut values, digest)
            })
        })
        .collect();
    let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(
        matches!(outcomes[0], Err(Error::Diverged { island: 1, .. })),
        "{:?}",
        outcomes[0]
    );
    assert!(
        matches!(outcomes[1], Err(Error::Aborted { .. })),
        "{:?}",
        outcomes[1]
    );
}

#[test]
fn plaintext_to_a_remote_address_is_refused() {
    let mut peers = peers(2);
    peers[1].address = "10.1.2.3:29600".into();
    let err = OuterGroup::new(config(0, &peers, 1, Codec::F32))
        .err()
        .unwrap();
    assert!(matches!(err, Error::Config(_)), "{err}");
}

struct Pki {
    dir: PathBuf,
    ca: rcgen::Certificate,
    ca_key: KeyPair,
}

impl Pki {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "tmcomm-pki-{}-{}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let ca_key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca = params.self_signed(&ca_key).unwrap();
        std::fs::write(dir.join("ca.pem"), ca.pem()).unwrap();
        Self { dir, ca, ca_key }
    }

    /// Files for a node whose certificate names `cert_name`.
    fn node(&self, label: &str, cert_name: &str) -> TlsFiles {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(vec![cert_name.to_owned()]).unwrap();
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];
        let cert = params.signed_by(&key, &self.ca, &self.ca_key).unwrap();
        let files = TlsFiles {
            cert: self.dir.join(format!("{label}.pem")),
            key: self.dir.join(format!("{label}.key")),
            ca: self.dir.join("ca.pem"),
        };
        std::fs::write(&files.cert, cert.pem()).unwrap();
        std::fs::write(&files.key, key.serialize_pem()).unwrap();
        files
    }
}

#[test]
fn mutual_tls_with_the_cluster_ca() {
    let pki = Pki::new();
    let peers = peers(2);
    let configs = (0..2u32)
        .map(|i| GroupConfig {
            tls: Some(pki.node(&format!("n{i}"), &format!("node-{i}"))),
            ..config(i, &peers, 2, Codec::Bf16)
        })
        .collect();
    for outcome in run_islands(configs, 2, 512) {
        let (values, _) = outcome.unwrap();
        assert!(values.iter().all(|v| *v == 1.5));
    }
}

#[test]
fn a_certificate_for_another_node_is_refused() {
    let pki = Pki::new();
    let peers = peers(2);
    let mut configs: Vec<GroupConfig> = (0..2u32)
        .map(|i| GroupConfig {
            tls: Some(pki.node(&format!("n{i}"), &format!("node-{i}"))),
            ..config(i, &peers, 2, Codec::F32)
        })
        .collect();
    // Island 1 presents a valid cluster certificate, but one issued to a different node.
    configs[1].tls = Some(pki.node("imposter", "node-9"));
    configs[0].startup_timeout = Duration::from_secs(2);
    configs[1].startup_timeout = Duration::from_secs(2);
    let outcomes = run_islands(configs, 1, 16);
    assert!(
        matches!(outcomes[0], Err(Error::QuorumLost { .. })),
        "the root must not accept the imposter: {:?}",
        outcomes[0]
    );
    assert!(outcomes[1].is_err());
}
