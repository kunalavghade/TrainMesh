//! The outer all-reduce between island leaders, with quorum and root failover.
//!
//! One round:
//! 1. The root is the lowest-numbered island still alive. Every other leader sends it a
//!    `Contribution` (its compressed pseudo-gradient plus a digest of its model state).
//! 2. The root waits until every alive island has contributed or the round deadline passes,
//!    checks that all digests match its own, averages, compresses the mean once and sends the
//!    same `Result` to everyone (lowest island first). Every leader, the root included, applies
//!    the decoded mean, so all islands stay bit-identical.
//! 3. A leader that loses its connection to the root, or gets no result in time, drops the root
//!    from its alive set and repeats the round with the next-lowest island as root. The new root
//!    answers a contribution for a round it already finished with the cached result.
//!
//! The group continues while at least `quorum` islands contribute. Use a majority quorum: then
//! two halves of a partitioned group can never both continue.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::ToSocketAddrs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustls::pki_types::CertificateDer;
use tokio::io::{AsyncRead, AsyncWrite, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio::time::timeout;
use tokio_rustls::{TlsAcceptor, TlsConnector};

use crate::codec::{decode_add, decode_into, Codec, Encoded, ErrorFeedback};
use crate::error::{Error, Result};
use crate::tls::{check_peer_name, server_name, Endpoints, TlsFiles};
use crate::wire::{read_message, write_message, Message, DIGEST_LEN};

const DIAL_RETRY: Duration = Duration::from_millis(200);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const RUNTIME_THREADS: usize = 2;

pub type Digest = [u8; DIGEST_LEN];

/// One island leader as every other leader sees it.
#[derive(Clone, Debug)]
pub struct Peer {
    pub island: u32,
    /// `host:port` its tmcomm listener is reachable at.
    pub address: String,
    /// Node ID: the TLS server name and the name its certificate must carry.
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct GroupConfig {
    pub island: u32,
    pub peers: Vec<Peer>,
    pub listen: String,
    /// None is plaintext, allowed only when every peer address is loopback.
    pub tls: Option<TlsFiles>,
    pub quorum: usize,
    pub codec: Codec,
    /// Largest vector a peer may send (the model size).
    pub max_values: usize,
    /// How long the root waits for contributions after the first round.
    pub round_timeout: Duration,
    /// The same for the first round, when islands may still be starting.
    pub startup_timeout: Duration,
    pub connect_timeout: Duration,
}

/// What one round did, from this island's point of view.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundReport {
    pub round: u64,
    pub root: u32,
    /// Islands whose contribution is in the mean.
    pub members: Vec<u32>,
    /// Islands the root still considers alive.
    pub alive: Vec<u32>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub seconds: f64,
}

trait Stream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Stream for T {}
type BoxStream = Box<dyn Stream>;
type Writers = Arc<Mutex<HashMap<u32, (u64, WriteHalf<BoxStream>)>>>;

enum Inbound {
    Message {
        island: u32,
        message: Message,
        bytes: u64,
    },
    Closed {
        island: u32,
        conn: u64,
    },
}

struct Upstream {
    root: u32,
    reader: ReadHalf<BoxStream>,
    writer: WriteHalf<BoxStream>,
}

/// Why a round attempt ended without a result.
enum Failure {
    /// The root failed: drop it and retry the round.
    Root(u32, Error),
    /// The job cannot continue.
    Fatal(Error),
}

impl From<Error> for Failure {
    fn from(err: Error) -> Self {
        Failure::Fatal(err)
    }
}

pub struct OuterGroup {
    config: GroupConfig,
    runtime: tokio::runtime::Runtime,
    connector: Option<TlsConnector>,
    alive: BTreeSet<u32>,
    inbox: mpsc::UnboundedReceiver<Inbound>,
    writers: Writers,
    upstream: Option<Upstream>,
    uplink: ErrorFeedback,
    downlink: ErrorFeedback,
    /// This island's encoded contribution for the round in progress (sent again on failover).
    pending: Option<(u64, Encoded)>,
    /// The last result this island sent as root, for members that retry that round.
    last_result: Option<(u64, Message)>,
    rounds_done: u64,
    accept_task: tokio::task::JoinHandle<()>,
}

impl OuterGroup {
    /// Validates the config, loads TLS material and starts listening.
    pub fn new(config: GroupConfig) -> Result<Self> {
        validate(&config)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(RUNTIME_THREADS)
            .thread_name("tmcomm")
            .enable_all()
            .build()?;
        let endpoints = config.tls.as_ref().map(Endpoints::load).transpose()?;
        let listener = runtime
            .block_on(TcpListener::bind(&config.listen))
            .map_err(|e| Error::Config(format!("listening on {}: {e}", config.listen)))?;
        let (inbox_tx, inbox) = mpsc::unbounded_channel();
        let writers: Writers = Arc::default();
        let context = Arc::new(AcceptContext {
            acceptor: endpoints.as_ref().map(|e| e.acceptor.clone()),
            names: config
                .peers
                .iter()
                .map(|p| (p.island, p.name.clone()))
                .collect(),
            writers: writers.clone(),
            inbox: inbox_tx,
            max_values: config.max_values,
            next_conn: AtomicU64::new(1),
        });
        let accept_task = runtime.spawn(accept_loop(listener, context));
        Ok(Self {
            alive: config.peers.iter().map(|p| p.island).collect(),
            connector: endpoints.map(|e| e.connector),
            config,
            runtime,
            inbox,
            writers,
            upstream: None,
            uplink: ErrorFeedback::new(),
            downlink: ErrorFeedback::new(),
            pending: None,
            last_result: None,
            rounds_done: 0,
            accept_task,
        })
    }

    pub fn alive(&self) -> Vec<u32> {
        self.alive.iter().copied().collect()
    }

    /// Replaces `values` with the mean over the islands that contribute this round.
    pub fn all_reduce_mean(
        &mut self,
        round: u64,
        values: &mut [f32],
        digest: Digest,
    ) -> Result<RoundReport> {
        if values.len() > self.config.max_values {
            return Err(Error::Config(format!(
                "{} values exceed max_values {}",
                values.len(),
                self.config.max_values
            )));
        }
        let started = Instant::now();
        loop {
            self.check_quorum(round)?;
            let root = *self
                .alive
                .first()
                .expect("quorum >= 1 keeps alive non-empty");
            let outcome = if root == self.config.island {
                self.run_root(round, values, digest, started)
            } else {
                self.run_member(root, round, values, digest, started)
            };
            match outcome {
                Ok(report) => {
                    self.rounds_done += 1;
                    return Ok(report);
                }
                Err(Failure::Root(root, err)) => {
                    eprintln!(
                        "tmcomm: island {} lost root island {root} in round {round} ({err}); electing the next root",
                        self.config.island
                    );
                    self.alive.remove(&root);
                    self.upstream = None;
                }
                Err(Failure::Fatal(err)) => return Err(err),
            }
        }
    }

    fn wait_timeout(&self) -> Duration {
        if self.rounds_done == 0 {
            self.config.startup_timeout
        } else {
            self.config.round_timeout
        }
    }

    fn check_quorum(&self, round: u64) -> Result<()> {
        if self.alive.len() < self.config.quorum {
            return Err(Error::QuorumLost {
                round,
                alive: self.alive.len(),
                quorum: self.config.quorum,
            });
        }
        Ok(())
    }

    fn run_root(
        &mut self,
        round: u64,
        values: &mut [f32],
        digest: Digest,
        started: Instant,
    ) -> std::result::Result<RoundReport, Failure> {
        let me = self.config.island;
        let deadline = Instant::now() + self.wait_timeout();
        let mut contributions: BTreeMap<u32, Encoded> = BTreeMap::new();
        let mut bytes_received = 0u64;
        loop {
            let waiting = self
                .alive
                .iter()
                .any(|i| *i != me && !contributions.contains_key(i));
            let remaining = deadline.saturating_duration_since(Instant::now());
            if !waiting || remaining.is_zero() {
                break;
            }
            let inbox = &mut self.inbox;
            let next = self
                .runtime
                .block_on(async { timeout(remaining, inbox.recv()).await });
            let Ok(next) = next else {
                break; // deadline
            };
            let Some(inbound) = next else {
                return Err(Error::Protocol("listener stopped".into()).into());
            };
            match inbound {
                Inbound::Closed { island, conn } => {
                    if self.forget_connection(island, conn) {
                        self.alive.remove(&island);
                        contributions.remove(&island);
                        if let Err(err) = self.check_quorum(round) {
                            self.abort_all(round, &err.to_string());
                            return Err(err.into());
                        }
                    }
                }
                Inbound::Message {
                    island,
                    message:
                        Message::Contribution {
                            round: their_round,
                            island: from,
                            digest: their_digest,
                            payload,
                        },
                    bytes,
                } => {
                    if from != island {
                        self.send_to(island, &abort(round, "contribution names another island"));
                    } else if their_round < round {
                        self.resend_result(island, their_round);
                    } else if their_round > round {
                        let err = Error::Protocol(format!(
                            "island {island} is in round {their_round} but the root is in round {round}"
                        ));
                        self.abort_all(round, &err.to_string());
                        return Err(err.into());
                    } else if !self.alive.contains(&island) {
                        self.send_to(island, &abort(round, "excluded: declared dead earlier"));
                    } else if their_digest != digest {
                        let err = Error::Diverged { round, island };
                        self.abort_all(round, &err.to_string());
                        return Err(err.into());
                    } else {
                        bytes_received += bytes;
                        contributions.insert(island, payload);
                    }
                }
                Inbound::Message { .. } => {} // results/aborts are only for members
            }
        }

        // Islands that never connected by the deadline are treated as dead; connected but slow
        // ones stay alive and simply miss this round.
        let connected: BTreeSet<u32> = self
            .runtime
            .block_on(self.writers.lock())
            .keys()
            .copied()
            .collect();
        self.alive
            .retain(|i| *i == me || contributions.contains_key(i) || connected.contains(i));

        let mut members: Vec<u32> = contributions.keys().copied().collect();
        members.push(me);
        members.sort_unstable();
        if members.len() < self.config.quorum {
            let err = Error::QuorumLost {
                round,
                alive: members.len(),
                quorum: self.config.quorum,
            };
            self.abort_all(round, &err.to_string());
            return Err(err.into());
        }

        let mut sum = values.to_vec();
        for payload in contributions.values() {
            decode_add(payload, &mut sum)?;
        }
        let count = members.len() as f32;
        sum.iter_mut().for_each(|v| *v /= count);
        let encoded = self.downlink.compress(self.config.codec, &sum)?;
        decode_into(&encoded, values)?;

        let result = Message::Result {
            round,
            members: members.clone(),
            alive: self.alive(),
            payload: encoded,
        };
        let mut bytes_sent = 0u64;
        // Lowest island first: if the root dies mid-broadcast, the next root has the result.
        let others: Vec<u32> = self.alive.iter().copied().filter(|i| *i != me).collect();
        for island in others {
            match self.send_to(island, &result) {
                Some(bytes) => bytes_sent += bytes,
                None => {
                    self.alive.remove(&island);
                }
            }
        }
        self.last_result = Some((round, result));
        self.pending = None;
        Ok(RoundReport {
            round,
            root: me,
            members,
            alive: self.alive(),
            bytes_sent,
            bytes_received,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn run_member(
        &mut self,
        root: u32,
        round: u64,
        values: &mut [f32],
        digest: Digest,
        started: Instant,
    ) -> std::result::Result<RoundReport, Failure> {
        let me = self.config.island;
        if self.upstream.as_ref().is_some_and(|u| u.root != root) {
            self.upstream = None;
        }
        if self.upstream.is_none() {
            let peer = self.peer(root)?.clone();
            let patience = if self.rounds_done == 0 {
                self.config.startup_timeout
            } else {
                self.config.connect_timeout
            };
            let dialled = self.runtime.block_on(dial(
                &peer,
                self.connector.as_ref(),
                me,
                Instant::now() + patience,
                self.config.connect_timeout,
            ));
            self.upstream = Some(dialled.map_err(|e| Failure::Root(root, e))?);
        }
        if self.pending.as_ref().is_none_or(|(r, _)| *r != round) {
            let encoded = self.uplink.compress(self.config.codec, values)?;
            self.pending = Some((round, encoded));
        }
        let contribution = Message::Contribution {
            round,
            island: me,
            digest,
            payload: self.pending.as_ref().expect("set above").1.clone(),
        };
        // The root may wait a full round for slower islands before it answers.
        let deadline = Instant::now() + self.wait_timeout() * 2 + self.config.connect_timeout;
        let max_values = self.config.max_values;
        let upstream = self.upstream.as_mut().expect("dialled above");
        let bytes_sent = self
            .runtime
            .block_on(write_message(&mut upstream.writer, &contribution))
            .map_err(|e| Failure::Root(root, e))?;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let reader = &mut upstream.reader;
            let read = self
                .runtime
                .block_on(async { timeout(remaining, read_message(reader, max_values)).await });
            let (message, bytes) = match read {
                Err(_) => {
                    return Err(Failure::Root(
                        root,
                        Error::Protocol(format!("no result within {remaining:?}")),
                    ))
                }
                Ok(Err(err)) => return Err(Failure::Root(root, err)),
                Ok(Ok(frame)) => frame,
            };
            match message {
                Message::Result {
                    round: their_round,
                    members,
                    alive,
                    payload,
                } if their_round == round => {
                    decode_into(&payload, values)?;
                    if !alive.contains(&me) {
                        return Err(Error::Excluded {
                            round,
                            reason: "the root no longer counts this island as alive".into(),
                        }
                        .into());
                    }
                    self.alive = alive.iter().copied().collect();
                    self.pending = None;
                    return Ok(RoundReport {
                        round,
                        root,
                        members,
                        alive,
                        bytes_sent,
                        bytes_received: bytes,
                        seconds: started.elapsed().as_secs_f64(),
                    });
                }
                Message::Result { round: r, .. } if r < round => {} // duplicate of an old round
                Message::Abort { round, reason } if reason.starts_with("excluded") => {
                    return Err(Error::Excluded { round, reason }.into())
                }
                Message::Abort { round, reason } => {
                    return Err(Error::Aborted { round, reason }.into())
                }
                other => {
                    return Err(Error::Protocol(format!(
                        "unexpected frame from the root: {other:?}"
                    ))
                    .into())
                }
            }
        }
    }

    fn peer(&self, island: u32) -> Result<&Peer> {
        self.config
            .peers
            .iter()
            .find(|p| p.island == island)
            .ok_or_else(|| Error::Config(format!("no peer for island {island}")))
    }

    /// Drops `island`'s writer if `conn` is still its current connection.
    fn forget_connection(&mut self, island: u32, conn: u64) -> bool {
        let mut writers = self.runtime.block_on(self.writers.lock());
        if writers
            .get(&island)
            .is_some_and(|(current, _)| *current == conn)
        {
            writers.remove(&island);
            return true;
        }
        false
    }

    /// Sends to an island that connected to us; None if it is not connected or the write failed.
    fn send_to(&mut self, island: u32, message: &Message) -> Option<u64> {
        let writers = self.writers.clone();
        self.runtime.block_on(async move {
            let mut writers = writers.lock().await;
            let (_, writer) = writers.get_mut(&island)?;
            match write_message(writer, message).await {
                Ok(bytes) => Some(bytes),
                Err(_) => {
                    writers.remove(&island);
                    None
                }
            }
        })
    }

    fn resend_result(&mut self, island: u32, round: u64) {
        let reply = match &self.last_result {
            Some((done, result)) if *done == round => result.clone(),
            _ => abort(round, "the root has no result for that round"),
        };
        self.send_to(island, &reply);
    }

    fn abort_all(&mut self, round: u64, reason: &str) {
        let islands: Vec<u32> = self
            .runtime
            .block_on(self.writers.lock())
            .keys()
            .copied()
            .collect();
        for island in islands {
            self.send_to(island, &abort(round, reason));
        }
    }
}

impl Drop for OuterGroup {
    fn drop(&mut self) {
        self.accept_task.abort();
    }
}

fn abort(round: u64, reason: &str) -> Message {
    Message::Abort {
        round,
        reason: reason.to_owned(),
    }
}

fn validate(config: &GroupConfig) -> Result<()> {
    let islands: BTreeSet<u32> = config.peers.iter().map(|p| p.island).collect();
    if islands.len() != config.peers.len() {
        return Err(Error::Config("two peers share an island number".into()));
    }
    if !islands.contains(&config.island) {
        return Err(Error::Config(format!(
            "island {} is not in the peer list",
            config.island
        )));
    }
    if config.quorum == 0 || config.quorum > config.peers.len() {
        return Err(Error::Config(format!(
            "quorum {} must be between 1 and the {} islands",
            config.quorum,
            config.peers.len()
        )));
    }
    if config.tls.is_none() {
        for peer in &config.peers {
            let loopback = peer
                .address
                .to_socket_addrs()
                .map(|mut addrs| addrs.all(|a| a.ip().is_loopback()))
                .unwrap_or(false);
            if !loopback {
                return Err(Error::Config(format!(
                    "plaintext is only allowed on loopback; peer {} is at {} (pass TLS files)",
                    peer.island, peer.address
                )));
            }
        }
    }
    Ok(())
}

struct AcceptContext {
    acceptor: Option<TlsAcceptor>,
    names: HashMap<u32, String>,
    writers: Writers,
    inbox: mpsc::UnboundedSender<Inbound>,
    max_values: usize,
    next_conn: AtomicU64,
}

async fn accept_loop(listener: TcpListener, context: Arc<AcceptContext>) {
    loop {
        match listener.accept().await {
            Ok((tcp, _)) => {
                let context = context.clone();
                tokio::spawn(async move {
                    if let Err(err) = serve(tcp, context).await {
                        eprintln!("tmcomm: refused a connection: {err}");
                    }
                });
            }
            Err(_) => tokio::time::sleep(DIAL_RETRY).await,
        }
    }
}

/// Authenticates one incoming connection, registers its writer and forwards its frames.
async fn serve(tcp: TcpStream, context: Arc<AcceptContext>) -> Result<()> {
    tcp.set_nodelay(true)?;
    let (stream, peer_cert): (BoxStream, Option<CertificateDer<'static>>) = match &context.acceptor
    {
        Some(acceptor) => {
            let tls = timeout(HANDSHAKE_TIMEOUT, acceptor.accept(tcp))
                .await
                .map_err(|_| Error::Tls("handshake timed out".into()))??;
            let cert = tls
                .get_ref()
                .1
                .peer_certificates()
                .and_then(|certs| certs.first())
                .map(|cert| cert.clone().into_owned());
            (Box::new(tls), cert)
        }
        None => (Box::new(tcp), None),
    };
    let (mut reader, writer) = tokio::io::split(stream);
    let (hello, _) = timeout(
        HANDSHAKE_TIMEOUT,
        read_message(&mut reader, context.max_values),
    )
    .await
    .map_err(|_| Error::Protocol("no hello".into()))??;
    let Message::Hello { island } = hello else {
        return Err(Error::Protocol("first frame was not a hello".into()));
    };
    let name = context
        .names
        .get(&island)
        .ok_or_else(|| Error::Protocol(format!("unknown island {island}")))?;
    match (&context.acceptor, &peer_cert) {
        (Some(_), Some(cert)) => check_peer_name(cert, name)?,
        (Some(_), None) => return Err(Error::Tls("no client certificate".into())),
        (None, _) => {}
    }
    let conn = context.next_conn.fetch_add(1, Ordering::Relaxed);
    context.writers.lock().await.insert(island, (conn, writer));
    loop {
        match read_message(&mut reader, context.max_values).await {
            Ok((message, bytes)) => {
                let inbound = Inbound::Message {
                    island,
                    message,
                    bytes,
                };
                if context.inbox.send(inbound).is_err() {
                    return Ok(());
                }
            }
            Err(_) => {
                let _ = context.inbox.send(Inbound::Closed { island, conn });
                return Ok(());
            }
        }
    }
}

/// Connects to the root and says hello, retrying until `deadline` (the root may still be
/// starting).
async fn dial(
    peer: &Peer,
    connector: Option<&TlsConnector>,
    me: u32,
    deadline: Instant,
    connect_timeout: Duration,
) -> Result<Upstream> {
    loop {
        let attempt = async {
            let tcp = TcpStream::connect(&peer.address).await?;
            tcp.set_nodelay(true)?;
            let stream: BoxStream = match connector {
                Some(connector) => {
                    Box::new(connector.connect(server_name(&peer.name)?, tcp).await?)
                }
                None => Box::new(tcp),
            };
            let (reader, mut writer) = tokio::io::split(stream);
            write_message(&mut writer, &Message::Hello { island: me }).await?;
            Ok::<_, Error>(Upstream {
                root: peer.island,
                reader,
                writer,
            })
        };
        let remaining = deadline.saturating_duration_since(Instant::now());
        let error = match timeout(connect_timeout.min(remaining.max(DIAL_RETRY)), attempt).await {
            Ok(Ok(upstream)) => return Ok(upstream),
            Ok(Err(err)) => err,
            Err(_) => Error::Protocol(format!("connecting to {} timed out", peer.address)),
        };
        if Instant::now() + DIAL_RETRY >= deadline {
            return Err(error);
        }
        tokio::time::sleep(DIAL_RETRY).await;
    }
}
