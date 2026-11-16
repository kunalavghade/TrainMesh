package store

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	bolt "go.etcd.io/bbolt"

	"github.com/trainmesh/trainmesh/internal/security"
)

const (
	schemaVersion   = "1"
	fileMode        = 0o600
	openLockTimeout = 2 * time.Second
)

var (
	bucketMeta   = []byte("meta")
	bucketTokens = []byte("tokens")
	bucketNodes  = []byte("nodes")
	bucketJobs   = []byte("jobs")
	keySchema    = []byte("schema_version")
)

// BoltStore is a Store backed by a single bbolt file.
type BoltStore struct {
	db *bolt.DB
}

// OpenBolt opens (or creates) the state file. It fails fast if another process holds it.
func OpenBolt(path string) (*BoltStore, error) {
	db, err := bolt.Open(path, fileMode, &bolt.Options{Timeout: openLockTimeout})
	if err != nil {
		return nil, fmt.Errorf("opening state file %s (is another meshd running?): %w", path, err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		for _, name := range [][]byte{bucketMeta, bucketTokens, bucketNodes, bucketJobs} {
			if _, err := tx.CreateBucketIfNotExists(name); err != nil {
				return err
			}
		}
		meta := tx.Bucket(bucketMeta)
		if existing := meta.Get(keySchema); existing != nil && string(existing) != schemaVersion {
			return fmt.Errorf("state file schema %s is not supported (want %s)", existing, schemaVersion)
		}
		return meta.Put(keySchema, []byte(schemaVersion))
	})
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("initialising state file %s: %w", path, err)
	}
	return &BoltStore{db: db}, nil
}

// View runs fn in a read-only transaction.
func (s *BoltStore) View(fn func(Tx) error) error {
	return s.db.View(func(tx *bolt.Tx) error { return fn(boltTx{tx}) })
}

// Update runs fn in a read-write transaction.
func (s *BoltStore) Update(fn func(Tx) error) error {
	return s.db.Update(func(tx *bolt.Tx) error { return fn(boltTx{tx}) })
}

// Close releases the state file.
func (s *BoltStore) Close() error { return s.db.Close() }

type boltTx struct{ tx *bolt.Tx }

func (t boltTx) Token(hash string) (security.TokenRecord, error) {
	var record security.TokenRecord
	return record, t.get(bucketTokens, hash, &record)
}

func (t boltTx) PutToken(record security.TokenRecord) error {
	return t.put(bucketTokens, record.Hash, record)
}

func (t boltTx) Node(id string) (Node, error) {
	var node Node
	return node, t.get(bucketNodes, id, &node)
}

func (t boltTx) PutNode(node Node) error { return t.put(bucketNodes, node.ID, node) }

func (t boltTx) Nodes() ([]Node, error) { return list[Node](t.tx.Bucket(bucketNodes)) }

func (t boltTx) Job(id string) (Job, error) {
	var job Job
	return job, t.get(bucketJobs, id, &job)
}

func (t boltTx) PutJob(job Job) error { return t.put(bucketJobs, job.ID, job) }

func (t boltTx) Jobs() ([]Job, error) { return list[Job](t.tx.Bucket(bucketJobs)) }

func (t boltTx) get(bucket []byte, key string, into any) error {
	raw := t.tx.Bucket(bucket).Get([]byte(key))
	if raw == nil {
		return fmt.Errorf("%s %q: %w", bucket, key, ErrNotFound)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("decoding %s %q: %w", bucket, key, err)
	}
	return nil
}

func (t boltTx) put(bucket []byte, key string, value any) error {
	if key == "" {
		return fmt.Errorf("%s: empty key", bucket)
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encoding %s %q: %w", bucket, key, err)
	}
	return t.tx.Bucket(bucket).Put([]byte(key), raw)
}

// list decodes every record in a bucket, ordered by key.
func list[T any](bucket *bolt.Bucket) ([]T, error) {
	var keys []string
	records := map[string]T{}
	err := bucket.ForEach(func(key, raw []byte) error {
		var record T
		if err := json.Unmarshal(raw, &record); err != nil {
			return fmt.Errorf("decoding %q: %w", key, err)
		}
		keys = append(keys, string(key))
		records[string(key)] = record
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(keys)
	out := make([]T, 0, len(keys))
	for _, key := range keys {
		out = append(out, records[key])
	}
	return out, nil
}
