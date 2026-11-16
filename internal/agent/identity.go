// Package agent implements tm-agent: it joins a cluster once, then heartbeats to the
// coordinator, runs the job assignments it receives and reports their outcome.
package agent

import (
	"crypto"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/trainmesh/trainmesh/internal/security"
)

// Files in the agent state directory.
const (
	identityFile = "node.json"
	nodeCertFile = "node.pem"
	nodeKeyFile  = "node-key.pem"
)

// ErrNotJoined means the state directory holds no identity and no join string was given.
var ErrNotJoined = errors.New("this machine has not joined a cluster: pass --join <join string>")

// Identity is what an agent keeps after joining.
type Identity struct {
	NodeID      string `json:"node_id"`
	Coordinator string `json:"coordinator"`

	CA      *x509.Certificate `json:"-"`
	CertDER []byte            `json:"-"`
	Key     crypto.Signer     `json:"-"`
}

// LoadIdentity reads a saved identity; it returns ErrNotJoined when there is none.
func LoadIdentity(dir string) (Identity, error) {
	raw, err := os.ReadFile(filepath.Join(dir, identityFile)) // #nosec G304 -- agent's own state dir
	if errors.Is(err, os.ErrNotExist) {
		return Identity{}, ErrNotJoined
	}
	if err != nil {
		return Identity{}, fmt.Errorf("reading identity: %w", err)
	}
	var identity Identity
	if err := json.Unmarshal(raw, &identity); err != nil {
		return Identity{}, fmt.Errorf("parsing %s: %w", identityFile, err)
	}
	if identity.CA, err = security.ReadCert(filepath.Join(dir, security.CAFile)); err != nil {
		return Identity{}, err
	}
	cert, key, err := security.ReadCertAndKey(filepath.Join(dir, nodeCertFile), filepath.Join(dir, nodeKeyFile))
	if err != nil {
		return Identity{}, err
	}
	identity.CertDER, identity.Key = cert.Raw, key
	return identity, nil
}

// Save writes the identity files (key owner-only).
func (i Identity) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("creating state dir %s: %w", dir, err)
	}
	if err := security.WriteCert(filepath.Join(dir, security.CAFile), i.CA.Raw); err != nil {
		return err
	}
	if err := security.WriteCertAndKey(filepath.Join(dir, nodeCertFile), filepath.Join(dir, nodeKeyFile), i.CertDER, i.Key); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(i, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, identityFile), append(raw, '\n'), 0o600)
}
