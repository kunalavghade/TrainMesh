package security

import (
	"context"
	"crypto"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"slices"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
)

// ErrUnauthenticated means the caller presented no usable client certificate.
var ErrUnauthenticated = errors.New("no verified client certificate")

// Identity is who a verified client certificate says the caller is.
type Identity struct {
	Name string // node ID, or "admin"
	Role string // RoleNode or RoleAdmin
}

// ServerTLS accepts clients with or without a certificate; per-RPC authorization decides which
// calls need one (only Join works without).
func ServerTLS(certDER []byte, key crypto.Signer, ca *x509.Certificate) *tls.Config {
	pool := x509.NewCertPool()
	pool.AddCert(ca)
	return &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{{Certificate: [][]byte{certDER, ca.Raw}, PrivateKey: key}},
		ClientCAs:    pool,
		ClientAuth:   tls.VerifyClientCertIfGiven,
	}
}

// ClientTLS trusts servers whose certificate chains to ca and has the server-auth usage. The
// host name is not checked: the cluster CA is the trust anchor, and node addresses change.
// A nil certificate makes an anonymous client (used only for Join).
func ClientTLS(ca *x509.Certificate, certDER []byte, key crypto.Signer) *tls.Config {
	config := &tls.Config{
		MinVersion:         tls.VersionTLS13,
		InsecureSkipVerify: true, // #nosec G402 -- replaced by VerifyConnection against the cluster CA
		VerifyConnection:   verifyAgainstCA(ca),
	}
	if certDER != nil {
		config.Certificates = []tls.Certificate{{Certificate: [][]byte{certDER}, PrivateKey: key}}
	}
	return config
}

// PinnedTLS is used before the agent has the CA certificate: it accepts the server only if its
// chain contains a CA whose hash equals the one from the join string, and the leaf chains to it.
func PinnedTLS(caHash string) *tls.Config {
	return &tls.Config{
		MinVersion:         tls.VersionTLS13,
		InsecureSkipVerify: true, // #nosec G402 -- replaced by VerifyConnection with the pinned CA hash
		VerifyConnection: func(state tls.ConnectionState) error {
			index := slices.IndexFunc(state.PeerCertificates, func(c *x509.Certificate) bool {
				return CertificateHash(c.Raw) == caHash
			})
			if index < 0 {
				return errors.New("coordinator CA does not match the join string's ca hash")
			}
			return verifyAgainstCA(state.PeerCertificates[index])(state)
		},
	}
}

func verifyAgainstCA(ca *x509.Certificate) func(tls.ConnectionState) error {
	pool := x509.NewCertPool()
	pool.AddCert(ca)
	return func(state tls.ConnectionState) error {
		if len(state.PeerCertificates) == 0 {
			return errors.New("server sent no certificate")
		}
		_, err := state.PeerCertificates[0].Verify(x509.VerifyOptions{
			Roots:     pool,
			KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		})
		if err != nil {
			return fmt.Errorf("coordinator certificate not issued by the cluster CA: %w", err)
		}
		return nil
	}
}

// IdentityFromContext returns the verified client identity of a gRPC call.
func IdentityFromContext(ctx context.Context) (Identity, error) {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return Identity{}, ErrUnauthenticated
	}
	info, ok := p.AuthInfo.(credentials.TLSInfo)
	if !ok || len(info.State.VerifiedChains) == 0 || len(info.State.VerifiedChains[0]) == 0 {
		return Identity{}, ErrUnauthenticated
	}
	leaf := info.State.VerifiedChains[0][0]
	if len(leaf.Subject.OrganizationalUnit) != 1 {
		return Identity{}, fmt.Errorf("%w: certificate has no role", ErrUnauthenticated)
	}
	return Identity{Name: leaf.Subject.CommonName, Role: leaf.Subject.OrganizationalUnit[0]}, nil
}
