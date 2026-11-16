package coordinator

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/trainmesh/trainmesh/internal/security"
)

// StateFile is the bbolt file inside the coordinator state directory.
const StateFile = "state.db"

// Certificate lifetimes for material the coordinator creates for itself.
const (
	CAValidity         = 10 * 365 * 24 * time.Hour
	ServerCertValidity = 365 * 24 * time.Hour
	AdminCertValidity  = 365 * 24 * time.Hour
)

// PrepareStateDir creates (or loads) the CA and the admin certificate in dir and issues a fresh
// server certificate for the advertise address plus localhost. It returns the CA and the
// server TLS config.
func PrepareStateDir(dir, advertiseAddress string, now time.Time) (*security.CA, *tls.Config, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, nil, fmt.Errorf("creating state dir %s: %w", dir, err)
	}
	ca, err := security.LoadOrCreateCA(dir, now, CAValidity)
	if err != nil {
		return nil, nil, err
	}
	if err := ensureAdminCert(dir, ca, now); err != nil {
		return nil, nil, err
	}
	host, _, err := net.SplitHostPort(advertiseAddress)
	if err != nil {
		return nil, nil, fmt.Errorf("advertise address %q: %w", advertiseAddress, err)
	}
	serverDER, serverKey, err := ca.IssueServer([]string{host, "localhost", "127.0.0.1"}, now, ServerCertValidity)
	if err != nil {
		return nil, nil, err
	}
	return ca, security.ServerTLS(serverDER, serverKey, ca.Cert), nil
}

func ensureAdminCert(dir string, ca *security.CA, now time.Time) error {
	certPath := filepath.Join(dir, security.AdminFile)
	if _, err := os.Stat(certPath); !errors.Is(err, os.ErrNotExist) {
		return err
	}
	der, key, err := ca.IssueClient(security.RoleAdmin, security.RoleAdmin, now, AdminCertValidity)
	if err != nil {
		return err
	}
	return security.WriteCertAndKey(certPath, filepath.Join(dir, security.AdminKeyFile), der, key)
}
