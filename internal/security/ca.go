package security

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"
)

// Roles carried in a client certificate's OrganizationalUnit.
const (
	RoleNode  = "node"
	RoleAdmin = "admin"
)

// CoordinatorCommonName is the subject of meshd's server certificate. Clients require it, so a
// member certificate (which can also authenticate a tmcomm server) cannot pose as meshd.
const CoordinatorCommonName = "meshd"

const (
	caCommonName    = "trainmesh-cluster-ca"
	certFileMode    = 0o644
	keyFileMode     = 0o600
	serialBits      = 128
	clockSkewMargin = 5 * time.Minute
)

// File names inside a state directory.
const (
	CAFile        = "ca.pem"
	CAKeyFile     = "ca-key.pem"
	ServerFile    = "server.pem"
	ServerKeyFile = "server-key.pem"
	AdminFile     = "admin.pem"
	AdminKeyFile  = "admin-key.pem"
)

// ErrCSRInvalid is returned when a join request carries an unusable CSR.
var ErrCSRInvalid = errors.New("certificate signing request is invalid")

// CA is the cluster certificate authority owned by the coordinator.
type CA struct {
	Cert *x509.Certificate
	key  crypto.Signer
}

// NewCA creates a self-signed CA valid for `validity`.
func NewCA(now time.Time, validity time.Duration) (*CA, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generating CA key: %w", err)
	}
	template, err := baseTemplate(now, validity, pkix.Name{CommonName: caCommonName})
	if err != nil {
		return nil, err
	}
	template.IsCA = true
	template.BasicConstraintsValid = true
	template.KeyUsage = x509.KeyUsageCertSign | x509.KeyUsageCRLSign
	der, err := x509.CreateCertificate(rand.Reader, template, template, key.Public(), key)
	if err != nil {
		return nil, fmt.Errorf("self-signing CA: %w", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		return nil, err
	}
	return &CA{Cert: cert, key: key}, nil
}

// LoadOrCreateCA reads ca.pem/ca-key.pem from dir, creating them on first use.
func LoadOrCreateCA(dir string, now time.Time, validity time.Duration) (*CA, error) {
	certPath, keyPath := filepath.Join(dir, CAFile), filepath.Join(dir, CAKeyFile)
	if _, err := os.Stat(certPath); errors.Is(err, os.ErrNotExist) {
		ca, err := NewCA(now, validity)
		if err != nil {
			return nil, err
		}
		if err := WriteCertAndKey(certPath, keyPath, ca.Cert.Raw, ca.key); err != nil {
			return nil, err
		}
		return ca, nil
	}
	cert, key, err := ReadCertAndKey(certPath, keyPath)
	if err != nil {
		return nil, err
	}
	return &CA{Cert: cert, key: key}, nil
}

// Hash is the CA fingerprint used in join strings.
func (ca *CA) Hash() string { return CertificateHash(ca.Cert.Raw) }

// IssueServer creates a server key and certificate for the given hosts (names or IPs).
func (ca *CA) IssueServer(hosts []string, now time.Time, validity time.Duration) ([]byte, crypto.Signer, error) {
	key, err := GenerateKey()
	if err != nil {
		return nil, nil, err
	}
	template, err := baseTemplate(now, validity, pkix.Name{CommonName: CoordinatorCommonName})
	if err != nil {
		return nil, nil, err
	}
	template.KeyUsage = x509.KeyUsageDigitalSignature
	template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
	for _, host := range hosts {
		if ip := net.ParseIP(host); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		} else {
			template.DNSNames = append(template.DNSNames, host)
		}
	}
	der, err := x509.CreateCertificate(rand.Reader, template, ca.Cert, key.Public(), ca.key)
	if err != nil {
		return nil, nil, fmt.Errorf("issuing server certificate: %w", err)
	}
	return der, key, nil
}

// IssueClient creates a key and a client certificate for (commonName, role).
func (ca *CA) IssueClient(commonName, role string, now time.Time, validity time.Duration) ([]byte, crypto.Signer, error) {
	key, err := GenerateKey()
	if err != nil {
		return nil, nil, err
	}
	der, err := ca.signClient(key.Public(), commonName, role, now, validity)
	return der, key, err
}

// SignCSR issues a client certificate for the CSR's key. The identity comes from the caller,
// never from the CSR, so an agent cannot choose its own node ID or role.
func (ca *CA) SignCSR(csrDER []byte, commonName, role string, now time.Time, validity time.Duration) ([]byte, error) {
	csr, err := x509.ParseCertificateRequest(csrDER)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrCSRInvalid, err)
	}
	if err := csr.CheckSignature(); err != nil {
		return nil, fmt.Errorf("%w: signature: %w", ErrCSRInvalid, err)
	}
	return ca.signClient(csr.PublicKey, commonName, role, now, validity)
}

func (ca *CA) signClient(publicKey any, commonName, role string, now time.Time, validity time.Duration) ([]byte, error) {
	template, err := baseTemplate(now, validity, pkix.Name{CommonName: commonName, OrganizationalUnit: []string{role}})
	if err != nil {
		return nil, err
	}
	template.KeyUsage = x509.KeyUsageDigitalSignature
	template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	if role == RoleNode {
		// Island leaders (tmcomm) accept connections from each other and dial each other by node
		// ID, so node certificates also authenticate servers and name the node as a DNS SAN.
		template.ExtKeyUsage = append(template.ExtKeyUsage, x509.ExtKeyUsageServerAuth)
		template.DNSNames = []string{commonName}
	}
	der, err := x509.CreateCertificate(rand.Reader, template, ca.Cert, publicKey, ca.key)
	if err != nil {
		return nil, fmt.Errorf("issuing client certificate for %s: %w", commonName, err)
	}
	return der, nil
}

// GenerateKey creates an ECDSA P-256 key.
func GenerateKey() (crypto.Signer, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generating key: %w", err)
	}
	return key, nil
}

// CreateCSR builds a PKCS#10 request for key (used by agents at join).
func CreateCSR(key crypto.Signer) ([]byte, error) {
	der, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{Subject: pkix.Name{CommonName: "trainmesh-agent"}}, key)
	if err != nil {
		return nil, fmt.Errorf("creating CSR: %w", err)
	}
	return der, nil
}

// WriteCertAndKey writes PEM files: the certificate world-readable, the key owner-only.
func WriteCertAndKey(certPath, keyPath string, certDER []byte, key crypto.Signer) error {
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return fmt.Errorf("encoding key: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(certPath), 0o700); err != nil {
		return fmt.Errorf("creating %s: %w", filepath.Dir(certPath), err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}), keyFileMode); err != nil {
		return fmt.Errorf("writing %s: %w", keyPath, err)
	}
	return WriteCert(certPath, certDER)
}

// WriteCert writes a PEM certificate.
func WriteCert(path string, der []byte) error {
	if err := os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), certFileMode); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}

// ReadCert parses a PEM certificate file.
func ReadCert(path string) (*x509.Certificate, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- path inside the operator's state dir
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	block, _ := pem.Decode(raw)
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, fmt.Errorf("%s: no PEM certificate", path)
	}
	return x509.ParseCertificate(block.Bytes)
}

// ReadCertAndKey parses a PEM certificate and its PKCS#8 key.
func ReadCertAndKey(certPath, keyPath string) (*x509.Certificate, crypto.Signer, error) {
	cert, err := ReadCert(certPath)
	if err != nil {
		return nil, nil, err
	}
	raw, err := os.ReadFile(keyPath) // #nosec G304 -- path inside the operator's state dir
	if err != nil {
		return nil, nil, fmt.Errorf("reading %s: %w", keyPath, err)
	}
	block, _ := pem.Decode(raw)
	if block == nil {
		return nil, nil, fmt.Errorf("%s: no PEM key", keyPath)
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing %s: %w", keyPath, err)
	}
	signer, ok := parsed.(crypto.Signer)
	if !ok {
		return nil, nil, fmt.Errorf("%s: key cannot sign", keyPath)
	}
	return cert, signer, nil
}

func baseTemplate(now time.Time, validity time.Duration, subject pkix.Name) (*x509.Certificate, error) {
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), serialBits))
	if err != nil {
		return nil, fmt.Errorf("generating serial: %w", err)
	}
	return &x509.Certificate{
		SerialNumber: serial,
		Subject:      subject,
		NotBefore:    now.Add(-clockSkewMargin),
		NotAfter:     now.Add(validity),
	}, nil
}
