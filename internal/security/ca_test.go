package security

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

const testValidity = 24 * time.Hour

func TestLoadOrCreateCAIsStable(t *testing.T) {
	dir := t.TempDir()
	first, err := LoadOrCreateCA(dir, testNow, testValidity)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	second, err := LoadOrCreateCA(dir, testNow, testValidity)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if first.Hash() != second.Hash() {
		t.Fatal("reloading the CA produced a different certificate")
	}
	if !first.Cert.IsCA {
		t.Fatal("CA certificate is not marked as a CA")
	}
}

func TestSignCSRUsesCallerIdentityNotCSRSubject(t *testing.T) {
	ca, _ := NewCA(testNow, testValidity)
	key, _ := GenerateKey()
	csr, _ := CreateCSR(key)

	der, err := ca.SignCSR(csr, "node-1234", RoleNode, testNow, testValidity)

	if err != nil {
		t.Fatalf("SignCSR() error = %v", err)
	}
	cert, _ := x509.ParseCertificate(der)
	if cert.Subject.CommonName != "node-1234" || cert.Subject.OrganizationalUnit[0] != RoleNode {
		t.Fatalf("subject = %+v, want node-1234/node", cert.Subject)
	}
	pool := x509.NewCertPool()
	pool.AddCert(ca.Cert)
	if _, err := cert.Verify(x509.VerifyOptions{Roots: pool, KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}, CurrentTime: testNow}); err != nil {
		t.Fatalf("issued certificate does not verify: %v", err)
	}
}

func TestSignCSRRejectsGarbage(t *testing.T) {
	ca, _ := NewCA(testNow, testValidity)
	if _, err := ca.SignCSR([]byte("nope"), "n", RoleNode, testNow, testValidity); !errors.Is(err, ErrCSRInvalid) {
		t.Fatalf("SignCSR(garbage) = %v, want ErrCSRInvalid", err)
	}
}

func TestCertAndKeyFilesRoundTrip(t *testing.T) {
	ca, _ := NewCA(testNow, testValidity)
	der, key, err := ca.IssueClient("admin", RoleAdmin, testNow, testValidity)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	certPath, keyPath := filepath.Join(dir, AdminFile), filepath.Join(dir, AdminKeyFile)
	if err := WriteCertAndKey(certPath, keyPath, der, key); err != nil {
		t.Fatal(err)
	}

	cert, loadedKey, err := ReadCertAndKey(certPath, keyPath)

	if err != nil || cert.Subject.CommonName != "admin" || loadedKey == nil {
		t.Fatalf("ReadCertAndKey() = %v, %v", cert.Subject, err)
	}
}

// handshake runs a real TLS handshake over an in-memory pipe and returns both sides' errors.
func handshake(t *testing.T, server, client *tls.Config) (serverErr, clientErr error) {
	t.Helper()
	serverConn, clientConn := net.Pipe()
	done := make(chan error, 1)
	go func() {
		conn := tls.Server(serverConn, server)
		done <- conn.Handshake()
		_ = conn.Close()
	}()
	conn := tls.Client(clientConn, client)
	clientErr = conn.Handshake()
	_ = conn.Close()
	return <-done, clientErr
}

func TestPinnedAndCATLS(t *testing.T) {
	ca, _ := NewCA(time.Now(), testValidity)
	serverDER, serverKey, _ := ca.IssueServer([]string{"127.0.0.1", "localhost"}, time.Now(), testValidity)
	server := ServerTLS(serverDER, serverKey, ca.Cert)

	if _, err := handshake(t, server, PinnedTLS(ca.Hash())); err != nil {
		t.Errorf("pinned handshake with the right hash failed: %v", err)
	}
	if _, err := handshake(t, server, PinnedTLS(CertificateHash([]byte("other")))); err == nil {
		t.Error("pinned handshake with a wrong hash succeeded")
	}
	if _, err := handshake(t, server, ClientTLS(ca.Cert, nil, nil)); err != nil {
		t.Errorf("CA-verified handshake failed: %v", err)
	}
	otherCA, _ := NewCA(time.Now(), testValidity)
	if _, err := handshake(t, server, ClientTLS(otherCA.Cert, nil, nil)); err == nil {
		t.Error("handshake trusting another CA succeeded")
	}
}

func TestNodeCertificateCannotImpersonateServer(t *testing.T) {
	ca, _ := NewCA(time.Now(), testValidity)
	nodeDER, nodeKey, _ := ca.IssueClient("node-1", RoleNode, time.Now(), testValidity)
	impostor := ServerTLS(nodeDER, nodeKey, ca.Cert)

	if _, err := handshake(t, impostor, ClientTLS(ca.Cert, nil, nil)); err == nil {
		t.Fatal("a node (client-auth) certificate was accepted as a server")
	}
}

func TestNodeCertificateNamesTheNodeForPeerTLS(t *testing.T) {
	ca, _ := NewCA(time.Now(), testValidity)
	nodeDER, _, err := ca.IssueClient("node-1", RoleNode, time.Now(), testValidity)
	if err != nil {
		t.Fatal(err)
	}
	node, _ := x509.ParseCertificate(nodeDER)
	if !slices.Equal(node.DNSNames, []string{"node-1"}) {
		t.Errorf("node DNS names = %v, want [node-1] (tmcomm dials leaders by node ID)", node.DNSNames)
	}
	if !slices.Contains(node.ExtKeyUsage, x509.ExtKeyUsageServerAuth) || !slices.Contains(node.ExtKeyUsage, x509.ExtKeyUsageClientAuth) {
		t.Errorf("node key usages = %v, want server and client auth", node.ExtKeyUsage)
	}

	adminDER, _, _ := ca.IssueClient("admin", RoleAdmin, time.Now(), testValidity)
	admin, _ := x509.ParseCertificate(adminDER)
	if len(admin.DNSNames) != 0 || slices.Contains(admin.ExtKeyUsage, x509.ExtKeyUsageServerAuth) {
		t.Errorf("admin certificates must stay client-only: %v %v", admin.DNSNames, admin.ExtKeyUsage)
	}
}
