package security

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/url"
)

const joinScheme = "trainmesh"

// ErrJoinStringInvalid wraps every problem with a join string.
var ErrJoinStringInvalid = errors.New("invalid join string")

// JoinString is what an operator copies to a new machine: where the coordinator is, the token,
// and the SHA-256 of the cluster CA certificate. The CA hash lets the agent trust the coordinator
// on first contact without any pre-shared file.
type JoinString struct {
	Address string // host:port of the coordinator
	Token   string
	CAHash  string // hex SHA-256 of the CA certificate DER
}

// String renders trainmesh://host:port?token=…&ca=….
func (j JoinString) String() string {
	query := url.Values{"token": {j.Token}, "ca": {j.CAHash}}
	return (&url.URL{Scheme: joinScheme, Host: j.Address, RawQuery: query.Encode()}).String()
}

// ParseJoinString validates and splits a join string.
func ParseJoinString(raw string) (JoinString, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != joinScheme {
		return JoinString{}, fmt.Errorf("%w: expected %s://host:port?token=…&ca=…", ErrJoinStringInvalid, joinScheme)
	}
	if _, _, err := net.SplitHostPort(parsed.Host); err != nil {
		return JoinString{}, fmt.Errorf("%w: address %q: %w", ErrJoinStringInvalid, parsed.Host, err)
	}
	j := JoinString{Address: parsed.Host, Token: parsed.Query().Get("token"), CAHash: parsed.Query().Get("ca")}
	if err := ValidateTokenFormat(j.Token); err != nil {
		return JoinString{}, fmt.Errorf("%w: %w", ErrJoinStringInvalid, err)
	}
	if decoded, err := hex.DecodeString(j.CAHash); err != nil || len(decoded) != sha256.Size {
		return JoinString{}, fmt.Errorf("%w: ca must be a hex SHA-256", ErrJoinStringInvalid)
	}
	return j, nil
}

// CertificateHash is the hex SHA-256 of a DER certificate, as used in join strings.
func CertificateHash(der []byte) string {
	sum := sha256.Sum256(der)
	return hex.EncodeToString(sum[:])
}
