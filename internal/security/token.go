// Package security holds join tokens, join strings and the cluster certificate authority.
package security

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	tokenPrefix      = "tm1_"
	tokenSecretBytes = 32
)

// Join token errors. All are non-retryable for the caller.
var (
	ErrTokenUnknown   = errors.New("join token is not known to this cluster")
	ErrTokenExpired   = errors.New("join token has expired")
	ErrTokenExhausted = errors.New("join token has already been used")
	ErrTokenRevoked   = errors.New("join token was revoked")
	ErrTokenMalformed = errors.New("join token is malformed")
)

// TokenRecord is what the coordinator stores for a token: never the secret, only its hash.
type TokenRecord struct {
	Hash      string    `json:"hash"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
	UsesLeft  int       `json:"uses_left"`
	Revoked   bool      `json:"revoked"`
}

// NewToken returns a fresh secret (shown once to the operator) and the record to persist.
func NewToken(now time.Time, ttl time.Duration, uses int) (string, TokenRecord, error) {
	if ttl <= 0 || uses <= 0 {
		return "", TokenRecord{}, fmt.Errorf("token ttl (%s) and uses (%d) must be positive", ttl, uses)
	}
	raw := make([]byte, tokenSecretBytes)
	if _, err := rand.Read(raw); err != nil {
		return "", TokenRecord{}, fmt.Errorf("generating token: %w", err)
	}
	secret := tokenPrefix + base64.RawURLEncoding.EncodeToString(raw)
	record := TokenRecord{Hash: HashToken(secret), CreatedAt: now.UTC(), ExpiresAt: now.Add(ttl).UTC(), UsesLeft: uses}
	return secret, record, nil
}

// HashToken is the lookup key for a token secret.
func HashToken(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(sum[:])
}

// ValidateTokenFormat rejects obviously wrong input before any lookup.
func ValidateTokenFormat(secret string) error {
	encoded, ok := strings.CutPrefix(secret, tokenPrefix)
	if !ok {
		return fmt.Errorf("%w: missing %q prefix", ErrTokenMalformed, tokenPrefix)
	}
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil || len(raw) != tokenSecretBytes {
		return fmt.Errorf("%w: wrong encoding or length", ErrTokenMalformed)
	}
	return nil
}

// CheckRedeemable says whether a stored token may be used now. The store calls it inside the same
// transaction that decrements UsesLeft, so a single-use token cannot be redeemed twice.
func CheckRedeemable(record TokenRecord, now time.Time) error {
	switch {
	case record.Revoked:
		return ErrTokenRevoked
	case !now.Before(record.ExpiresAt):
		return ErrTokenExpired
	case record.UsesLeft <= 0:
		return ErrTokenExhausted
	default:
		return nil
	}
}
