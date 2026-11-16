package security

import (
	"errors"
	"strings"
	"testing"
	"time"
)

var testNow = time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)

func TestNewTokenStoresOnlyTheHash(t *testing.T) {
	secret, record, err := NewToken(testNow, 15*time.Minute, 1)

	if err != nil {
		t.Fatalf("NewToken() error = %v", err)
	}
	if err := ValidateTokenFormat(secret); err != nil {
		t.Fatalf("generated token is malformed: %v", err)
	}
	if record.Hash != HashToken(secret) || strings.Contains(record.Hash, secret) {
		t.Fatalf("record must hold only the hash of the secret")
	}
	if !record.ExpiresAt.Equal(testNow.Add(15 * time.Minute)) {
		t.Errorf("ExpiresAt = %s", record.ExpiresAt)
	}
	other, _, _ := NewToken(testNow, time.Minute, 1)
	if other == secret {
		t.Error("two tokens are identical")
	}
}

func TestNewTokenRejectsNonPositiveInputs(t *testing.T) {
	if _, _, err := NewToken(testNow, 0, 1); err == nil {
		t.Error("zero ttl accepted")
	}
	if _, _, err := NewToken(testNow, time.Minute, 0); err == nil {
		t.Error("zero uses accepted")
	}
}

func TestCheckRedeemable(t *testing.T) {
	base := TokenRecord{ExpiresAt: testNow.Add(time.Minute), UsesLeft: 1}
	tests := []struct {
		name   string
		mutate func(*TokenRecord)
		want   error
	}{
		{"valid", func(*TokenRecord) {}, nil},
		{"expired", func(r *TokenRecord) { r.ExpiresAt = testNow }, ErrTokenExpired},
		{"used up", func(r *TokenRecord) { r.UsesLeft = 0 }, ErrTokenExhausted},
		{"revoked", func(r *TokenRecord) { r.Revoked = true }, ErrTokenRevoked},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record := base
			tt.mutate(&record)
			err := CheckRedeemable(record, testNow)
			if (tt.want == nil && err != nil) || (tt.want != nil && !errors.Is(err, tt.want)) {
				t.Fatalf("CheckRedeemable() = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestValidateTokenFormat(t *testing.T) {
	for _, bad := range []string{"", "tm1_", "abc", "tm1_!!!", "tm1_" + strings.Repeat("A", 10)} {
		if !errors.Is(ValidateTokenFormat(bad), ErrTokenMalformed) {
			t.Errorf("ValidateTokenFormat(%q) accepted", bad)
		}
	}
}

func TestJoinStringRoundTrip(t *testing.T) {
	secret, _, _ := NewToken(testNow, time.Minute, 1)
	original := JoinString{Address: "192.168.1.10:7443", Token: secret, CAHash: CertificateHash([]byte("ca"))}

	parsed, err := ParseJoinString(original.String())

	if err != nil || parsed != original {
		t.Fatalf("ParseJoinString(String()) = %+v, %v; want %+v", parsed, err, original)
	}
}

func TestParseJoinStringRejects(t *testing.T) {
	secret, _, _ := NewToken(testNow, time.Minute, 1)
	hash := CertificateHash([]byte("ca"))
	for _, raw := range []string{
		"http://host:1?token=" + secret + "&ca=" + hash,
		"trainmesh://host?token=" + secret + "&ca=" + hash,
		"trainmesh://host:1?token=bad&ca=" + hash,
		"trainmesh://host:1?token=" + secret + "&ca=zz",
	} {
		if _, err := ParseJoinString(raw); !errors.Is(err, ErrJoinStringInvalid) {
			t.Errorf("ParseJoinString(%q) = %v, want ErrJoinStringInvalid", raw, err)
		}
	}
}
