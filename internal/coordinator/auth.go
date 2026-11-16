package coordinator

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/security"
)

// roleAnonymous marks the one call allowed without a client certificate.
const roleAnonymous = ""

// methodRoles is the authorization table: every RPC must appear here or it is denied.
var methodRoles = map[string]string{
	trainmeshv1.CoordinatorService_Join_FullMethodName:             roleAnonymous,
	trainmeshv1.CoordinatorService_Heartbeat_FullMethodName:        security.RoleNode,
	trainmeshv1.CoordinatorService_ReportAssignment_FullMethodName: security.RoleNode,
	trainmeshv1.CoordinatorService_CreateToken_FullMethodName:      security.RoleAdmin,
	trainmeshv1.CoordinatorService_ListNodes_FullMethodName:        security.RoleAdmin,
	trainmeshv1.CoordinatorService_SubmitJob_FullMethodName:        security.RoleAdmin,
	trainmeshv1.CoordinatorService_GetJob_FullMethodName:           security.RoleAdmin,
	trainmeshv1.CoordinatorService_CancelJob_FullMethodName:        security.RoleAdmin,
}

type identityKey struct{}

// authorize is a unary interceptor enforcing methodRoles.
func authorize(ctx context.Context, request any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	required, known := methodRoles[info.FullMethod]
	if !known {
		return nil, status.Errorf(codes.PermissionDenied, "method %s is not authorized", info.FullMethod)
	}
	if required == roleAnonymous {
		return handler(ctx, request)
	}
	identity, err := security.IdentityFromContext(ctx)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, err.Error())
	}
	if identity.Role != required {
		return nil, status.Errorf(codes.PermissionDenied, "%s needs role %q, caller %q has %q", info.FullMethod, required, identity.Name, identity.Role)
	}
	return handler(context.WithValue(ctx, identityKey{}, identity), request)
}

func callerID(ctx context.Context) string {
	identity, _ := ctx.Value(identityKey{}).(security.Identity)
	return identity.Name
}
