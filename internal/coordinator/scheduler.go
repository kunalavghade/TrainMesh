package coordinator

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"sort"
	"strings"

	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/store"
)

// ErrNotEnoughNodes means fewer assignable nodes exist than the job needs.
var ErrNotEnoughNodes = errors.New("not enough READY nodes")

// ErrVersionMismatch means the chosen nodes run different PyTorch or runtime versions.
var ErrVersionMismatch = errors.New("nodes run different PyTorch/runtime versions")

// pickNodes chooses `machines` READY nodes, ordered by ID so rank assignment is deterministic.
// Rank 0 (the rendezvous master) is the first. All chosen nodes must run the same versions,
// otherwise DDP may disagree numerically or fail to rendezvous.
func pickNodes(nodes []store.Node, machines int) ([]store.Node, error) {
	ready := slices.DeleteFunc(slices.Clone(nodes), func(n store.Node) bool {
		return !membership.Assignable(n.State) || n.Address == ""
	})
	if len(ready) < machines {
		return nil, fmt.Errorf("%w: job needs %d, %d available", ErrNotEnoughNodes, machines, len(ready))
	}
	sort.Slice(ready, func(i, j int) bool { return ready[i].ID < ready[j].ID })
	chosen := ready[:machines]

	versions := map[string]bool{}
	for _, node := range chosen {
		versions[node.Capabilities.TorchVersion+"/"+node.Capabilities.RuntimeVersion] = true
	}
	if len(versions) > 1 {
		return nil, fmt.Errorf("%w: %s", ErrVersionMismatch, strings.Join(slices.Sorted(maps.Keys(versions)), ", "))
	}
	return chosen, nil
}
