package api

import (
	"log"
	"net/http"

	"github.com/plturrell/aModels/services/catalog/triplestore"
)

// SPARQLHandler provides HTTP handler for SPARQL endpoint.
type SPARQLHandler struct {
	endpoint *triplestore.SPARQLEndpoint
	logger   *log.Logger
}

// NewSPARQLHandler creates a new SPARQL handler.
func NewSPARQLHandler(endpoint *triplestore.SPARQLEndpoint, logger *log.Logger) *SPARQLHandler {
	return &SPARQLHandler{
		endpoint: endpoint,
		logger:   logger,
	}
}

// HandleSPARQL handles SPARQL query requests.
func (h *SPARQLHandler) HandleSPARQL(w http.ResponseWriter, r *http.Request) {
	h.endpoint.HandleSPARQL(w, r)
}

