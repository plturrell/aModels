package main

import (
	"html/template"
	"net/http"
)

const webUITemplate = `
<!DOCTYPE html>
<html>
<head>
	<title>Extraction Flow Explorer</title>
</head>
<body>
	<h1>Extraction Flow Explorer</h1>

	<h2>Catalog</h2>
	<div id="catalog">
		<h3>Projects</h3>
		<ul id="projects"></ul>

		<h3>Systems</h3>
		<ul id="systems"></ul>

		<h3>Information Systems</h3>
		<ul id="information-systems"></ul>
	</div>

	<h2>Extraction Flow</h2>
	<div id="flow">
		<!-- Placeholder for the flow modeling UI -->
	</div>

	<h2>Results</h2>
	<div id="results">
		<!-- Placeholder for the results view -->
	</div>

	<script>
		// Web UI implementation deferred - use catalog endpoints for now:
		// GET /catalog/projects, /catalog/systems, /catalog/information-systems
		// POST /graph for graph visualization
	</script>
</body>
</html>
`

func (s *extractServer) handleWebUI(w http.ResponseWriter, r *http.Request) {
	t, err := template.New("webui").Parse(webUITemplate)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := t.Execute(w, nil); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
