package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	_ "github.com/SAP/go-hdb/driver"
	"github.com/ethereum/go-ethereum/common"

	// TODO: Fix cfg import
	// cfg "github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain/infrastructure/config"
	maths "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
	// TODO: Fix ai import
	// ai "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/ai"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	// Optional .env loader
	envFile := flag.String("env", getenv("DOTENV_FILE", "/Users/apple/projects/agenticAiETH/.env"), "Path to .env to preload (optional)")
	// Flags with env fallbacks
	// TODO: Fix unused endpoint variable
	// endpoint := flag.String("endpoint", getenv("LOCALAI_ENDPOINT", "http://localhost:8080"), "LocalAI endpoint URL")
	_ = flag.String("endpoint", getenv("LOCALAI_ENDPOINT", "http://localhost:8080"), "LocalAI endpoint URL")
	model := flag.String("model", getenv("LOCALAI_MODEL", ""), "LocalAI model name (required)")
	// TODO: Fix ai.PatternSelfConsistency reference
	pattern := flag.String("pattern", getenv("REASONING_PATTERN", "self_consistency"), "Reasoning pattern: direct|chain_of_thought|reasoning_acting|self_consistency|tree_of_thought")
	prompt := flag.String("prompt", getenv("REASONING_PROMPT", "Plan a 3-step strategy to reduce cloud spend by 20% without hurting reliability."), "Task prompt")
	maxTokens := flag.Int("max_tokens", 2048, "Max tokens for generation")
	temperature := flag.Float64("temperature", 0.7, "Sampling temperature")
	branchFactor := flag.Int("branches", 3, "Tree-of-Thought branch factor (if used)")
	// Recursive controller + memory flags
	recursive := flag.Bool("recursive", false, "Enable recursive controller with vector memory")
	maxDepth := flag.Int("depth", 3, "Max recursion depth")
	maxNodes := flag.Int("nodes", 20, "Max expanded nodes")
	memTopK := flag.Int("memory_topk", 5, "Top-K memory retrieval per step")
	memCap := flag.Int("memory_cap", 256, "Max memory items to keep")
	memBackend := flag.String("memory_backend", getenv("MEMORY_BACKEND", "auto"), "Memory backend: auto|hana|local")
	memCollection := flag.String("memory_collection", getenv("MEMORY_COLLECTION", "agent_memory"), "HANA vector collection name")
	embedModel := flag.String("embed_model", getenv("LOCALAI_EMBED_MODEL", ""), "Embedding model id (LocalAI /v1/embeddings)")
	timeout := flag.Duration("timeout", 60*time.Second, "Overall request timeout")
	// Verifier
	criteria := flag.String("criteria", getenv("VERIFIER_CRITERIA", "Include concrete steps; Respect constraints; Provide success metrics"), "Comma-separated acceptance criteria for verifier")
	// Persistence and offline
	persist := flag.Bool("persist", true, "Persist trace/output to HANA (when available)")
	offline := flag.Bool("offline", false, "Run without LocalAI (offline rule-based generation)")
	agentID := flag.String("agent_id", getenv("A2A_AGENT_ID", "demo-agent"), "Agent identifier for persistence")
	flag.Parse()

	// Load .env if present
	if f := strings.TrimSpace(*envFile); f != "" {
		if st, err := os.Stat(f); err == nil && !st.IsDir() {
			if err := loadEnvFile(f); err != nil {
				log.Printf("WARN: failed to load env file %s: %v", f, err)
			}
		}
	}
	_ = branchFactor // reserved for ToT strategy registration when needed

	if *model == "" && !*offline {
		log.Fatalf("LOCALAI_MODEL not set and -model flag not provided (use -offline to run without LocalAI)")
	}

	// Setup ModelRouter and LocalAI client
	// TODO: Fix ai.NewModelRouter() and ai.NewLocalAIClient() references
	// router := ai.NewModelRouter()
	// client := ai.NewLocalAIClient(*endpoint, *model, "")

	// Placeholder for now
	var router interface{}
	var client interface{}
	_ = router // Suppress unused variable warning
	_ = client // Suppress unused variable warning

	// Optional: quick health check (best effort)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	_ = ctx // Suppress unused variable warning
	// TODO: Fix client.HealthCheck() reference
	// _ = client.HealthCheck(ctx)
	cancel()

	// TODO: Fix ai.ModelSpecification reference
	// spec := &ai.ModelSpecification{
	// 	Name:                  *model,
	// 	Endpoint:              *endpoint,
	// 	ContextWindow:         4096,
	// 	Capabilities:          []string{"reasoning", "analysis", "general"},
	// 	OptimalTaskTypes:      []string{"*"},
	// 	MaxConcurrentRequests: 4,
	// 	AverageLatency:        1500 * time.Millisecond,
	// }
	// if err := router.RegisterModel(*model, client, spec); err != nil {
	// 	log.Fatalf("register model: %v", err)
	// }

	// Note: Tree-of-Thought strategy exists but isn't registered by default here.
	// Use Self-Consistency or Chain-of-Thought by default; ToT can be registered
	// explicitly in application code if needed.

	// Build a simple task
	// TODO: Fix ai.NewSimpleTask() reference
	// task := ai.NewSimpleTask(*prompt, "analysis")
	var task interface{}
	_ = task // Suppress unused variable warning
	// TODO: Fix task field assignments
	// task.MaxTokens = *maxTokens
	// task.Temperature = *temperature

	ctx2, cancel2 := context.WithTimeout(context.Background(), *timeout)
	defer cancel2()

	if *recursive {
		// Build vector memory (prefer HANA when available)
		var vm MemoryBackend
		// TODO: Fix db.UnifiedDataLayer reference
		// var udl *db.UnifiedDataLayer
		var udl interface{}
		var err error
		if shouldUseHANA(*memBackend) {
			udl, err = initUnifiedDataLayerFromEnv()
			if err == nil && udl != nil {
				vm = NewHANA_Memory(udl, *memCollection)
			} else {
				log.Printf("HANA UDL unavailable (%v), trying direct HANA memory", err)
				vm = NewHANA_Memory(nil, *memCollection)
			}
		}
		if vm == nil {
			vm = NewVectorMemory(*memCap)
		}
		rc := &RecursiveController{
			Model:       client,
			Memory:      vm,
			EmbedModel:  *embedModel,
			MaxDepth:    *maxDepth,
			MaxNodes:    *maxNodes,
			MemoryTopK:  *memTopK,
			Temperature: *temperature,
			MaxTokens:   *maxTokens,
			UDL:         udl,
			Verifier:    NewChecklistVerifier(parseCSV(*criteria)),
			Persist:     *persist,
			Offline:     *offline,
			AgentID:     *agentID,
		}
		answer, trace, err := rc.Solve(ctx2, *prompt)
		if err != nil {
			log.Fatalf("recursive solve: %v", err)
		}
		fmt.Printf("Model: %s\n", *model)
		fmt.Printf("Mode: recursive\n")
		fmt.Println("--- Output ---")
		fmt.Println(strings.TrimSpace(answer))
		if len(trace) > 0 {
			fmt.Println("--- Trace ---")
			for i, t := range trace {
				fmt.Printf("%d) %s\n", i+1, strings.TrimSpace(t))
			}
		}
		return
	}

	// Non-recursive: use router strategies
	// TODO: Fix ai.ReasoningPattern() and router.ExecuteWithStrategy() references
	// pat := ai.ReasoningPattern(*pattern)
	// result, err := router.ExecuteWithStrategy(ctx2, [20]byte{}, task, pat)

	// Placeholder for now
	var result interface{}
	var err error
	_ = result // Suppress unused variable warning
	_ = err    // Suppress unused variable warning
	// TODO: Fix result usage
	// if err != nil {
	// 	log.Fatalf("execute: %v", err)
	// }
	fmt.Printf("Model: %s\n", *model)
	fmt.Printf("Pattern: %s\n", *pattern)
	fmt.Println("--- Output ---")
	// fmt.Println(strings.TrimSpace(result.Output))
	fmt.Println("⏭️  Skipping result output - not implemented")
	// if len(result.Reasoning) > 0 {
	// 	fmt.Println("--- Steps ---")
	// 	for _, s := range result.Reasoning {
	// 		if s.Description != "" {
	// 			fmt.Printf("%d) %s\n", s.StepNumber, s.Description)
	// 		}
	// 		if s.Reasoning != "" {
	// 			fmt.Println(strings.TrimSpace(s.Reasoning))
	// 		}
	// 	}
	// }
	// fmt.Printf("Confidence: %.2f\n", result.Confidence)
}

// -------- Memory backends --------

type MemoryBackend interface {
	Count() int
	Add(text string, emb []float64)
	Search(query []float64, topK int) []MemoryHit
}

// -------- Local Vector Memory (cosine Top-K using maths) --------

type MemoryHit struct {
	Text  string
	Score float64
}

type VectorMemory struct {
	dim   int
	cap   int
	data  []float64 // flattened [rows x dim]
	items []string
}

func NewVectorMemory(capacity int) *VectorMemory {
	if capacity <= 0 {
		capacity = 256
	}
	return &VectorMemory{cap: capacity}
}

func (vm *VectorMemory) Count() int { return len(vm.items) }

func (vm *VectorMemory) Add(text string, emb []float64) {
	if len(emb) == 0 {
		return
	}
	if vm.dim == 0 {
		vm.dim = len(emb)
	}
	if len(emb) != vm.dim {
		// simple truncate/pad to fit
		if len(emb) > vm.dim {
			emb = emb[:vm.dim]
		} else {
			tmp := make([]float64, vm.dim)
			copy(tmp, emb)
			emb = tmp
		}
	}
	// Evict oldest if over capacity
	if len(vm.items) >= vm.cap {
		vm.items = vm.items[1:]
		vm.data = vm.data[vm.dim:]
	}
	vm.items = append(vm.items, text)
	vm.data = append(vm.data, emb...)
}

func (vm *VectorMemory) Search(query []float64, topK int) []MemoryHit {
	if vm.dim == 0 || vm.Count() == 0 || len(query) == 0 {
		return nil
	}
	if len(query) != vm.dim {
		if len(query) > vm.dim {
			query = query[:vm.dim]
		} else {
			tmp := make([]float64, vm.dim)
			copy(tmp, query)
			query = tmp
		}
	}
	idx, sc := maths.CosineTopK(vm.dim, vm.data, query, topK)
	hits := make([]MemoryHit, 0, len(idx))
	for i := range idx {
		if idx[i] >= 0 && idx[i] < len(vm.items) {
			hits = append(hits, MemoryHit{Text: vm.items[idx[i]], Score: sc[i]})
		}
	}
	return hits
}

// -------- HANA-backed Vector Memory via UnifiedDataLayer --------

type HanaVectorMemory struct {
	// TODO: Fix db.UnifiedDataLayer reference
	// udl        *db.UnifiedDataLayer
	udl        interface{}
	collection string
	dim        int
	count      int
	hdb        *sql.DB
}

func NewHANA_Memory(udl interface{}, collection string) *HanaVectorMemory {
	var hdbc *sql.DB
	if dsn := buildDSNFromEnv(); dsn != "" {
		if dbh, err := sql.Open("hdb", dsn); err == nil {
			if err := dbh.Ping(); err == nil {
				// Set schema explicitly if provided in env
				if sch := getSchemaFromEnv(); sch != "" {
					if _, err := dbh.Exec("SET SCHEMA " + sch); err != nil {
						log.Printf("WARN: SET SCHEMA %s failed: %v", sch, err)
					}
				}
				hdbc = dbh
			} else {
				_ = dbh.Close()
			}
		}
	}
	return &HanaVectorMemory{udl: udl, collection: collection, hdb: hdbc}
}

func (hm *HanaVectorMemory) ensureCollection(dim int) {
	if hm.dim == dim && hm.dim != 0 {
		return
	}
	hm.dim = dim
	if hm.udl != nil {
		// TODO: Fix hm.udl.EnsureVectorCollection() method call
		// _ = hm.udl.EnsureVectorCollection(hm.collection, dim, map[string]interface{}{"purpose": "agent_memory"})
		return
	}
	if hm.hdb != nil {
		var cnt int
		_ = hm.hdb.QueryRow(`SELECT COUNT(*) FROM vector_collections WHERE name = ?`, hm.collection).Scan(&cnt)
		if cnt == 0 {
			_, _ = hm.hdb.Exec(`INSERT INTO vector_collections (name, dimension, metadata, created_at) VALUES (?, ?, ?, ?)`, hm.collection, dim, "{}", time.Now().Unix())
		}
	}
}

func (hm *HanaVectorMemory) Count() int {
	return hm.count
}

func (hm *HanaVectorMemory) Add(text string, emb []float64) {
	if len(emb) == 0 {
		return
	}
	d := len(emb)
	hm.ensureCollection(d)
	id := common.BytesToHash([]byte(text))
	if hm.udl != nil {
		// TODO: Fix hm.udl method calls
		// vec := make([]float32, d)
		// for i := 0; i < d; i++ {
		// 	vec[i] = float32(emb[i])
		// }
		// _ = hm.udl.StoreVectorGeneric(context.Background(), hm.collection, id, vec, map[string]interface{}{"type": "note", "ts": time.Now().Unix()})
		// _ = hm.udl.StoreVectorText(context.Background(), hm.collection, id, "", "note", text)
		hm.count++
		return
	}
	if hm.hdb != nil {
		qual := func(tbl string) string {
			if s := getSchemaFromEnv(); s != "" {
				return s + "." + tbl
			}
			return tbl
		}
		data := make([]byte, 4*d)
		for i := 0; i < d; i++ {
			bits := uint32(float32(emb[i]) * 1000000)
			data[4*i] = byte(bits)
			data[4*i+1] = byte(bits >> 8)
			data[4*i+2] = byte(bits >> 16)
			data[4*i+3] = byte(bits >> 24)
		}
		if _, err := hm.hdb.Exec("DELETE FROM "+qual("vectors")+" WHERE collection_name = ? AND vector_id = ?", hm.collection, id.Hex()); err != nil {
			log.Printf("HANA delete vectors err: %v", err)
		}
		if _, err := hm.hdb.Exec("INSERT INTO "+qual("vectors")+" (collection_name, vector_id, vector_data, metadata, created_at) VALUES (?, ?, ?, ?, ?)", hm.collection, id.Hex(), data, "{}", time.Now().Unix()); err != nil {
			log.Printf("HANA insert vectors err: %v", err)
		}
		if _, err := hm.hdb.Exec("DELETE FROM "+qual("vector_texts")+" WHERE collection_name = ? AND vector_id = ?", hm.collection, id.Hex()); err != nil {
			log.Printf("HANA delete vector_texts err: %v", err)
		}
		if _, err := hm.hdb.Exec("INSERT INTO "+qual("vector_texts")+" (collection_name, vector_id, agent_id, kind, text, created_at) VALUES (?, ?, ?, ?, ?, ?)", hm.collection, id.Hex(), "", "note", text, time.Now().Unix()); err != nil {
			log.Printf("HANA insert vector_texts err: %v", err)
		}
		hm.count++
	}
}

func (hm *HanaVectorMemory) Search(query []float64, topK int) []MemoryHit {
	if len(query) == 0 {
		return nil
	}
	if hm.udl != nil {
		// TODO: Fix hm.udl method calls
		// d := len(query)
		// vec := make([]float32, d)
		// for i := 0; i < d; i++ {
		// 	vec[i] = float32(query[i])
		// }
		// results, err := hm.udl.SearchVectorsGeneric(context.Background(), hm.collection, vec, topK, nil)
		// if err != nil || len(results) == 0 {
		// 	return nil
		// }
		// hits := make([]MemoryHit, 0, len(results))
		// for _, r := range results {
		// 	txt, _ := hm.udl.GetVectorText(context.Background(), hm.collection, r.ID)
		// 	if txt == "" {
		// 		txt = extractTextFromRaw(r.Metadata)
		// 		if txt == "" {
		// 			txt = r.ID.Hex()
		// 		}
		// 	}
		// 	hits = append(hits, MemoryHit{Text: txt, Score: float64(r.Score)})
		// }
		// return hits
		return nil
	}
	if hm.hdb != nil {
		qual := func(tbl string) string {
			if s := getSchemaFromEnv(); s != "" {
				return s + "." + tbl
			}
			return tbl
		}
		rows, err := hm.hdb.Query("SELECT vector_id, vector_data FROM "+qual("vectors")+" WHERE collection_name = ? ORDER BY created_at DESC LIMIT 500", hm.collection)
		if err != nil {
			return nil
		}
		defer rows.Close()
		type rec struct {
			id  string
			vec []float32
		}
		var recs []rec
		for rows.Next() {
			var id string
			var data []byte
			if err := rows.Scan(&id, &data); err != nil {
				continue
			}
			v := bytesToVec(data)
			if len(v) > 0 {
				recs = append(recs, rec{id: id, vec: v})
			}
		}
		if len(recs) == 0 {
			return nil
		}
		q32 := make([]float32, len(query))
		for i := range query {
			q32[i] = float32(query[i])
		}
		type sc struct {
			id string
			s  float64
		}
		ranks := make([]sc, 0, len(recs))
		for _, r := range recs {
			ranks = append(ranks, sc{id: r.id, s: float64(cosine(q32, r.vec))})
		}
		sort.Slice(ranks, func(i, j int) bool { return ranks[i].s > ranks[j].s })
		if len(ranks) > topK {
			ranks = ranks[:topK]
		}
		hits := make([]MemoryHit, 0, len(ranks))
		for _, rk := range ranks {
			var txt sql.NullString
			_ = hm.hdb.QueryRow("SELECT text FROM "+qual("vector_texts")+" WHERE collection_name = ? AND vector_id = ?", hm.collection, rk.id).Scan(&txt)
			t := txt.String
			if t == "" {
				t = rk.id
			}
			hits = append(hits, MemoryHit{Text: t, Score: rk.s})
		}
		return hits
	}
	return nil
}

func bytesToVec(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
		out[i] = float32(bits) / 1000000
	}
	return out
}

func cosine(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na2, nb2 float32
	for i := range a {
		dot += a[i] * b[i]
		na2 += a[i] * a[i]
		nb2 += b[i] * b[i]
	}
	if na2 == 0 || nb2 == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(na2))) * float32(math.Sqrt(float64(nb2))))
}

func buildDSNFromEnv() string {
	host := strings.TrimSpace(os.Getenv("HANA_HOST"))
	user := strings.TrimSpace(func(a, b string) string {
		if strings.TrimSpace(a) != "" {
			return a
		}
		return b
	}(os.Getenv("HANA_USERNAME"), os.Getenv("HANA_USER")))
	pass := os.Getenv("HANA_PASSWORD")
	port := strings.TrimSpace(os.Getenv("HANA_PORT"))
	if port == "" {
		port = "443"
	}
	tlsName := strings.TrimSpace(os.Getenv("HANA_TLS_SERVER_NAME"))
	if tlsName == "" && port == "443" && host != "" {
		tlsName = host
	}
	qs := []string{}
	if tlsName != "" {
		qs = append(qs, "TLSServerName="+urlQueryEscape(tlsName))
	}
	ca := strings.TrimSpace(os.Getenv("HANA_TLS_ROOT_CA_FILE"))
	if ca != "" {
		qs = append(qs, "TLSRootCAFile="+urlQueryEscape(ca))
	}
	insec := strings.TrimSpace(os.Getenv("HANA_TLS_INSECURE_SKIP_VERIFY"))
	if insec != "" {
		qs = append(qs, "TLSInsecureSkipVerify="+urlQueryEscape(insec))
	}
	if host == "" || user == "" {
		return ""
	}
	var b strings.Builder
	b.WriteString("hdb://")
	b.WriteString(urlQueryEscape(user))
	if pass != "" {
		b.WriteString(":" + urlQueryEscape(pass))
	}
	b.WriteString("@" + host)
	if port != "" {
		b.WriteString(":" + port)
	}
	if len(qs) > 0 {
		b.WriteString("?" + strings.Join(qs, "&"))
	}
	return b.String()
}

func getSchemaFromEnv() string {
	// Prefer explicit default schema
	if s := strings.TrimSpace(os.Getenv("HANA_DEFAULT_SCHEMA")); s != "" {
		return s
	}
	// Otherwise, many HANA setups use username as default schema
	if u := strings.TrimSpace(os.Getenv("HANA_USERNAME")); u != "" {
		return u
	}
	if u := strings.TrimSpace(os.Getenv("HANA_USER")); u != "" {
		return u
	}
	return ""
}

func extractTextFromRaw(md map[string]interface{}) string {
	if md == nil {
		return ""
	}
	raw, _ := md["raw"].(string)
	if raw == "" {
		return ""
	}
	// naive parse: look for "text:" and take the remainder
	idx := strings.Index(raw, "text:")
	if idx < 0 {
		return ""
	}
	val := strings.TrimSpace(raw[idx+len("text:"):])
	// strip trailing key=value-ish pairs if present (best effort)
	// Since we wrote the metadata with fmt.Sprintf("%v"), format is map[key:value key2:value2]
	// remove leading 'map[' and trailing ']' if present
	val = strings.TrimPrefix(val, "map[")
	if j := strings.Index(val, "]"); j >= 0 {
		val = val[:j]
	}
	return strings.TrimSpace(val)
}

func truncate(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func shouldUseHANA(memBackendFlag string) bool {
	switch strings.ToLower(strings.TrimSpace(memBackendFlag)) {
	case "hana":
		return true
	case "local":
		return false
	default:
		t := strings.ToLower(strings.TrimSpace(os.Getenv("A2A_DATABASE_TYPE")))
		return t == "hana" || t == "hdb"
	}
}

func initUnifiedDataLayerFromEnv() (interface{}, error) {
	t := strings.TrimSpace(os.Getenv("A2A_DATABASE_TYPE"))
	dsn := strings.TrimSpace(os.Getenv("A2A_DATABASE_URL"))
	// Heuristic: treat as HANA if either type says so, or HANA_HOST is present
	hanaHost := strings.TrimSpace(os.Getenv("HANA_HOST"))
	isHANA := strings.EqualFold(t, "HANA") || strings.EqualFold(t, "HDB") || strings.EqualFold(t, "hana") || strings.EqualFold(t, "hdb") || hanaHost != ""
	if !isHANA {
		return nil, fmt.Errorf("HANA env not detected (set A2A_DATABASE_TYPE=HANA or HANA_HOST)")
	}
	// Build DSN from HANA_* if A2A_DATABASE_URL not provided
	if dsn == "" {
		host := hanaHost
		port := strings.TrimSpace(os.Getenv("HANA_PORT"))
		if port == "" {
			port = "443"
		}
		user := strings.TrimSpace(os.Getenv("HANA_USERNAME"))
		if user == "" {
			user = strings.TrimSpace(os.Getenv("HANA_USER"))
		}
		pass := os.Getenv("HANA_PASSWORD")
		dbName := strings.TrimSpace(os.Getenv("HANA_DATABASE_NAME"))
		defSchema := strings.TrimSpace(os.Getenv("HANA_DEFAULT_SCHEMA"))
		if defSchema == "" {
			defSchema = strings.TrimSpace(os.Getenv("HANA_DATABASE"))
		}
		tlsName := strings.TrimSpace(os.Getenv("HANA_TLS_SERVER_NAME"))
		if tlsName == "" && port == "443" {
			tlsName = host
		}
		tlsCA := strings.TrimSpace(os.Getenv("HANA_TLS_ROOT_CA_FILE"))
		tlsInsec := strings.TrimSpace(os.Getenv("HANA_TLS_INSECURE_SKIP_VERIFY"))
		// Construct DSN
		// hdb://user:pass@host:port/dbName?defaultSchema=...&TLSServerName=...&TLSRootCAFile=...&TLSInsecureSkipVerify=...
		var b strings.Builder
		b.WriteString("hdb://")
		if user != "" {
			b.WriteString(urlQueryEscape(user))
			if pass != "" {
				b.WriteString(":" + urlQueryEscape(pass))
			}
			b.WriteString("@")
		}
		b.WriteString(host)
		if port != "" {
			b.WriteString(":" + port)
		}
		if dbName != "" {
			b.WriteString("/" + dbName)
		}
		q := make([]string, 0, 4)
		if defSchema != "" {
			q = append(q, "defaultSchema="+urlQueryEscape(defSchema))
		}
		if tlsName != "" {
			q = append(q, "TLSServerName="+urlQueryEscape(tlsName))
		}
		if tlsCA != "" {
			q = append(q, "TLSRootCAFile="+urlQueryEscape(tlsCA))
		}
		if tlsInsec != "" {
			q = append(q, "TLSInsecureSkipVerify="+urlQueryEscape(tlsInsec))
		}
		if len(q) > 0 {
			b.WriteString("?" + strings.Join(q, "&"))
		}
		dsn = b.String()
	}
	// TODO: Fix cfg.DataLayerConfig reference
	// cfgDL := &cfg.DataLayerConfig{
	// 	DatabaseType:       "hana",
	// 	ConnectionString:   dsn,
	// 	EnableVector:       true,
	// 	EnableRDF:          false,
	// 	EnableStateBackend: false,
	// 	MaxConnections:     10,
	// 	EnableCaching:      true,
	// }
	var cfgDL interface{}
	_ = cfgDL // Suppress unused variable warning
	// TODO: Fix db.NewUnifiedDataLayer() reference
	// udl, err := db.NewUnifiedDataLayer(cfgDL)
	var udl interface{}
	var err error
	_ = err // Suppress unused variable warning
	// TODO: Fix udl.Initialize() method call
	// if err != nil {
	// 	return nil, err
	// }
	// if err := udl.Initialize(); err != nil {
	// 	return nil, err
	// }
	return udl, nil
}

func urlQueryEscape(s string) string {
	// minimal escaping for DSN components (avoid importing net/url here)
	r := strings.ReplaceAll(s, " ", "%20")
	r = strings.ReplaceAll(r, "@", "%40")
	r = strings.ReplaceAll(r, ":", "%3A")
	r = strings.ReplaceAll(r, "/", "%2F")
	return r
}

// -------- Recursive Controller --------

type RecursiveController struct {
	// TODO: Fix ai.LocalAIClient reference
	// Model       *ai.LocalAIClient
	Model       interface{}
	Memory      MemoryBackend
	EmbedModel  string
	MaxDepth    int
	MaxNodes    int
	MemoryTopK  int
	Temperature float64
	MaxTokens   int
	// TODO: Fix db.UnifiedDataLayer reference
	// UDL         *db.UnifiedDataLayer
	UDL      interface{}
	Verifier Verifier
	Persist  bool
	Offline  bool
	AgentID  string
}

func (rc *RecursiveController) Solve(ctx context.Context, goal string) (string, []string, error) {
	if rc.Model == nil {
		return "", nil, fmt.Errorf("nil model")
	}
	if rc.MaxDepth <= 0 {
		rc.MaxDepth = 3
	}
	if rc.MaxNodes <= 0 {
		rc.MaxNodes = 20
	}
	if rc.MemoryTopK <= 0 {
		rc.MemoryTopK = 5
	}
	if rc.MaxTokens <= 0 {
		rc.MaxTokens = 1024
	}
	trace := []string{}
	type node struct {
		goal  string
		depth int
	}
	stack := []node{{goal: goal, depth: 0}}
	seen := map[string]bool{}
	expanded := 0
	start := time.Now()
	for len(stack) > 0 {
		if expanded >= rc.MaxNodes {
			return "", trace, fmt.Errorf("node budget exhausted")
		}
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[cur.goal] {
			continue
		}
		seen[cur.goal] = true
		expanded++

		// Retrieve memory
		ctxNotes := rc.retrieve(ctx, cur.goal)
		contextStr := formatHits(ctxNotes)
		// Expand
		prompt := buildRecursivePrompt(cur.goal, contextStr)
		resp, err := rc.generate(ctx, prompt)
		if err != nil {
			return "", trace, err
		}
		trace = append(trace, fmt.Sprintf("Goal: %s\n%s", cur.goal, resp))

		// Add to memory (summary of reasoning)
		_ = rc.remember(ctx, fmt.Sprintf("Reasoning for '%s': %s", cur.goal, summarize(resp)))

		if ans := extractFinal(resp); ans != "" {
			// Verify if configured
			if rc.Verifier != nil {
				ok, score, reason := rc.Verifier.Verify(cur.goal, ans)
				if !ok && cur.depth < rc.MaxDepth {
					// Push a corrective subgoal based on verifier feedback
					fix := fmt.Sprintf("Address verifier feedback: %s", reason)
					stack = append(stack, node{goal: fix, depth: cur.depth + 1})
					continue
				}
				trace = append(trace, fmt.Sprintf("Verifier score=%.2f: %s", score, reason))
				if rc.Persist {
					_ = rc.persistResult(ctx, cur.goal, ans, trace, score, reason)
				}
			}
			return ans, trace, nil
		}
		if cur.depth < rc.MaxDepth {
			subs := extractSubgoals(resp, 3)
			// Push in reverse to respect ordering when LIFO
			for i := len(subs) - 1; i >= 0; i-- {
				g := strings.TrimSpace(subs[i])
				if g != "" && !seen[g] {
					stack = append(stack, node{goal: g, depth: cur.depth + 1})
				}
			}
		}
		// Soft time guard: 30s default
		if time.Since(start) > 30*time.Second {
			break
		}
	}
	return "", trace, fmt.Errorf("no solution found within limits")
}

func (rc *RecursiveController) generate(ctx context.Context, prompt string) (string, error) {
	if rc.Offline || rc.Model == nil {
		return rc.offlineGenerate(prompt), nil
	}
	// TODO: Fix ai.GenerateRequest reference
	// resp, err := rc.Model.Generate(ctx, &ai.GenerateRequest{
	// 	Prompt:      prompt,
	// 	Temperature: rc.Temperature,
	// 	MaxTokens:   rc.MaxTokens,
	// 	TopP:        0.9,
	// })

	// Placeholder for now
	var resp interface{}
	var err error
	_ = resp // Suppress unused variable warning
	_ = err  // Suppress unused variable warning
	// TODO: Fix resp.Text reference
	// if err != nil {
	// 	return "", err
	// }
	// return strings.TrimSpace(resp.Text), nil

	// Fallback to offline generation
	return rc.offlineGenerate(prompt), nil
}

func (rc *RecursiveController) remember(ctx context.Context, text string) error {
	if rc.Memory == nil {
		return nil
	}
	var vec []float64
	var err error
	_ = err // Suppress unused variable warning
	// TODO: Fix rc.Model.EmbedText() method call
	// if rc.EmbedModel != "" && rc.Model != nil {
	// 	var e []float64
	// 	e, err = rc.Model.EmbedText(ctx, rc.EmbedModel, text)
	// 	if err == nil {
	// 		vec = e
	// 	}
	// }
	if vec == nil {
		// Fallback: use UDL vector system hashing if available
		// TODO: Fix rc.UDL.GenerateEmbedding() method call
		// if rc.UDL != nil {
		// 	if e32, err2 := rc.UDL.GenerateEmbedding(ctx, text); err2 == nil {
		// 		v := make([]float64, len(e32))
		// 		for i := range e32 {
		// 			v[i] = float64(e32[i])
		// 		}
		// 		vec = v
		// 	}
		// }
	}
	if vec == nil {
		// Final fallback: deterministic hash embedding
		vec = hashEmbed(text, 384)
	}
	if vec != nil {
		rc.Memory.Add(text, vec)
	}
	return nil
}

func (rc *RecursiveController) retrieve(ctx context.Context, query string) []MemoryHit {
	if rc.Memory == nil || rc.Memory.Count() == 0 {
		return nil
	}
	var vec []float64
	// TODO: Fix rc.Model.EmbedText() method call
	// if rc.EmbedModel != "" && rc.Model != nil {
	// 	if e, err := rc.Model.EmbedText(ctx, rc.EmbedModel, query); err == nil {
	// 		vec = e
	// 	}
	// }
	if vec == nil {
		// TODO: Fix rc.UDL.GenerateEmbedding() method call
		// if rc.UDL != nil {
		// 	if e32, err := rc.UDL.GenerateEmbedding(ctx, query); err == nil {
		// 		v := make([]float64, len(e32))
		// 		for i := range e32 {
		// 			v[i] = float64(e32[i])
		// 		}
		// 		vec = v
		// 	}
		// }
	}
	if vec == nil {
		vec = hashEmbed(query, 384)
	}
	if vec == nil {
		return nil
	}
	return rc.Memory.Search(vec, rc.MemoryTopK)
}

func buildRecursivePrompt(goal, context string) string {
	b := &strings.Builder{}
	fmt.Fprintf(b, "You are a careful problem solver.\n")
	if strings.TrimSpace(context) != "" {
		fmt.Fprintf(b, "Relevant notes:\n%s\n\n", context)
	}
	fmt.Fprintf(b, "Task: %s\n\n", goal)
	fmt.Fprintf(b, "If you can fully solve now, write:\nFinal Answer: <concise solution>.\n")
	fmt.Fprintf(b, "Otherwise, propose up to 3 concrete subgoals as a bullet list starting with '- '.\n")
	fmt.Fprintf(b, "Then give a brief rationale.\n\n")
	return b.String()
}

func extractFinal(s string) string {
	lines := strings.Split(s, "\n")
	for _, ln := range lines {
		L := strings.TrimSpace(ln)
		if strings.HasPrefix(strings.ToLower(L), "final answer:") {
			return strings.TrimSpace(strings.TrimPrefix(L, "Final Answer:"))
		}
		if strings.HasPrefix(strings.ToLower(L), "conclusion:") {
			return strings.TrimSpace(strings.TrimPrefix(L, "Conclusion:"))
		}
	}
	return ""
}

func extractSubgoals(s string, max int) []string {
	out := []string{}
	for _, ln := range strings.Split(s, "\n") {
		L := strings.TrimSpace(ln)
		if strings.HasPrefix(L, "- ") {
			out = append(out, strings.TrimPrefix(L, "- "))
		} else if len(L) > 2 && (L[0] >= '0' && L[0] <= '9') && (L[1] == '.' || L[1] == ')') {
			out = append(out, strings.TrimSpace(L[2:]))
		}
		if max > 0 && len(out) >= max {
			break
		}
	}
	return out
}

func formatHits(hits []MemoryHit) string {
	if len(hits) == 0 {
		return ""
	}
	b := &strings.Builder{}
	for i, h := range hits {
		fmt.Fprintf(b, "%d) %s (%.2f)\n", i+1, h.Text, h.Score)
	}
	return b.String()
}

func summarize(s string) string {
	// crude summarizer: first 240 chars single-line
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 240 {
		s = s[:240] + "..."
	}
	return s
}

// offlineGenerate produces deterministic subgoals or a final answer using prompt context
func (rc *RecursiveController) offlineGenerate(prompt string) string {
	// detect context presence
	hasContext := strings.Contains(prompt, "Relevant notes:")
	goal := ""
	for _, ln := range strings.Split(prompt, "\n") {
		if strings.HasPrefix(strings.TrimSpace(ln), "Task:") {
			goal = strings.TrimSpace(strings.TrimPrefix(ln, "Task:"))
			break
		}
	}
	if goal == "" {
		goal = "the task"
	}
	if !hasContext {
		// propose subgoals
		return fmt.Sprintf("- Clarify constraints and success metrics for %s\n- Identify 3 high-impact actions and owners\n- Draft timeline, risks, and validation checks\n\nRationale: Start with constraints, then actions, then plan.", goal)
	}
	// generate final answer
	return fmt.Sprintf("Final Answer: A practical plan for %s with clear owners, milestones, and measurable success metrics, leveraging prior notes to de-risk execution.", goal)
}

// persistResult stores the final answer and trace in HANA via UDL and adds a vector entry
func (rc *RecursiveController) persistResult(ctx context.Context, goal, answer string, trace []string, score float64, reason string) error {
	if rc.UDL == nil {
		return nil
	}
	// Store message record
	// TODO: Fix rc.UDL.StoreAgentMessage() method call
	// payload := map[string]interface{}{
	// 	"goal":            goal,
	// 	"final_answer":    answer,
	// 	"verifier_score":  score,
	// 	"verifier_reason": reason,
	// 	"trace_joined":    strings.Join(trace, "\n---\n"),
	// 	"timestamp":       time.Now().Unix(),
	// }
	// _ = rc.UDL.StoreAgentMessage(ctx, rc.AgentID, "recursive_reasoning", payload)
	// Store final answer vector as memory
	_ = rc.remember(ctx, fmt.Sprintf("Final answer for '%s': %s", goal, answer))
	return nil
}

// -------- Verifier (simple checklist) --------

type Verifier interface {
	Verify(goal string, answer string) (ok bool, score float64, reason string)
}

type ChecklistVerifier struct{ criteria []string }

func NewChecklistVerifier(criteria []string) *ChecklistVerifier {
	// Clean
	out := make([]string, 0, len(criteria))
	for _, c := range criteria {
		c = strings.TrimSpace(c)
		if c != "" {
			out = append(out, c)
		}
	}
	return &ChecklistVerifier{criteria: out}
}

func (cv *ChecklistVerifier) Verify(goal, answer string) (bool, float64, string) {
	if len(cv.criteria) == 0 {
		return true, 1.0, "no criteria configured"
	}
	matched := 0
	missing := []string{}
	lower := strings.ToLower(answer)
	for _, c := range cv.criteria {
		// heuristic: each criterion is satisfied if key terms appear
		ok := true
		for _, tok := range tokenizeCriterion(c) {
			if !strings.Contains(lower, strings.ToLower(tok)) {
				ok = false
				break
			}
		}
		if ok {
			matched++
		} else {
			missing = append(missing, c)
		}
	}
	score := float64(matched) / float64(len(cv.criteria))
	if score >= 0.75 {
		return true, score, "criteria satisfied"
	}
	return false, score, "missing: " + strings.Join(missing, "; ")
}

func tokenizeCriterion(c string) []string {
	// Split on common separators, keep simple tokens
	c = strings.ReplaceAll(c, ",", " ")
	c = strings.ReplaceAll(c, ";", " ")
	fields := strings.Fields(c)
	if len(fields) == 0 {
		return []string{c}
	}
	return fields
}

func parseCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// loadEnvFile reads a simple KEY=VALUE .env and sets os.Environ, tolerating 'export ' prefix
func loadEnvFile(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	lines := strings.Split(string(b), "\n")
	for _, ln := range lines {
		s := strings.TrimSpace(ln)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		if strings.HasPrefix(s, "export ") {
			s = strings.TrimSpace(strings.TrimPrefix(s, "export "))
		}
		eq := strings.IndexByte(s, '=')
		if eq <= 0 {
			continue
		}
		k := strings.TrimSpace(s[:eq])
		v := strings.TrimSpace(s[eq+1:])
		v = strings.Trim(v, "\"'")
		_ = os.Setenv(k, v)
	}
	return nil
}

// hashEmbed creates a deterministic embedding from text
func hashEmbed(text string, dim int) []float64 {
	if dim <= 0 {
		dim = 384
	}
	h := sha256.Sum256([]byte(text))
	out := make([]float64, dim)
	for i := 0; i < dim; i++ {
		b := h[i%len(h)]
		out[i] = (float64(b) / 127.5) - 1.0
	}
	return out
}
