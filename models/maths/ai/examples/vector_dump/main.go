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
)

func getenv(k, d string) string {
    if v := os.Getenv(k); v != "" {
        return v
    }
    return d
}

func main() {
    // Flags
    envFile := flag.String("env", getenv("DOTENV_FILE", "/Users/apple/projects/agenticAiETH/.env"), "Path to .env to preload (optional)")
    collection := flag.String("collection", getenv("MEMORY_COLLECTION", "agent_memory"), "Vector collection name")
    query := flag.String("query", "", "Query text to embed and search")
    topK := flag.Int("top", 10, "Top-K results to show")
    timeout := flag.Duration("timeout", 15*time.Second, "Request timeout")
    flag.Parse()

    if f := strings.TrimSpace(*envFile); f != "" {
        if st, err := os.Stat(f); err == nil && !st.IsDir() {
            _ = loadEnvFile(f)
        }
    }

    if strings.TrimSpace(*collection) == "" {
        log.Fatal("-collection is required")
    }
    if strings.TrimSpace(*query) == "" {
        log.Fatal("-query is required")
    }

    // HANA env detection: use A2A_DATABASE_URL or build from HANA_*
    dsn := strings.TrimSpace(os.Getenv("A2A_DATABASE_URL"))
    if dsn == "" {
        host := getenv("HANA_HOST", "")
        port := getenv("HANA_PORT", "443")
        user := firstNonEmptyEnv(os.Getenv("HANA_USERNAME"), os.Getenv("HANA_USER"))
        pass := os.Getenv("HANA_PASSWORD")
        dbname := getenv("HANA_DATABASE_NAME", "")
        defSchema := strings.TrimSpace(os.Getenv("HANA_DEFAULT_SCHEMA"))
        tlsName := getenv("HANA_TLS_SERVER_NAME", "")
        if tlsName == "" && port == "443" && host != "" { tlsName = host }
        tlsCA := getenv("HANA_TLS_ROOT_CA_FILE", "")
        tlsInsec := getenv("HANA_TLS_INSECURE_SKIP_VERIFY", "")
        if host == "" || user == "" {
            log.Fatalf("HANA env not detected: set HANA_HOST, HANA_USER/HANA_USERNAME, and HANA_PASSWORD or provide A2A_DATABASE_URL")
        }
        dsn = buildDSN(host, port, user, pass, dbname, defSchema, tlsName, tlsCA, tlsInsec)
    }

    // Connect directly via HDB
    ctx, cancel := context.WithTimeout(context.Background(), *timeout)
    defer cancel()
    hdb, err := sql.Open("hdb", dsn)
    if err != nil { log.Fatalf("open HANA: %v", err) }
    defer hdb.Close()
    if err := hdb.PingContext(ctx); err != nil { log.Fatalf("ping HANA: %v", err) }
    // Set schema explicitly if provided
    sch := strings.TrimSpace(os.Getenv("HANA_DEFAULT_SCHEMA"))
    if sch == "" { sch = strings.TrimSpace(os.Getenv("HANA_USERNAME")) }
    if sch == "" { sch = strings.TrimSpace(os.Getenv("HANA_USER")) }
    if sch != "" {
        if _, err := hdb.ExecContext(ctx, "SET SCHEMA "+sch); err != nil {
            log.Printf("WARN: SET SCHEMA %s failed: %v", sch, err)
        }
    }

    // Embed query using hash fallback
    qvec := hashEmbed(*query, 384)
    q32 := make([]float32, len(qvec))
    for i := range qvec { q32[i] = float32(qvec[i]) }

    // Fetch recent vectors from collection
    rows, err := hdb.QueryContext(ctx, `SELECT vector_id, vector_data FROM vectors WHERE collection_name = ? ORDER BY created_at DESC LIMIT 500`, *collection)
    if err != nil { log.Fatalf("query vectors: %v", err) }
    defer rows.Close()
    type rowrec struct { id string; vec []float32 }
    recs := make([]rowrec, 0, 500)
    for rows.Next() {
        var id string
        var data []byte
        if err := rows.Scan(&id, &data); err != nil { continue }
        vec := bytesToVec(data)
        if len(vec) > 0 { recs = append(recs, rowrec{id: id, vec: vec}) }
    }
    if len(recs) == 0 { fmt.Println("No results"); return }

    // Score cosine
    type scored struct { id string; score float64 }
    scores := make([]scored, 0, len(recs))
    for _, r := range recs {
        s := cosine(q32, r.vec)
        scores = append(scores, scored{id: r.id, score: float64(s)})
    }
    // Sort desc
    sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
    if len(scores) > *topK { scores = scores[:*topK] }

    fmt.Printf("Collection: %s\nQuery: %s\nTop: %d\n---\n", *collection, *query, *topK)
    for i, s := range scores {
        var txt sql.NullString
        row := hdb.QueryRowContext(ctx, `SELECT text FROM vector_texts WHERE collection_name = ? AND vector_id = ?`, *collection, s.id)
        _ = row.Scan(&txt)
        t := txt.String
        if strings.TrimSpace(t) == "" { t = "<no text stored>" }
        if len(t) > 300 { t = t[:300] + "..." }
        fmt.Printf("%2d) score=%.4f id=%s\n    %s\n", i+1, s.score, s.id, oneLine(t))
    }
}

func extractTextFromRaw(md map[string]interface{}) string {
    if md == nil { return "" }
    raw, _ := md["raw"].(string)
    if raw == "" { return "" }
    idx := strings.Index(raw, "text:")
    if idx < 0 { return "" }
    val := strings.TrimSpace(raw[idx+len("text:"):])
    val = strings.TrimPrefix(val, "map[")
    if j := strings.Index(val, "]"); j >= 0 { val = val[:j] }
    return strings.TrimSpace(val)
}

func oneLine(s string) string {
    s = strings.ReplaceAll(s, "\n", " ")
    s = strings.ReplaceAll(s, "\r", " ")
    return strings.TrimSpace(s)
}

func firstNonEmptyEnv(vals ...string) string {
    for _, v := range vals { if strings.TrimSpace(v) != "" { return strings.TrimSpace(v) } }
    return ""
}

func buildDSN(host, port, user, pass, dbname, defSchema, tlsName, tlsCA, tlsInsec string) string {
    var b strings.Builder
    b.WriteString("hdb://")
    if user != "" {
        b.WriteString(urlEscape(user))
        if pass != "" { b.WriteString(":" + urlEscape(pass)) }
        b.WriteString("@")
    }
    b.WriteString(host)
    if port != "" { b.WriteString(":" + port) }
    if dbname != "" { b.WriteString("/" + dbname) }
    qs := []string{}
    if defSchema != "" { qs = append(qs, "defaultSchema="+urlEscape(defSchema)) }
    if tlsName != "" { qs = append(qs, "TLSServerName="+urlEscape(tlsName)) }
    if tlsCA != "" { qs = append(qs, "TLSRootCAFile="+urlEscape(tlsCA)) }
    if tlsInsec != "" { qs = append(qs, "TLSInsecureSkipVerify="+urlEscape(tlsInsec)) }
    if len(qs) > 0 { b.WriteString("?" + strings.Join(qs, "&")) }
    return b.String()
}

func urlEscape(s string) string {
    r := strings.ReplaceAll(s, " ", "%20")
    r = strings.ReplaceAll(r, "@", "%40")
    r = strings.ReplaceAll(r, ":", "%3A")
    r = strings.ReplaceAll(r, "/", "%2F")
    return r
}

func bytesToVec(b []byte) []float32 {
    if len(b)%4 != 0 { return nil }
    n := len(b) / 4
    out := make([]float32, n)
    for i := 0; i < n; i++ {
        bits := uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
        out[i] = float32(bits) / 1000000
    }
    return out
}

func cosine(a, b []float32) float32 {
    if len(a) != len(b) { return 0 }
    var dot, na2, nb2 float32
    for i := range a {
        dot += a[i] * b[i]
        na2 += a[i] * a[i]
        nb2 += b[i] * b[i]
    }
    if na2 == 0 || nb2 == 0 { return 0 }
    return dot / (float32(math.Sqrt(float64(na2))) * float32(math.Sqrt(float64(nb2))))
}

func hashEmbed(text string, dim int) []float64 {
    if dim <= 0 { dim = 384 }
    h := sha256.Sum256([]byte(text))
    out := make([]float64, dim)
    for i := 0; i < dim; i++ {
        b := h[i%len(h)]
        out[i] = (float64(b)/127.5) - 1.0
    }
    return out
}

func loadEnvFile(path string) error {
    b, err := os.ReadFile(path)
    if err != nil { return err }
    for _, ln := range strings.Split(string(b), "\n") {
        s := strings.TrimSpace(ln)
        if s == "" || strings.HasPrefix(s, "#") { continue }
        if strings.HasPrefix(s, "export ") { s = strings.TrimSpace(strings.TrimPrefix(s, "export ")) }
        eq := strings.IndexByte(s, '=')
        if eq <= 0 { continue }
        k := strings.TrimSpace(s[:eq])
        v := strings.TrimSpace(s[eq+1:])
        v = strings.Trim(v, "\"'")
        _ = os.Setenv(k, v)
    }
    return nil
}
