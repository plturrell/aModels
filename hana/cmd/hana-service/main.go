package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	_ "github.com/SAP/go-hdb/driver"
)

type sqlRequest struct {
	Query string        `json:"query"`
	Args  []interface{} `json:"args"`
}

type sqlResponse struct {
	Rows []map[string]interface{} `json:"rows"`
}

func main() {
	dsn := os.Getenv("HANA_DSN")
	if dsn == "" {
		log.Fatal("HANA_DSN is required, e.g. hdb://user:pass@host:39015")
	}

	db, err := sql.Open("hdb", dsn)
	if err != nil {
		log.Fatalf("open hdb: %v", err)
	}
	defer db.Close()

	maxOpen := 5
	if v := os.Getenv("HANA_MAX_OPEN_CONNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			maxOpen = n
		}
	}
	db.SetMaxOpenConns(maxOpen)
	db.SetConnMaxLifetime(30 * time.Minute)

	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := db.Ping(); err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte("unhealthy"))
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	http.HandleFunc("/sql", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var req sqlRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("invalid json"))
			return
		}
		if req.Query == "" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("query required"))
			return
		}

		rows, err := db.Query(req.Query, req.Args...)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(err.Error()))
			return
		}
		defer rows.Close()

		cols, err := rows.Columns()
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(err.Error()))
			return
		}
		var out []map[string]interface{}
		for rows.Next() {
			dests := make([]interface{}, len(cols))
			destPtrs := make([]interface{}, len(cols))
			for i := range dests {
				destPtrs[i] = &dests[i]
			}
			if err := rows.Scan(destPtrs...); err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte(err.Error()))
				return
			}
			rowMap := make(map[string]interface{}, len(cols))
			for i, c := range cols {
				rowMap[c] = dests[i]
			}
			out = append(out, rowMap)
		}
		if err := rows.Err(); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(err.Error()))
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(sqlResponse{Rows: out})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8083"
	}
	log.Printf("hana-service listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))

}


