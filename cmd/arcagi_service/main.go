package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	_ "github.com/SAP/go-hdb/driver"
	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
)

type Task struct {
	TaskID    string `json:"task_id"`
	Split     string `json:"split"`
	RawJSON   string `json:"raw_json,omitempty"`
	CreatedAt string `json:"created_at"`
}

type Sample struct {
	TaskID      string `json:"task_id"`
	Split       string `json:"split"`
	SampleType  string `json:"sample_type"`
	SampleIndex int    `json:"sample_index"`
	InputGrid   string `json:"input_grid"`
	OutputGrid  string `json:"output_grid"`
}

type GridCell struct {
	TaskID      string `json:"task_id"`
	Split       string `json:"split"`
	SampleType  string `json:"sample_type"`
	SampleIndex int    `json:"sample_index"`
	GridRole    string `json:"grid_role"`
	RowIndex    int    `json:"row_index"`
	ColIndex    int    `json:"col_index"`
	CellValue   int    `json:"cell_value"`
}

type Server struct {
	db     *sql.DB
	router *mux.Router
}

func main() {
	// Try multiple .env locations
	envPaths := []string{
		"../../agenticAiETH_layer4/.env",
		"../agenticAiETH_layer4/.env",
		".env",
	}

	loaded := false
	for _, envPath := range envPaths {
		if err := godotenv.Load(envPath); err == nil {
			log.Printf("Loaded .env from: %s", envPath)
			loaded = true
			break
		}
	}

	if !loaded {
		log.Printf("Warning: No .env file loaded, using environment variables")
	}

	host := os.Getenv("HANA_HOST")
	port := os.Getenv("HANA_PORT")
	user := os.Getenv("HANA_USER")
	password := os.Getenv("HANA_PASSWORD")

	if host == "" || port == "" || user == "" || password == "" {
		log.Fatal("Missing required HANA environment variables")
	}

	dsn := fmt.Sprintf("hdb://%s:%s@%s:%s", user, password, host, port)

	db, err := sql.Open("hdb", dsn)
	if err != nil {
		log.Fatalf("Failed to open HANA connection: %v", err)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping HANA: %v", err)
	}

	log.Println("Connected to HANA successfully")

	server := &Server{
		db:     db,
		router: mux.NewRouter(),
	}

	server.setupRoutes()

	serverPort := os.Getenv("ARCAGI_SERVICE_PORT")
	if serverPort == "" {
		serverPort = "8090"
	}

	log.Printf("Starting ARC-AGI Query Service on port %s", serverPort)
	if err := http.ListenAndServe(":"+serverPort, server.router); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func (s *Server) setupRoutes() {
	api := s.router.PathPrefix("/api/v1/arcagi").Subrouter()

	// Task routes (ARC-AGI)
	api.HandleFunc("/tasks", s.listTasks).Methods("GET")
	api.HandleFunc("/tasks/{taskId}", s.getTask).Methods("GET")
	api.HandleFunc("/tasks/{taskId}/samples", s.getTaskSamples).Methods("GET")
	api.HandleFunc("/tasks/{taskId}/grids", s.getTaskGrids).Methods("GET")

	// Sample routes
	api.HandleFunc("/samples", s.listSamples).Methods("GET")

	// Grid routes
	api.HandleFunc("/grids", s.listGrids).Methods("GET")

	// Stats route
	api.HandleFunc("/stats", s.getStats).Methods("GET")

	// ARC-AGI-2 routes
	api2 := s.router.PathPrefix("/api/v1/arcagi2").Subrouter()
	api2.HandleFunc("/tasks", s.listTasks2).Methods("GET")
	api2.HandleFunc("/tasks/{taskId}", s.getTask2).Methods("GET")
	api2.HandleFunc("/tasks/{taskId}/samples", s.getTaskSamples2).Methods("GET")
	api2.HandleFunc("/tasks/{taskId}/grids", s.getTaskGrids2).Methods("GET")
	api2.HandleFunc("/samples", s.listSamples2).Methods("GET")
	api2.HandleFunc("/grids", s.listGrids2).Methods("GET")
	api2.HandleFunc("/stats", s.getStats2).Methods("GET")

	// BoolQ routes
	boolq := s.router.PathPrefix("/api/v1/boolq").Subrouter()
	boolq.HandleFunc("/questions", s.listBoolQQuestions).Methods("GET")
	boolq.HandleFunc("/question", s.getBoolQQuestion).Methods("GET")
	boolq.HandleFunc("/search", s.searchBoolQQuestions).Methods("GET")
	boolq.HandleFunc("/stats", s.getBoolQStats).Methods("GET")

	// HellaSwag routes
	hellaswag := s.router.PathPrefix("/api/v1/hellaswag").Subrouter()
	hellaswag.HandleFunc("/examples", s.listHellaSwagExamples).Methods("GET")
	hellaswag.HandleFunc("/example", s.getHellaSwagExample).Methods("GET")
	hellaswag.HandleFunc("/search", s.searchHellaSwagExamples).Methods("GET")
	hellaswag.HandleFunc("/stats", s.getHellaSwagStats).Methods("GET")

	// Health check
	s.router.HandleFunc("/health", s.healthCheck).Methods("GET")
}

func (s *Server) listTasks(w http.ResponseWriter, r *http.Request) {
	split := r.URL.Query().Get("split")
	limitStr := r.URL.Query().Get("limit")
	offsetStr := r.URL.Query().Get("offset")

	limit := 100
	offset := 0

	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}
	if offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil {
			offset = o
		}
	}

	query := "SELECT TASK_ID, SPLIT, CREATED_AT FROM ARC_AGI_TASKS"
	args := []interface{}{}

	if split != "" {
		query += " WHERE SPLIT = ?"
		args = append(args, split)
	}

	query += " ORDER BY TASK_ID LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	tasks := []Task{}
	for rows.Next() {
		var task Task
		if err := rows.Scan(&task.TaskID, &task.Split, &task.CreatedAt); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		tasks = append(tasks, task)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(tasks),
		"offset":  offset,
		"limit":   limit,
		"data":    tasks,
	})
}

func (s *Server) getTask(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["taskId"]

	query := "SELECT TASK_ID, SPLIT, RAW_JSON, CREATED_AT FROM ARC_AGI_TASKS WHERE TASK_ID = ?"

	var task Task
	err := s.db.QueryRow(query, taskID).Scan(&task.TaskID, &task.Split, &task.RawJSON, &task.CreatedAt)
	if err == sql.ErrNoRows {
		respondError(w, http.StatusNotFound, "Task not found")
		return
	}
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    task,
	})
}

func (s *Server) getTaskSamples(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["taskId"]

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, INPUT_GRID, OUTPUT_GRID FROM ARC_AGI_SAMPLES WHERE TASK_ID = ? ORDER BY SAMPLE_TYPE, SAMPLE_INDEX"

	rows, err := s.db.Query(query, taskID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	samples := []Sample{}
	for rows.Next() {
		var sample Sample
		if err := rows.Scan(&sample.TaskID, &sample.Split, &sample.SampleType, &sample.SampleIndex, &sample.InputGrid, &sample.OutputGrid); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		samples = append(samples, sample)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(samples),
		"data":    samples,
	})
}

func (s *Server) getTaskGrids(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["taskId"]

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX, CELL_VALUE FROM ARC_AGI_GRIDS WHERE TASK_ID = ? ORDER BY SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX"

	rows, err := s.db.Query(query, taskID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	cells := []GridCell{}
	for rows.Next() {
		var cell GridCell
		if err := rows.Scan(&cell.TaskID, &cell.Split, &cell.SampleType, &cell.SampleIndex, &cell.GridRole, &cell.RowIndex, &cell.ColIndex, &cell.CellValue); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		cells = append(cells, cell)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(cells),
		"data":    cells,
	})
}

func (s *Server) listSamples(w http.ResponseWriter, r *http.Request) {
	split := r.URL.Query().Get("split")
	sampleType := r.URL.Query().Get("type")
	limitStr := r.URL.Query().Get("limit")
	offsetStr := r.URL.Query().Get("offset")

	limit := 100
	offset := 0

	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}
	if offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil {
			offset = o
		}
	}

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, INPUT_GRID, OUTPUT_GRID FROM ARC_AGI_SAMPLES WHERE 1=1"
	args := []interface{}{}

	if split != "" {
		query += " AND SPLIT = ?"
		args = append(args, split)
	}
	if sampleType != "" {
		query += " AND SAMPLE_TYPE = ?"
		args = append(args, sampleType)
	}

	query += " ORDER BY TASK_ID, SAMPLE_TYPE, SAMPLE_INDEX LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	samples := []Sample{}
	for rows.Next() {
		var sample Sample
		if err := rows.Scan(&sample.TaskID, &sample.Split, &sample.SampleType, &sample.SampleIndex, &sample.InputGrid, &sample.OutputGrid); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		samples = append(samples, sample)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(samples),
		"offset":  offset,
		"limit":   limit,
		"data":    samples,
	})
}

func (s *Server) listGrids(w http.ResponseWriter, r *http.Request) {
	taskID := r.URL.Query().Get("taskId")
	limitStr := r.URL.Query().Get("limit")
	offsetStr := r.URL.Query().Get("offset")

	limit := 1000
	offset := 0

	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}
	if offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil {
			offset = o
		}
	}

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX, CELL_VALUE FROM ARC_AGI_GRIDS WHERE 1=1"
	args := []interface{}{}

	if taskID != "" {
		query += " AND TASK_ID = ?"
		args = append(args, taskID)
	}

	query += " ORDER BY TASK_ID, SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	cells := []GridCell{}
	for rows.Next() {
		var cell GridCell
		if err := rows.Scan(&cell.TaskID, &cell.Split, &cell.SampleType, &cell.SampleIndex, &cell.GridRole, &cell.RowIndex, &cell.ColIndex, &cell.CellValue); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		cells = append(cells, cell)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(cells),
		"offset":  offset,
		"limit":   limit,
		"data":    cells,
	})
}

func (s *Server) getStats(w http.ResponseWriter, r *http.Request) {
	stats := make(map[string]interface{})

	// Count tasks
	var taskCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM ARC_AGI_TASKS").Scan(&taskCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_tasks"] = taskCount

	// Count by split
	rows, err := s.db.Query("SELECT SPLIT, COUNT(*) FROM ARC_AGI_TASKS GROUP BY SPLIT")
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	splitCounts := make(map[string]int)
	for rows.Next() {
		var split string
		var count int
		if err := rows.Scan(&split, &count); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		splitCounts[split] = count
	}
	stats["tasks_by_split"] = splitCounts

	// Count samples
	var sampleCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM ARC_AGI_SAMPLES").Scan(&sampleCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_samples"] = sampleCount

	// Count grid cells
	var cellCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM ARC_AGI_GRIDS").Scan(&cellCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_grid_cells"] = cellCount

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    stats,
	})
}

func (s *Server) healthCheck(w http.ResponseWriter, r *http.Request) {
	if err := s.db.Ping(); err != nil {
		respondError(w, http.StatusServiceUnavailable, "Database connection failed")
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"status":  "healthy",
		"service": "arcagi-query-service",
	})
}

func respondJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(payload)
}

func respondError(w http.ResponseWriter, status int, message string) {
	respondJSON(w, status, map[string]interface{}{
		"success": false,
		"error":   message,
	})
}
