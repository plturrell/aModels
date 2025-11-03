package main

import (
	"database/sql"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
)

// ARC-AGI-2 handlers - same logic as ARC-AGI but with different table names

func (s *Server) listTasks2(w http.ResponseWriter, r *http.Request) {
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

	query := "SELECT TASK_ID, SPLIT, CREATED_AT FROM ARC_AGI_2_TASKS"
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

func (s *Server) getTask2(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["taskId"]

	query := "SELECT TASK_ID, SPLIT, RAW_JSON, CREATED_AT FROM ARC_AGI_2_TASKS WHERE TASK_ID = ?"

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

func (s *Server) getTaskSamples2(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["taskId"]

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, INPUT_GRID, OUTPUT_GRID FROM ARC_AGI_2_SAMPLES WHERE TASK_ID = ? ORDER BY SAMPLE_TYPE, SAMPLE_INDEX"

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

func (s *Server) getTaskGrids2(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["taskId"]

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX, CELL_VALUE FROM ARC_AGI_2_GRIDS WHERE TASK_ID = ? ORDER BY SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX"

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

func (s *Server) listSamples2(w http.ResponseWriter, r *http.Request) {
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

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, INPUT_GRID, OUTPUT_GRID FROM ARC_AGI_2_SAMPLES WHERE 1=1"
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

func (s *Server) listGrids2(w http.ResponseWriter, r *http.Request) {
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

	query := "SELECT TASK_ID, SPLIT, SAMPLE_TYPE, SAMPLE_INDEX, GRID_ROLE, ROW_INDEX, COL_INDEX, CELL_VALUE FROM ARC_AGI_2_GRIDS WHERE 1=1"
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

func (s *Server) getStats2(w http.ResponseWriter, r *http.Request) {
	stats := make(map[string]interface{})

	// Count tasks
	var taskCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM ARC_AGI_2_TASKS").Scan(&taskCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_tasks"] = taskCount

	// Count by split
	rows, err := s.db.Query("SELECT SPLIT, COUNT(*) FROM ARC_AGI_2_TASKS GROUP BY SPLIT")
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
	if err := s.db.QueryRow("SELECT COUNT(*) FROM ARC_AGI_2_SAMPLES").Scan(&sampleCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_samples"] = sampleCount

	// Count grid cells
	var cellCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM ARC_AGI_2_GRIDS").Scan(&cellCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_grid_cells"] = cellCount

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    stats,
	})
}
