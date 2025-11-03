package main

import (
	"database/sql"
	"net/http"
	"strconv"
)

type HellaSwagExample struct {
	Ind           int      `json:"ind"`
	Split         string   `json:"split"`
	SplitType     string   `json:"split_type"`
	ActivityLabel string   `json:"activity_label"`
	CtxA          string   `json:"ctx_a"`
	CtxB          string   `json:"ctx_b"`
	Ctx           string   `json:"ctx"`
	Label         int      `json:"label"`
	SourceID      string   `json:"source_id"`
	Endings       []string `json:"endings"`
	CreatedAt     string   `json:"created_at"`
}

func (s *Server) listHellaSwagExamples(w http.ResponseWriter, r *http.Request) {
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

	query := "SELECT IND, SPLIT, SPLIT_TYPE, ACTIVITY_LABEL, CTX_A, CTX_B, CTX, LABEL, SOURCE_ID, CREATED_AT FROM HELLASWAG_EXAMPLES"
	args := []interface{}{}

	if split != "" {
		query += " WHERE SPLIT = ?"
		args = append(args, split)
	}

	query += " ORDER BY IND LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	examples := []HellaSwagExample{}
	for rows.Next() {
		var ex HellaSwagExample
		var splitType, activityLabel, ctxA, ctxB, ctx, sourceID sql.NullString
		var label sql.NullInt64

		if err := rows.Scan(&ex.Ind, &ex.Split, &splitType, &activityLabel, &ctxA, &ctxB, &ctx, &label, &sourceID, &ex.CreatedAt); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}

		if splitType.Valid {
			ex.SplitType = splitType.String
		}
		if activityLabel.Valid {
			ex.ActivityLabel = activityLabel.String
		}
		if ctxA.Valid {
			ex.CtxA = ctxA.String
		}
		if ctxB.Valid {
			ex.CtxB = ctxB.String
		}
		if ctx.Valid {
			ex.Ctx = ctx.String
		}
		if label.Valid {
			ex.Label = int(label.Int64)
		}
		if sourceID.Valid {
			ex.SourceID = sourceID.String
		}

		// Fetch endings
		endingRows, err := s.db.Query("SELECT ENDING_TEXT FROM HELLASWAG_ENDINGS WHERE IND = ? ORDER BY ENDING_INDEX", ex.Ind)
		if err == nil {
			defer endingRows.Close()
			for endingRows.Next() {
				var ending string
				if err := endingRows.Scan(&ending); err == nil {
					ex.Endings = append(ex.Endings, ending)
				}
			}
		}

		examples = append(examples, ex)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(examples),
		"offset":  offset,
		"limit":   limit,
		"data":    examples,
	})
}

func (s *Server) getHellaSwagExample(w http.ResponseWriter, r *http.Request) {
	indStr := r.URL.Query().Get("ind")
	if indStr == "" {
		respondError(w, http.StatusBadRequest, "ind parameter required")
		return
	}

	ind, err := strconv.Atoi(indStr)
	if err != nil {
		respondError(w, http.StatusBadRequest, "invalid ind parameter")
		return
	}

	query := "SELECT IND, SPLIT, SPLIT_TYPE, ACTIVITY_LABEL, CTX_A, CTX_B, CTX, LABEL, SOURCE_ID, CREATED_AT FROM HELLASWAG_EXAMPLES WHERE IND = ?"

	var ex HellaSwagExample
	var splitType, activityLabel, ctxA, ctxB, ctx, sourceID sql.NullString
	var label sql.NullInt64

	err = s.db.QueryRow(query, ind).Scan(&ex.Ind, &ex.Split, &splitType, &activityLabel, &ctxA, &ctxB, &ctx, &label, &sourceID, &ex.CreatedAt)
	if err == sql.ErrNoRows {
		respondError(w, http.StatusNotFound, "Example not found")
		return
	}
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if splitType.Valid {
		ex.SplitType = splitType.String
	}
	if activityLabel.Valid {
		ex.ActivityLabel = activityLabel.String
	}
	if ctxA.Valid {
		ex.CtxA = ctxA.String
	}
	if ctxB.Valid {
		ex.CtxB = ctxB.String
	}
	if ctx.Valid {
		ex.Ctx = ctx.String
	}
	if label.Valid {
		ex.Label = int(label.Int64)
	}
	if sourceID.Valid {
		ex.SourceID = sourceID.String
	}

	// Fetch endings
	endingRows, err := s.db.Query("SELECT ENDING_TEXT FROM HELLASWAG_ENDINGS WHERE IND = ? ORDER BY ENDING_INDEX", ex.Ind)
	if err == nil {
		defer endingRows.Close()
		for endingRows.Next() {
			var ending string
			if err := endingRows.Scan(&ending); err == nil {
				ex.Endings = append(ex.Endings, ending)
			}
		}
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    ex,
	})
}

func (s *Server) getHellaSwagStats(w http.ResponseWriter, r *http.Request) {
	stats := make(map[string]interface{})

	// Count total examples
	var totalCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM HELLASWAG_EXAMPLES").Scan(&totalCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_examples"] = totalCount

	// Count by split
	rows, err := s.db.Query("SELECT SPLIT, COUNT(*) FROM HELLASWAG_EXAMPLES GROUP BY SPLIT")
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
	stats["examples_by_split"] = splitCounts

	// Count by split_type
	rows2, err := s.db.Query("SELECT SPLIT_TYPE, COUNT(*) FROM HELLASWAG_EXAMPLES WHERE SPLIT_TYPE IS NOT NULL GROUP BY SPLIT_TYPE")
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows2.Close()

	splitTypeCounts := make(map[string]int)
	for rows2.Next() {
		var splitType string
		var count int
		if err := rows2.Scan(&splitType, &count); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		splitTypeCounts[splitType] = count
	}
	stats["examples_by_split_type"] = splitTypeCounts

	// Count total endings
	var endingsCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM HELLASWAG_ENDINGS").Scan(&endingsCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_endings"] = endingsCount

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    stats,
	})
}

func (s *Server) searchHellaSwagExamples(w http.ResponseWriter, r *http.Request) {
	searchTerm := r.URL.Query().Get("q")
	if searchTerm == "" {
		respondError(w, http.StatusBadRequest, "q parameter required")
		return
	}

	split := r.URL.Query().Get("split")
	limitStr := r.URL.Query().Get("limit")

	limit := 50
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}

	query := "SELECT IND, SPLIT, SPLIT_TYPE, ACTIVITY_LABEL, CTX_A, CTX_B, CTX, LABEL, SOURCE_ID, CREATED_AT FROM HELLASWAG_EXAMPLES WHERE CONTAINS(CTX, ?, FUZZY(0.8))"
	args := []interface{}{searchTerm}

	if split != "" {
		query += " AND SPLIT = ?"
		args = append(args, split)
	}

	query += " ORDER BY IND LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	examples := []HellaSwagExample{}
	for rows.Next() {
		var ex HellaSwagExample
		var splitType, activityLabel, ctxA, ctxB, ctx, sourceID sql.NullString
		var label sql.NullInt64

		if err := rows.Scan(&ex.Ind, &ex.Split, &splitType, &activityLabel, &ctxA, &ctxB, &ctx, &label, &sourceID, &ex.CreatedAt); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}

		if splitType.Valid {
			ex.SplitType = splitType.String
		}
		if activityLabel.Valid {
			ex.ActivityLabel = activityLabel.String
		}
		if ctxA.Valid {
			ex.CtxA = ctxA.String
		}
		if ctxB.Valid {
			ex.CtxB = ctxB.String
		}
		if ctx.Valid {
			ex.Ctx = ctx.String
		}
		if label.Valid {
			ex.Label = int(label.Int64)
		}
		if sourceID.Valid {
			ex.SourceID = sourceID.String
		}

		// Fetch endings
		endingRows, err := s.db.Query("SELECT ENDING_TEXT FROM HELLASWAG_ENDINGS WHERE IND = ? ORDER BY ENDING_INDEX", ex.Ind)
		if err == nil {
			defer endingRows.Close()
			for endingRows.Next() {
				var ending string
				if err := endingRows.Scan(&ending); err == nil {
					ex.Endings = append(ex.Endings, ending)
				}
			}
		}

		examples = append(examples, ex)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(examples),
		"data":    examples,
	})
}
