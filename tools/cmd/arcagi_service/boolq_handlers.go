package main

import (
	"database/sql"
	"net/http"
	"strconv"
)

type BoolQQuestion struct {
	ID        int    `json:"id"`
	Split     string `json:"split"`
	Question  string `json:"question"`
	Passage   string `json:"passage"`
	Answer    bool   `json:"answer"`
	Title     string `json:"title"`
	CreatedAt string `json:"created_at"`
}

func (s *Server) listBoolQQuestions(w http.ResponseWriter, r *http.Request) {
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

	query := "SELECT ID, SPLIT, QUESTION, PASSAGE, ANSWER, TITLE, CREATED_AT FROM BOOLQ_QUESTIONS"
	args := []interface{}{}

	if split != "" {
		query += " WHERE SPLIT = ?"
		args = append(args, split)
	}

	query += " ORDER BY ID LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	questions := []BoolQQuestion{}
	for rows.Next() {
		var q BoolQQuestion
		var answerInt int
		if err := rows.Scan(&q.ID, &q.Split, &q.Question, &q.Passage, &answerInt, &q.Title, &q.CreatedAt); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		q.Answer = answerInt == 1
		questions = append(questions, q)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(questions),
		"offset":  offset,
		"limit":   limit,
		"data":    questions,
	})
}

func (s *Server) getBoolQQuestion(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	if idStr == "" {
		respondError(w, http.StatusBadRequest, "id parameter required")
		return
	}

	id, err := strconv.Atoi(idStr)
	if err != nil {
		respondError(w, http.StatusBadRequest, "invalid id parameter")
		return
	}

	query := "SELECT ID, SPLIT, QUESTION, PASSAGE, ANSWER, TITLE, CREATED_AT FROM BOOLQ_QUESTIONS WHERE ID = ?"

	var q BoolQQuestion
	var answerInt int
	err = s.db.QueryRow(query, id).Scan(&q.ID, &q.Split, &q.Question, &q.Passage, &answerInt, &q.Title, &q.CreatedAt)
	if err == sql.ErrNoRows {
		respondError(w, http.StatusNotFound, "Question not found")
		return
	}
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}

	q.Answer = answerInt == 1

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    q,
	})
}

func (s *Server) getBoolQStats(w http.ResponseWriter, r *http.Request) {
	stats := make(map[string]interface{})

	// Count total questions
	var totalCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM BOOLQ_QUESTIONS").Scan(&totalCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total_questions"] = totalCount

	// Count by split
	rows, err := s.db.Query("SELECT SPLIT, COUNT(*) FROM BOOLQ_QUESTIONS GROUP BY SPLIT")
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
	stats["questions_by_split"] = splitCounts

	// Count by answer
	var trueCount, falseCount int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM BOOLQ_QUESTIONS WHERE ANSWER = 1").Scan(&trueCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := s.db.QueryRow("SELECT COUNT(*) FROM BOOLQ_QUESTIONS WHERE ANSWER = 0").Scan(&falseCount); err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["true_answers"] = trueCount
	stats["false_answers"] = falseCount

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"data":    stats,
	})
}

func (s *Server) searchBoolQQuestions(w http.ResponseWriter, r *http.Request) {
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

	query := "SELECT ID, SPLIT, QUESTION, PASSAGE, ANSWER, TITLE, CREATED_AT FROM BOOLQ_QUESTIONS WHERE CONTAINS(QUESTION, ?, FUZZY(0.8))"
	args := []interface{}{searchTerm}

	if split != "" {
		query += " AND SPLIT = ?"
		args = append(args, split)
	}

	query += " ORDER BY ID LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	questions := []BoolQQuestion{}
	for rows.Next() {
		var q BoolQQuestion
		var answerInt int
		if err := rows.Scan(&q.ID, &q.Split, &q.Question, &q.Passage, &answerInt, &q.Title, &q.CreatedAt); err != nil {
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		q.Answer = answerInt == 1
		questions = append(questions, q)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"count":   len(questions),
		"data":    questions,
	})
}
