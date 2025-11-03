package piqa

// Phrase-Indexed Question Answering (PIQA) - Seo et al. 2018, EMNLP
// Paper: https://arxiv.org/abs/1804.07726
// GitHub: https://github.com/seominjoon/piqa
//
// PIQA enforces complete independence between document encoder and question encoder
// for scalable document comprehension via phrase retrieval.

// SQuAD data structures
type SQuADDataset struct {
	Version string         `json:"version"`
	Data    []SQuADArticle `json:"data"`
}

type SQuADArticle struct {
	Title      string           `json:"title"`
	Paragraphs []SQuADParagraph `json:"paragraphs"`
}

type SQuADParagraph struct {
	Context string    `json:"context"`
	QAs     []SQuADQA `json:"qas"`
}

type SQuADQA struct {
	ID               string        `json:"id"`
	Question         string        `json:"question"`
	Answers          []SQuADAnswer `json:"answers"`
	IsImpossible     bool          `json:"is_impossible,omitempty"`
	PlausibleAnswers []SQuADAnswer `json:"plausible_answers,omitempty"`
}

type SQuADAnswer struct {
	Text        string `json:"text"`
	AnswerStart int    `json:"answer_start"`
}

// PIQA-specific structures for split encoding

// ContextOnly contains only context information (no questions)
type ContextOnly struct {
	Version string               `json:"version"`
	Data    []ContextOnlyArticle `json:"data"`
}

type ContextOnlyArticle struct {
	Title      string                 `json:"title"`
	Paragraphs []ContextOnlyParagraph `json:"paragraphs"`
}

type ContextOnlyParagraph struct {
	Context     string `json:"context"`
	ParagraphID string `json:"paragraph_id"` // format: "Title_Index"
}

// QuestionOnly contains only question information (no context)
type QuestionOnly struct {
	Version   string             `json:"version"`
	Questions []QuestionOnlyItem `json:"questions"`
}

type QuestionOnlyItem struct {
	ID          string        `json:"id"`
	Question    string        `json:"question"`
	Answers     []SQuADAnswer `json:"answers,omitempty"`
	ParagraphID string        `json:"paragraph_id"`
}

// Phrase represents a text span in a document
type Phrase struct {
	Text       string `json:"text"`
	Start      int    `json:"start"`       // character offset in context
	End        int    `json:"end"`         // character offset in context
	StartToken int    `json:"start_token"` // token offset
	EndToken   int    `json:"end_token"`   // token offset
}

// PhraseEmbedding represents a phrase with its vector representation
type PhraseEmbedding struct {
	Phrase    Phrase    `json:"phrase"`
	Embedding []float32 `json:"embedding"`
}

// ContextEmbeddings contains all phrase embeddings for a paragraph
type ContextEmbeddings struct {
	ParagraphID string      `json:"paragraph_id"`
	Phrases     []Phrase    `json:"phrases"`
	Embeddings  [][]float32 `json:"embeddings"` // N x D matrix
}

// QuestionEmbedding contains the embedding for a single question
type QuestionEmbedding struct {
	QuestionID string    `json:"question_id"`
	Embedding  []float32 `json:"embedding"` // 1 x D vector
}

// RetrievalResult represents a retrieved phrase answer
type RetrievalResult struct {
	QuestionID  string  `json:"question_id"`
	PhraseText  string  `json:"phrase_text"`
	Score       float32 `json:"score"`
	ParagraphID string  `json:"paragraph_id"`
	CharStart   int     `json:"char_start"`
	CharEnd     int     `json:"char_end"`
}

// Prediction in SQuAD format
type Prediction struct {
	QuestionID string `json:"question_id"`
	Answer     string `json:"answer"`
}
