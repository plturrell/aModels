package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"time"
)

// Logger is a minimal structured logger used by CLI commands.
type Logger struct {
	*log.Logger
	eventSink io.Writer
}

// NewLogger constructs a logger that writes to stdout with a standard prefix.
func NewLogger(prefix string) Logger {
	return Logger{Logger: log.New(os.Stdout, fmt.Sprintf("[%s] ", prefix), log.LstdFlags)}
}

// WithEventSink returns a copy of the logger that also writes structured events
// to the provided writer.
func (l Logger) WithEventSink(w io.Writer) Logger {
	l.eventSink = w
	return l
}

// Event emits a structured telemetry record alongside the standard log output.
func (l Logger) Event(name string, fields map[string]any) {
	payload := map[string]any{
		"event": name,
		"ts":    time.Now().UTC().Format(time.RFC3339Nano),
	}
	for k, v := range fields {
		payload[k] = v
	}
	data, err := json.Marshal(payload)
	if err != nil {
		if l.Logger != nil {
			l.Println("event", name, "marshal error:", err)
		}
		return
	}
	if l.Logger != nil {
		l.Println(string(data))
	}
	if l.eventSink != nil {
		_, _ = l.eventSink.Write(append(data, '\n'))
	}
}
