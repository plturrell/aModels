package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/fsnotify/fsnotify"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatalf("Usage: %s <directory> <graph-server-url>", os.Args[0])
	}
	directory := os.Args[1]
	graphServerURL := os.Args[2]

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatal(err)
	}
	defer watcher.Close()

	done := make(chan bool)
	go func() {
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if event.Op&fsnotify.Create == fsnotify.Create {
					log.Printf("New file detected: %s", event.Name)
					triggerGraph(graphServerURL, event.Name)
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Println("error:", err)
			}
		}
	}()

	err = watcher.Add(directory)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Watching directory: %s", directory)
	<-done
}

func triggerGraph(graphServerURL, filePath string) {
	log.Printf("Triggering graph for file: %s", filePath)

	requestBody, err := json.Marshal(map[string]string{
		"file_path": filePath,
	})
	if err != nil {
		log.Printf("Error marshalling request body: %v", err)
		return
	}

	resp, err := http.Post(graphServerURL+"/run", "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		log.Printf("Error triggering graph: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Graph execution failed: %s", resp.Status)
		return
	}

	log.Printf("Graph executed successfully for file: %s", filePath)
}