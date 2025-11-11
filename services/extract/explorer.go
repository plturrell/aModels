package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func (s *extractServer) startExplorer() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Welcome to the Catalog Explorer!")

	for {
		fmt.Print("> ")
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)

		s.handleExplorerCommand(text)
	}
}

func (s *extractServer) handleExplorerCommand(command string) {
	args := strings.Split(command, " ")
	cmd := args[0]

	switch cmd {
	case "help":
		fmt.Println("Available commands:")
		fmt.Println("  projects - list all projects")
		fmt.Println("  systems - list all systems")
		fmt.Println("  isystems - list all information systems")
		fmt.Println("  exit - exit the explorer")
	case "projects":
		s.catalog.mu.RLock()
		defer s.catalog.mu.RUnlock()
		for _, p := range s.catalog.Projects {
			fmt.Printf("- %s (%s)\n", p.Name, p.ID)
		}
	case "systems":
		s.catalog.mu.RLock()
		defer s.catalog.mu.RUnlock()
		for _, sys := range s.catalog.Systems {
			fmt.Printf("- %s (%s)\n", sys.Name, sys.ID)
		}
	case "isystems":
		s.catalog.mu.RLock()
		defer s.catalog.mu.RUnlock()
		for _, is := range s.catalog.InformationSystems {
			fmt.Printf("- %s (%s)\n", is.Name, is.ID)
		}
	case "exit":
		os.Exit(0)
	default:
		fmt.Println("Unknown command. Type 'help' for a list of commands.")
	}
}
