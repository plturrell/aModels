package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/cli"
	"github.com/langchain-ai/langgraph-go/pkg/server"
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		usage()
		return
	}

	switch os.Args[1] {
	case "demo":
		if err := handleDemo(os.Args[2:]); err != nil {
			log.Fatalf("demo failed: %v", err)
		}
	case "init":
		if err := handleInit(os.Args[2:]); err != nil {
			log.Fatalf("init failed: %v", err)
		}
	case "run":
		if err := handleRun(os.Args[2:]); err != nil {
			log.Fatalf("run failed: %v", err)
		}
	case "server":
		if err := handleServer(os.Args[2:]); err != nil {
			log.Fatalf("server failed: %v", err)
		}
	case "help", "--help", "-h":
		usage()
	default:
		log.Printf("unknown command %q", os.Args[1])
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stdout, "LangGraph-Go CLI\n\n"+
		"Usage:\n"+
		"  langgraph demo [flags]\n"+
		"  langgraph init [flags]\n"+
		"  langgraph run [flags]\n"+
		"  langgraph server [flags]\n\n"+
		"Flags (demo):\n"+
		"  -input float      Input value for the demo graph (default 1.0)\n"+
		"  -checkpoint value Checkpoint backend: sqlite:/path/to.db | redis://host[:port]/db | hana (default %s)\n"+
		"  -resume           Attempt to resume from existing checkpoints\n"+
		"  -parallel int    Worker pool size for the demo graph\n"+
		"  -events path     Write JSON events to the specified file\n\n"+
		"Flags (init):\n"+
		"  -dir path         Target directory for the project (default .)\n"+
		"  -name string      Project name (optional)\n"+
		"  -checkpoint value Default checkpoint backend for the project (default %s)\n"+
		"  -input float      Default initial input value (default 1.0)\n\n"+
		"Flags (run):\n"+
		"  -project path     Path to the project config (default langgraph.project.json)\n"+
		"  -input float      Override the project's initial input\n"+
		"  -parallel int    Worker pool size (overrides config)\n"+
		"  -events path     Write JSON events to the specified file\n"+
		"  -resume           Attempt to resume from existing checkpoints\n\n"+
		"Flags (server):\n"+
		"  -addr string      Address to listen on (default :8081)\n",
		cli.DefaultDevCheckpoint, cli.DefaultDevCheckpoint)
}

func handleDemo(args []string) error {
	fs := flag.NewFlagSet("demo", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	input := fs.Float64("input", 1.0, "Input value for the demo graph")
	checkpoint := fs.String("checkpoint", cli.DefaultDevCheckpoint, "Checkpoint backend: sqlite:/path/to.db | redis://host[:port]/db | hana")
	resume := fs.Bool("resume", false, "Resume from previous checkpoints")
	parallel := fs.Int("parallel", 0, "Worker pool size for the demo graph")
	mode := fs.String("mode", "", "Execution mode override (async|sync)")
	events := fs.String("events", "", "Write JSON events to the specified file")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	logger := cli.NewLogger("demo")
	var sink *os.File
	if *events != "" {
		var err error
		sink, err = os.OpenFile(*events, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
		defer sink.Close()
	}
	project := &cli.ProjectConfig{
		Name:         "LangGraph Demo",
		Description:  "Sample pipeline executed via the demo command",
		Checkpoint:   *checkpoint,
		InitialInput: *input,
	}
	cli.EnsureProjectDefaults(project)

	runCfg := cli.RunConfig{
		Resume:        *resume,
		Input:         *input,
		OverrideInput: true,
		EventSink:     sink,
	}

	if *parallel > 0 {
		runCfg.OverrideParallelism = *parallel
	}
	if strings.TrimSpace(*mode) != "" {
		runCfg.OverrideMode = *mode
	}
	return cli.ExecuteProject(ctx, project, runCfg, logger)
}

func handleInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	dir := fs.String("dir", ".", "Target directory for the project")
	name := fs.String("name", "", "Project name")
	checkpoint := fs.String("checkpoint", cli.DefaultDevCheckpoint, "Default checkpoint backend (sqlite:/path | redis:// | hana)")
	input := fs.Float64("input", 1.0, "Default initial input value")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	cfg := cli.ProjectConfig{
		Name:         *name,
		Checkpoint:   *checkpoint,
		InitialInput: *input,
	}
	logger := cli.NewLogger("init")
	return cli.InitProject(*dir, cfg, logger)
}

type floatFlag struct {
	value float64
	set   bool
}

func (f *floatFlag) Set(s string) error {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return err
	}
	f.value = v
	f.set = true
	return nil
}

func (f *floatFlag) String() string {
	if f.set {
		return fmt.Sprintf("%f", f.value)
	}
	return ""
}

func handleRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	project := fs.String("project", "langgraph.project.json", "Path to the project config")
	resume := fs.Bool("resume", false, "Resume from existing checkpoints")
	input := &floatFlag{value: math.NaN()}
	fs.Var(input, "input", "Override the project's initial input")
	parallel := fs.Int("parallel", 0, "Worker pool size (overrides config)")
	events := fs.String("events", "", "Write JSON events to the specified file")
	mode := fs.String("mode", "", "Execution mode override (async|sync)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	logger := cli.NewLogger("run")
	var sink *os.File
	if *events != "" {
		var err error
		sink, err = os.OpenFile(*events, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if sink != nil {
		defer sink.Close()
	}

	runCfg := cli.RunConfig{
		ProjectPath:   *project,
		Resume:        *resume,
		Input:         input.value,
		OverrideInput: input.set,
		EventSink:     sink,
	}

	if *parallel > 0 {
		runCfg.OverrideParallelism = *parallel
	}
	if strings.TrimSpace(*mode) != "" {
		runCfg.OverrideMode = *mode
	}

	return cli.RunProject(ctx, runCfg, logger)
}

func handleServer(args []string) error {
	fs := flag.NewFlagSet("server", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	addr := fs.String("addr", ":8081", "Address to listen on")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	logger := cli.NewLogger("server")
	apiKey := os.Getenv("LANGGRAPH_GO_SERVER_API_KEY")
	srv := server.NewServer(*addr, logger, apiKey)
	return srv.Run()
}
