package main
import (\t"context"
"fmt"
"log"
"os"
"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)
func main() {
model := pickModel()
client := llms.NewLocalAI("http://localhost:8080", model)
pt := prompts.NewPromptTemplate("Explain {{.concept}} in {{.style}} style.", []string{"concept", "style"})
pv, _ := pt.FormatPrompt(map[string]any{"concept": "blockchain", "style": "haiku"})
resp, _ := client.GenerateContent(context.TODO(), pv.Messages())
if tv, ok := pv.(llms.TokenAwarePromptValue); ok {
fmt.Println("Token tree:")
fmt.Println(prompts.TokensString(tv.Tokens()))
}
fmt.Println("\nLocalAI response:")
fmt.Println(resp.Choices[0].Content)
}
func pickModel() string {
const dir = "/home/aModels/models"
entries, _ := os.ReadDir(dir)
for _, e := range entries {
name := e.Name()
if !e.IsDir() && (len(name) > 5 && (name[len(name)-5:] == ".gguf" || name[len(name)-12:] == ".safetensors")) {
return name[:len(name)-5]
}
}
return "llama-3-8b"
}
EOF && ./dev.sh go run cmd/localai-use.go