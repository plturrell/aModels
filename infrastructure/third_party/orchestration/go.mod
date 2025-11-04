module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration

go 1.24.4

toolchain go1.24.6

// Note: Thanks to Go's module graph pruning (https://go.dev/ref/mod#graph-pruning),
// importing orchestration does NOT pull in all dependencies listed below. You only
// get dependencies for the specific packages you import. For example:
//   - import ".../agenticAiETH_layer4_Orchestration/llms/localai" → only LocalAI deps
//   - import ".../agenticAiETH_layer4_Orchestration/chains" → only chain deps
// This keeps your builds lean despite this large go.mod file.

// Core dependencies
require github.com/google/uuid v1.6.0 // indirect

// Testing
require github.com/stretchr/testify v1.11.1

// LLM providers
require (
	github.com/cohere-ai/tokenizer v1.1.2
	github.com/pkoukk/tiktoken-go v0.1.6
)

// Vector stores
require go.mongodb.org/mongo-driver v1.14.0

// Cloud platforms and AI services
require (
	golang.org/x/oauth2 v0.30.0
	google.golang.org/api v0.246.0
	google.golang.org/grpc v1.75.0 // indirect
	google.golang.org/protobuf v1.36.10 // indirect
)

// Embeddings and ML
require github.com/AssemblyAI/assemblyai-go-sdk v1.3.0

// Database drivers
require (
	cloud.google.com/go/alloydbconn v1.13.2
	cloud.google.com/go/cloudsqlconn v1.14.1
	github.com/jackc/pgx/v5 v5.7.2
	github.com/mattn/go-sqlite3 v1.14.32
)

// Document processing and web scraping
require (
	github.com/PuerkitoBio/goquery v1.8.1
	github.com/ledongthuc/pdf v0.0.0-20220302134840-0c2507a12d80
	github.com/microcosm-cc/bluemonday v1.0.26
	gitlab.com/golang-commonmark/markdown v0.0.0-20211110145824-bf3e522c626a
)

// Memory and agent tools
require go.starlark.net v0.0.0-20230302034142-4b1e35fe2254

// Utilities and helpers
require (
	github.com/Code-Hex/go-generics-cache v1.3.1
	github.com/Masterminds/sprig/v3 v3.2.3
	github.com/google/go-cmp v0.7.0
	github.com/nikolalohinski/gonja v1.5.3
	golang.org/x/sync v0.17.0 // indirect
	sigs.k8s.io/yaml v1.3.0
)

// Indirect dependencies (automatically managed)

// Cloud platforms and AI services - indirect
require (
	cloud.google.com/go v0.121.4 // indirect
	cloud.google.com/go/alloydb v1.14.0 // indirect
	cloud.google.com/go/auth v0.16.3 // indirect
	cloud.google.com/go/auth/oauth2adapt v0.2.8 // indirect
	cloud.google.com/go/compute/metadata v0.7.0 // indirect
	cloud.google.com/go/longrunning v0.6.7 // indirect
	github.com/google/s2a-go v0.1.9 // indirect
	github.com/googleapis/enterprise-certificate-proxy v0.3.6 // indirect
	github.com/googleapis/gax-go/v2 v2.15.0 // indirect
	go.opencensus.io v0.24.0 // indirect
	go.opentelemetry.io/auto/sdk v1.1.0 // indirect
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.61.0 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.61.0 // indirect
	go.opentelemetry.io/otel v1.38.0 // indirect
	go.opentelemetry.io/otel/metric v1.38.0 // indirect
	go.opentelemetry.io/otel/trace v1.38.0 // indirect
	google.golang.org/genproto v0.0.0-20250603155806-513f23925822 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20250825161204-c5933d9347a5 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)

// Vector stores - indirect
require (
	github.com/montanaflynn/stats v0.7.0 // indirect
	github.com/xdg-go/pbkdf2 v1.0.0 // indirect
	github.com/xdg-go/scram v1.1.2 // indirect
	github.com/xdg-go/stringprep v1.0.4 // indirect
	github.com/youmark/pkcs8 v0.0.0-20240726163527-a2c0da244d78 // indirect
	nhooyr.io/websocket v1.8.7 // indirect
)

// Database drivers - indirect
require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
)

// Document processing and web scraping - indirect
require (
	github.com/andybalholm/cascadia v1.3.2 // indirect
	github.com/gorilla/css v1.0.0 // indirect
	gitlab.com/golang-commonmark/html v0.0.0-20191124015941-a22733972181 // indirect
	gitlab.com/golang-commonmark/linkify v0.0.0-20191026162114-a0c2df6c8f82 // indirect
	gitlab.com/golang-commonmark/mdurl v0.0.0-20191124015652-932350d1cb84 // indirect
	gitlab.com/golang-commonmark/puny v0.0.0-20191124015043-9f83538fa04f // indirect
)

// Testing infrastructure - indirect
require (
	github.com/cenkalti/backoff v2.2.1+incompatible // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
)

// Utilities and common libraries - indirect
require (
	github.com/Masterminds/goutils v1.1.1 // indirect
	github.com/Masterminds/semver/v3 v3.2.0 // indirect
	github.com/aymerick/douceur v0.2.0 // indirect
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da // indirect
	github.com/golang/snappy v1.0.0 // indirect
	github.com/google/go-querystring v1.1.0 // indirect
	github.com/goph/emperror v0.17.2 // indirect
	github.com/huandu/xstrings v1.3.3 // indirect
	github.com/imdario/mergo v0.3.13 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/klauspost/compress v1.18.0 // indirect
	github.com/klauspost/cpuid/v2 v2.2.9 // indirect
	github.com/mattn/go-colorable v0.1.14 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mitchellh/copystructure v1.0.0 // indirect
	github.com/mitchellh/reflectwalk v1.0.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/pelletier/go-toml/v2 v2.2.4 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/shopspring/decimal v1.3.1 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	github.com/spf13/cast v1.10.0 // indirect
	github.com/yargevad/filepathx v1.0.0 // indirect
	golang.org/x/crypto v0.42.0 // indirect
	golang.org/x/exp v0.0.0-20250620022241-b7579e27df2b // indirect
	golang.org/x/net v0.44.0 // indirect
	golang.org/x/sys v0.36.0 // indirect
	golang.org/x/text v0.30.0 // indirect
	golang.org/x/time v0.14.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

require (
	github.com/bytedance/sonic v1.10.0-rc3 // indirect
	github.com/gin-gonic/gin v1.9.1 // indirect
	github.com/go-playground/validator/v10 v10.14.1 // indirect
	github.com/gobwas/ws v1.3.0 // indirect
	github.com/goccy/go-json v0.10.4 // indirect
	github.com/gorilla/websocket v1.5.3 // indirect
	github.com/stretchr/objx v0.5.2 // indirect
	go.opentelemetry.io/otel/sdk v1.38.0 // indirect
	go.opentelemetry.io/otel/sdk/metric v1.38.0 // indirect
	golang.org/x/arch v0.4.0 // indirect
)
