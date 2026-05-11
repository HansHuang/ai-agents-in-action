package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// AgentService
// ---------------------------------------------------------------------------

// AgentServiceConfig holds configuration for the HTTP service.
type AgentServiceConfig struct {
	Host         string
	Port         int
	AgentModel   string
	SystemPrompt string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

// DefaultAgentServiceConfig returns sensible defaults.
func DefaultAgentServiceConfig() AgentServiceConfig {
	port := 8080
	return AgentServiceConfig{
		Host:         "0.0.0.0",
		Port:         port,
		AgentModel:   getEnvStr("AGENT_MODEL", "gpt-4o"),
		SystemPrompt: getEnvStr("SYSTEM_PROMPT", "You are a helpful AI assistant."),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}
}

// AgentService is a minimal HTTP server exposing the agent over REST.
type AgentService struct {
	cfg    AgentServiceConfig
	mux    *http.ServeMux
	server *http.Server
}

// NewAgentService creates a service with the provided config.
func NewAgentService(cfg AgentServiceConfig) *AgentService {
	s := &AgentService{cfg: cfg, mux: http.NewServeMux()}
	s.registerRoutes()
	s.server = &http.Server{
		Addr:         fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler:      s.mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}
	return s
}

// registerRoutes sets up all HTTP handlers.
func (s *AgentService) registerRoutes() {
	s.mux.HandleFunc("/health", s.healthHandler)
	s.mux.HandleFunc("/v1/chat", s.chatHandler)
	s.mux.HandleFunc("/v1/models", s.modelsHandler)
}

// healthHandler returns a JSON health payload.
func (s *AgentService) healthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","model":"%s","ts":"%s"}`,
		s.cfg.AgentModel, time.Now().UTC().Format(time.RFC3339))
}

// chatHandler handles POST /v1/chat requests.
func (s *AgentService) chatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// In production, decode body and call LLM. For demo, echo the path.
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"response":"(demo) echo from %s","model":"%s"}`,
		s.cfg.SystemPrompt[:20], s.cfg.AgentModel)
}

// modelsHandler returns available models.
func (s *AgentService) modelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"models":["%s"]}`, s.cfg.AgentModel)
}

// Addr returns the service's listen address.
func (s *AgentService) Addr() string { return s.server.Addr }

// ListenAndServe starts the HTTP server (blocking).
func (s *AgentService) ListenAndServe() error {
	fmt.Printf("[agent-service] listening on http://%s\n", s.server.Addr)
	return s.server.ListenAndServe()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func getEnvStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// RunAgentServiceDemo demonstrates creating and describing the agent service.
func RunAgentServiceDemo() {
	cfg := DefaultAgentServiceConfig()
	svc := NewAgentService(cfg)

	fmt.Println("Agent Service Configuration:")
	fmt.Printf("  Address : http://%s\n", svc.Addr())
	fmt.Printf("  Model   : %s\n", cfg.AgentModel)
	fmt.Printf("  Prompt  : %s\n", cfg.SystemPrompt[:min09(40, len(cfg.SystemPrompt))])
	fmt.Printf("  Timeout : read=%v write=%v\n", cfg.ReadTimeout, cfg.WriteTimeout)
	fmt.Println()
	fmt.Println("Registered routes:")
	for _, r := range []string{"/health", "/v1/chat", "/v1/models"} {
		fmt.Printf("  %s\n", r)
	}
	fmt.Println()
	fmt.Println("(Demo mode — server not started. Set PORT env var to override port.)")
}

// min09 is a local min helper to avoid conflicts.
func min09(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// truncateStr truncates a string for display.
func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// envVarReport prints env vars useful for deployment.
func envVarReport() string {
	keys := []string{"AGENT_MODEL", "SYSTEM_PROMPT", "OPENAI_API_KEY", "PORT", "LOG_LEVEL"}
	var sb strings.Builder
	for _, k := range keys {
		v := os.Getenv(k)
		if v == "" {
			v = "(not set)"
		} else if strings.Contains(k, "KEY") || strings.Contains(k, "SECRET") {
			v = "***"
		}
		fmt.Fprintf(&sb, "  %-25s = %s\n", k, v)
	}
	return sb.String()
}
