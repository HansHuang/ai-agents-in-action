// framework_advisor.go — Interactive framework advisor for AI agent projects.
//
// Asks up to eight questions about a project and recommends the best
// framework approach with rationale and migration path.
//
// This is pure Go logic — no LLM calls required. The recommendation engine
// uses a rule-based decision tree.
//
// Run:
//
//	go run framework_advisor.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go
//
// See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
package main

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
)

// ---------------------------------------------------------------------------
// Question definitions
// ---------------------------------------------------------------------------

// FWAQuestion describes a single advisor question.
type FWAQuestion struct {
	ID      string
	Text    string
	Options []string // empty → free text or numeric
}

var fwaQuestions = []FWAQuestion{
	{ID: "team_size", Text: "How many developers on your team? (number)"},
	{ID: "ai_experience", Text: "Team AI/LLM experience level?",
		Options: []string{"beginner", "intermediate", "expert"}},
	{ID: "use_case", Text: "Primary use case?",
		Options: []string{"simple_rag", "complex_workflows", "multi_agent", "chatbot", "code_generation", "data_extraction"}},
	{ID: "scale", Text: "Expected scale (requests/day)?",
		Options: []string{"<100", "100-1000", "1000-10000", "10000+"}},
	{ID: "lifetime", Text: "Expected project lifetime?",
		Options: []string{"prototype (weeks)", "medium (months)", "long-term (years)"}},
	{ID: "streaming", Text: "Is real-time streaming to a UI critical? [y/n]"},
	{ID: "multi_provider", Text: "Will you use multiple LLM providers? [y/n]"},
	{ID: "existing_stack", Text: "Primary tech stack?",
		Options: []string{"python", "typescript", "go", "other"}},
}

// ---------------------------------------------------------------------------
// Preset answer sets (for non-interactive demos)
// ---------------------------------------------------------------------------

// FWAPresets maps preset name → answers.
var FWAPresets = map[string]map[string]string{
	"expert": {
		"team_size": "4", "ai_experience": "expert", "use_case": "simple_rag",
		"scale": "10000+", "lifetime": "long-term (years)",
		"streaming": "n", "multi_provider": "y", "existing_stack": "python",
	},
	"beginner": {
		"team_size": "2", "ai_experience": "beginner", "use_case": "simple_rag",
		"scale": "<100", "lifetime": "prototype (weeks)",
		"streaming": "n", "multi_provider": "n", "existing_stack": "python",
	},
	"go_team": {
		"team_size": "3", "ai_experience": "intermediate", "use_case": "simple_rag",
		"scale": "1000-10000", "lifetime": "long-term (years)",
		"streaming": "n", "multi_provider": "n", "existing_stack": "go",
	},
	"multi_agent": {
		"team_size": "2", "ai_experience": "intermediate", "use_case": "multi_agent",
		"scale": "<100", "lifetime": "prototype (weeks)",
		"streaming": "n", "multi_provider": "n", "existing_stack": "python",
	},
	"streaming_ts": {
		"team_size": "3", "ai_experience": "intermediate", "use_case": "chatbot",
		"scale": "100-1000", "lifetime": "medium (months)",
		"streaming": "y", "multi_provider": "y", "existing_stack": "typescript",
	},
}

// ---------------------------------------------------------------------------
// Recommendation result
// ---------------------------------------------------------------------------

// FWARecommendation is the output of FrameworkAdvisor.Recommend.
type FWARecommendation struct {
	Primary             string   // "from_scratch" | "langchain" | "langgraph" | "crewai" | "vercel_ai"
	ForIntegrations     string   // What to use for vector DBs, loaders, etc.
	Avoid               []string // Frameworks to skip
	MigrationPath       string   // Multi-line migration guidance
	Explanation         string   // Narrative reasoning
	ArchitectureDiagram string   // ASCII art
}

// ---------------------------------------------------------------------------
// ASCII architecture diagrams
// ---------------------------------------------------------------------------

var fwaDiagrams = map[string]string{
	"from_scratch": strings.TrimSpace(`
┌────────────────────────────────────────────────────┐
│                FROM-SCRATCH AGENT                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ YOUR CODE (everything)                       │ │
│  │  • Agent orchestration loop                  │ │
│  │  • Tool design and execution                 │ │
│  │  • Context assembly and management           │ │
│  │  • Memory and state                          │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │ THIN HELPERS (optional, replaceable)         │ │
│  │  • openai SDK  (direct API calls)            │ │
│  │  • httpx       (custom HTTP if needed)       │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
Deps: 2-5  |  Debugging: direct stack trace
Best for: production, long-term, expert teams`),

	"langchain": strings.TrimSpace(`
┌────────────────────────────────────────────────────┐
│               LANGCHAIN AGENT                      │
│  ┌──────────────────────────────────────────────┐ │
│  │ LANGCHAIN (chains, retrievers, prompts)      │ │
│  │  • create_retrieval_chain()                  │ │
│  │  • FAISS / Chroma vector store               │ │
│  │  • 700+ document loaders                     │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │ YOUR CODE (customisation)                    │ │
│  │  • Domain-specific prompts                   │ │
│  │  • Business logic around results             │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
Deps: 12-20  |  Debugging: 7-layer stack
Best for: prototypes, RAG + many integrations`),

	"crewai": strings.TrimSpace(`
┌────────────────────────────────────────────────────┐
│                 CREWAI AGENT                       │
│  ┌──────────────────────────────────────────────┐ │
│  │ CREWAI Crew                                  │ │
│  │  • Agent(role="researcher", ...)             │ │
│  │  • Task(description="...", agent=...)        │ │
│  │  • Crew(agents=[...], tasks=[...])           │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │ YOUR CODE                                    │ │
│  │  • Role descriptions and goals               │ │
│  │  • Task definitions and expected outputs     │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
Deps: 8-15  |  Debugging: moderate
Best for: multi-agent prototypes, role-based thinking`),

	"vercel_ai": strings.TrimSpace(`
┌────────────────────────────────────────────────────┐
│             VERCEL AI SDK AGENT                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ VERCEL AI SDK                                │ │
│  │  • streamText() / generateText()             │ │
│  │  • useChat() React hook                      │ │
│  │  • Unified: OpenAI / Anthropic / Google      │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │ YOUR CODE                                    │ │
│  │  • Next.js API routes / app routes           │ │
│  │  • Tool definitions and RAG retrieval logic  │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
Deps: 3-6  |  Debugging: moderate
Best for: full-stack TypeScript, streaming UX`),

	"hybrid": strings.TrimSpace(`
┌────────────────────────────────────────────────────┐
│             HYBRID AGENT (recommended)             │
│  ┌──────────────────────────────────────────────┐ │
│  │ YOUR CODE (the important parts)              │ │
│  │  • Agent orchestration loop                  │ │
│  │  • Context assembly and management           │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │ FRAMEWORK (commodity parts only)             │ │
│  │  • Vector DB connectors  (LangChain)         │ │
│  │  • Document loaders      (LangChain)         │ │
│  │  • Streaming             (Vercel AI SDK)     │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
Rule: Your agent's BRAIN is custom. I/O can use frameworks.`),
}

// ---------------------------------------------------------------------------
// FrameworkAdvisor
// ---------------------------------------------------------------------------

// FrameworkAdvisor maps project answers to a framework recommendation.
type FrameworkAdvisor struct{}

// NewFrameworkAdvisor creates a new FrameworkAdvisor.
func NewFrameworkAdvisor() *FrameworkAdvisor { return &FrameworkAdvisor{} }

// Recommend derives a recommendation from the provided answers.
// Rules are evaluated in priority order; the first match wins.
func (a *FrameworkAdvisor) Recommend(answers map[string]string) FWARecommendation {
	stack := answers["existing_stack"]
	experience := answers["ai_experience"]
	useCase := answers["use_case"]
	lifetime := answers["lifetime"]
	streaming := answers["streaming"] == "y" || answers["streaming"] == "yes"
	multiProvider := answers["multi_provider"] == "y" || answers["multi_provider"] == "yes"
	scale := answers["scale"]

	isLongTerm := strings.HasPrefix(lifetime, "long-term")
	isPrototype := strings.HasPrefix(lifetime, "prototype")
	isExpert := experience == "expert"
	isBeginner := experience == "beginner"
	isGo := stack == "go"
	isTS := stack == "typescript"

	switch {
	case isGo:
		return a.recGo(answers)
	case isTS && (streaming || multiProvider):
		return a.recVercelAI(answers)
	case isTS:
		return a.recVercelAI(answers)
	case useCase == "multi_agent" && isPrototype:
		return a.recCrewAI(answers)
	case useCase == "complex_workflows" && !isPrototype:
		return a.recLangGraph(answers)
	case isExpert && isLongTerm:
		return a.recFromScratch(answers)
	case isBeginner && isPrototype:
		return a.recLangChain(answers)
	case scale == "10000+" && isLongTerm:
		return a.recFromScratch(answers)
	default:
		return a.recHybrid(answers)
	}
}

func (a *FrameworkAdvisor) recGo(ans map[string]string) FWARecommendation {
	return FWARecommendation{
		Primary:         "from_scratch",
		ForIntegrations: "Qdrant Go SDK or Weaviate Go client for vector search",
		Avoid:           []string{"langchain (no mature Go port)", "crewai (Python only)", "langgraph (Python only)"},
		MigrationPath: "Phase 1: Build agent loop in Go with direct OpenAI SDK.\n" +
			"Phase 2: Add Qdrant or Weaviate Go client for vector search.\n" +
			"Phase 3: Use pgvector with database/sql if you already run Postgres.\n" +
			"Phase 4: Consider LangChain4j if you move to the JVM ecosystem.",
		Explanation: "Go has no mature, production-ready AI agent framework equivalent to LangChain. " +
			"The OpenAI Go SDK is solid. For vector search, Qdrant and Weaviate have official Go clients. " +
			"Build the agent loop as custom Go code — you get idiomatic concurrency, strong typing, " +
			"and fast compilation. This is actually an advantage: Go teams often produce cleaner, " +
			"more maintainable agent code than Python teams using heavy frameworks.",
		ArchitectureDiagram: fwaDiagrams["from_scratch"],
	}
}

func (a *FrameworkAdvisor) recFromScratch(ans map[string]string) FWARecommendation {
	return FWARecommendation{
		Primary:         "from_scratch",
		ForIntegrations: "langchain (for vector DB connectors only)",
		Avoid:           []string{"crewai (premature)", "full langchain (over-engineered)"},
		MigrationPath: "Phase 1: Build core agent from scratch — done.\n" +
			"Phase 2: Add LangChain for specific integrations (e.g. new vector DB).\n" +
			"Phase 3: Add LangSmith / Arize for observability.\n" +
			"Phase 4: Consider LangGraph only if workflow branching becomes complex.",
		Explanation: fmt.Sprintf("Your team is expert-level with a long-term production commitment at %s scale. "+
			"The highest ROI is full control: you debug with plain stack traces, upgrade dependencies "+
			"on your own schedule, and keep the dependency footprint minimal. "+
			"Use LangChain only for vector DB connectors — a small, isolated surface.", ans["scale"]),
		ArchitectureDiagram: fwaDiagrams["hybrid"],
	}
}

func (a *FrameworkAdvisor) recLangChain(ans map[string]string) FWARecommendation {
	return FWARecommendation{
		Primary:         "langchain",
		ForIntegrations: "langchain (already included)",
		Avoid:           []string{"langgraph (too complex for prototype)", "from_scratch (too slow to start)"},
		MigrationPath: "Phase 1: Use LangChain chains for rapid prototyping.\n" +
			"Phase 2: Extract agent loop to custom code as requirements clarify.\n" +
			"Phase 3: Keep LangChain for connectors; build custom orchestration.\n" +
			"Phase 4: Replace problematic LangChain components one at a time.",
		Explanation: "Your team is getting started with LLMs and needs a working prototype quickly. " +
			"LangChain's pre-built chains reduce the learning curve and time-to-demo. " +
			"The trade-off — harder debugging and more dependencies — is acceptable at prototype stage. " +
			"Plan to extract the agent loop to custom code as the project matures.",
		ArchitectureDiagram: fwaDiagrams["langchain"],
	}
}

func (a *FrameworkAdvisor) recLangGraph(ans map[string]string) FWARecommendation {
	return FWARecommendation{
		Primary:         "langgraph",
		ForIntegrations: "langchain (for vector DB and document loaders)",
		Avoid:           []string{"crewai (lacks fine-grained control)", "full custom (StateGraph saves significant effort)"},
		MigrationPath: "Phase 1: Build StateGraph with LangGraph for workflow orchestration.\n" +
			"Phase 2: Implement individual nodes as custom Python functions.\n" +
			"Phase 3: Add checkpointing for persistence across steps.\n" +
			"Phase 4: Extract nodes that become problematic to custom code.",
		Explanation: fmt.Sprintf("Your use case is %s and you're building for %s use at %s scale. "+
			"LangGraph's StateGraph maps directly to branching agent logic, provides per-node state inspection, "+
			"and has built-in checkpointing. Individual nodes are plain Python functions, "+
			"so your business logic stays framework-independent.", ans["use_case"], ans["lifetime"], ans["scale"]),
		ArchitectureDiagram: fwaDiagrams["langchain"], // LangGraph builds on LangChain
	}
}

func (a *FrameworkAdvisor) recCrewAI(ans map[string]string) FWARecommendation {
	return FWARecommendation{
		Primary:         "crewai",
		ForIntegrations: "langchain (for document loading and vector search)",
		Avoid:           []string{"langgraph (overkill for prototype)", "full custom (too slow for prototype)"},
		MigrationPath: "Phase 1: Use CrewAI to explore multi-agent patterns quickly.\n" +
			"Phase 2: Identify which agents/tasks need custom logic.\n" +
			"Phase 3: Replace CrewAI orchestration with custom code for those.\n" +
			"Phase 4: Evaluate LangGraph if state management becomes painful.",
		Explanation: "You need multi-agent collaboration and you're in prototype mode. " +
			"CrewAI's role-based mental model (Agent + Task + Crew) is the fastest " +
			"path to a working multi-agent demo. Expect to outgrow it as you need " +
			"fine-grained control over agent communication.",
		ArchitectureDiagram: fwaDiagrams["crewai"],
	}
}

func (a *FrameworkAdvisor) recVercelAI(ans map[string]string) FWARecommendation {
	streamNote := ""
	if ans["streaming"] == "y" {
		streamNote = " Streaming is critical for your UX,"
	}
	mpNote := ""
	if ans["multi_provider"] == "y" {
		mpNote = " and you need multi-provider flexibility,"
	}
	return FWARecommendation{
		Primary:         "vercel_ai",
		ForIntegrations: "langchain.js (for document loading) or custom",
		Avoid:           []string{"python langchain (wrong language)", "crewai (no TypeScript support)"},
		MigrationPath: "Phase 1: Use Vercel AI SDK for streaming and provider abstraction.\n" +
			"Phase 2: Add RAG with LangChain.js or a direct vector DB client.\n" +
			"Phase 3: Build agent loop as custom TypeScript functions.\n" +
			"Phase 4: Extract provider-specific code behind your own interface.",
		Explanation: fmt.Sprintf("You're building in TypeScript.%s%s which is exactly what "+
			"Vercel AI SDK is designed for. Its unified interface across OpenAI, Anthropic, and Google "+
			"means you can switch providers without touching your application logic.", streamNote, mpNote),
		ArchitectureDiagram: fwaDiagrams["vercel_ai"],
	}
}

func (a *FrameworkAdvisor) recHybrid(ans map[string]string) FWARecommendation {
	return FWARecommendation{
		Primary:         "from_scratch",
		ForIntegrations: "langchain (for commodity connectors only)",
		Avoid:           []string{"full langchain ownership of agent logic"},
		MigrationPath: "Phase 1: Build agent loop from scratch — learn the concepts.\n" +
			"Phase 2: Add LangChain for specific connectors you need.\n" +
			"Phase 3: Evaluate LangGraph if workflow complexity grows.\n" +
			"Phase 4: Never let a framework own your agent's brain.",
		Explanation: "Your project sits in the middle of the decision matrix. The hybrid approach " +
			"gives you the best of both worlds: use LangChain for the commodity parts " +
			"(vector DB connectors, document loaders) and write custom code for the " +
			"parts that differentiate your agent.",
		ArchitectureDiagram: fwaDiagrams["hybrid"],
	}
}

// Explain formats a recommendation for printing.
func (a *FrameworkAdvisor) Explain(rec FWARecommendation) string {
	var sb strings.Builder
	sb.WriteString("\n" + strings.Repeat("=", 65) + "\n")
	sb.WriteString("  FRAMEWORK RECOMMENDATION\n")
	sb.WriteString(strings.Repeat("=", 65) + "\n\n")
	sb.WriteString(fmt.Sprintf("  PRIMARY:            %s\n", strings.ToUpper(strings.ReplaceAll(rec.Primary, "_", " "))))
	sb.WriteString(fmt.Sprintf("  FOR INTEGRATIONS:   %s\n", rec.ForIntegrations))
	sb.WriteString(fmt.Sprintf("  AVOID:              %s\n\n", strings.Join(rec.Avoid, ", ")))
	sb.WriteString("  REASONING:\n")
	for _, line := range strings.Split(rec.Explanation, ". ") {
		if line != "" {
			sb.WriteString(fmt.Sprintf("    %s.\n", strings.TrimRight(line, ".")))
		}
	}
	sb.WriteString("\n  MIGRATION PATH:\n")
	for _, step := range strings.Split(rec.MigrationPath, "\n") {
		sb.WriteString(fmt.Sprintf("    %s\n", step))
	}
	sb.WriteString("\n  ARCHITECTURE:\n\n")
	for _, line := range strings.Split(rec.ArchitectureDiagram, "\n") {
		sb.WriteString(fmt.Sprintf("  %s\n", line))
	}
	sb.WriteString(strings.Repeat("=", 65) + "\n")
	return sb.String()
}

// ---------------------------------------------------------------------------
// Demo entry point (no main())
// ---------------------------------------------------------------------------

// runFrameworkAdvisorDemo runs all presets and prints recommendations.
func runFrameworkAdvisorDemo() {
	advisor := NewFrameworkAdvisor()

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "\n%-20s\t%-15s\t%-30s\n", "PRESET", "PRIMARY", "AVOID")
	fmt.Fprintf(w, "%s\t%s\t%s\n", strings.Repeat("-", 20), strings.Repeat("-", 15), strings.Repeat("-", 30))

	for name, answers := range FWAPresets {
		rec := advisor.Recommend(answers)
		avoidStr := strings.Join(rec.Avoid, "; ")
		if len(avoidStr) > 28 {
			avoidStr = avoidStr[:28] + "…"
		}
		fmt.Fprintf(w, "%-20s\t%-15s\t%-30s\n", name, rec.Primary, avoidStr)
	}
	_ = w.Flush()

	// Print full recommendation for "go_team"
	goAnswers := FWAPresets["go_team"]
	rec := advisor.Recommend(goAnswers)
	fmt.Println(advisor.Explain(rec))
}
