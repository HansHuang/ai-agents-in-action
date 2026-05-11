// over_engineering_detector.go — Multi-agent over-engineering detector.
//
// Analyzes a project description and warns when a multi-agent architecture
// is likely overkill — adding cost, complexity, and latency without benefit.
//
// Two stages:
//  1. Rule-based pattern matching against known over-engineering signals
//  2. Optional LLM-as-judge for nuanced cases (requires OPENAI_API_KEY)
//
// Run:
//
//	go run over_engineering_detector.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go
//
// See: docs/06-frameworks-in-practice/03-crewai-autogen.md
package main

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Cost model constants
// ---------------------------------------------------------------------------

const (
	oeTokensPerAgentPerCall = 3_000 // average tokens per agent round-trip
	oeCallsPerAgentPerDay   = 200   // average daily queries
	oeCostPer1KTokens       = 0.005 // blended input+output rate (USD)
	oeDaysPerMonth          = 30
)

// ---------------------------------------------------------------------------
// Report type
// ---------------------------------------------------------------------------

// OEReport is the full analysis of a proposed multi-agent design.
type OEReport struct {
	RiskLevel                 string // "low" | "medium" | "high"
	AgentCount                int    // Proposed agent count extracted from description
	SuggestedAgentCount       int    // What the detector recommends
	Warnings                  []string
	SimplificationSuggestions []string
	CostSavingsEstimateUSD    float64 // Monthly USD saved by simplifying
	LLMVerdict                string  // Set when LLM-as-judge is used
}

// ---------------------------------------------------------------------------
// OverEngineeringDetector
// ---------------------------------------------------------------------------

// OverEngineeringDetector detects when a project is likely over-engineered.
type OverEngineeringDetector struct {
	UseLLMJudge bool
}

// NewOverEngineeringDetector creates a detector.
// Set useLLMJudge=true to get an LLM second opinion (requires OPENAI_API_KEY).
func NewOverEngineeringDetector(useLLMJudge bool) *OverEngineeringDetector {
	return &OverEngineeringDetector{UseLLMJudge: useLLMJudge}
}

// warningPatterns maps signal key → user-facing warning message.
var oeWarningPatterns = map[string]string{
	"faq_bot": "An FAQ or Q&A bot doesn't need multiple agents. " +
		"One agent with RAG retrieval handles this completely.",
	"crud_app": "CRUD operations don't benefit from agent collaboration. " +
		"Use deterministic code with tools, not agents.",
	"simple_workflow": "A strictly linear workflow doesn't justify multi-agent. " +
		"A single Plan-and-Execute agent is simpler and cheaper.",
	"single_domain": "All tasks appear to be in the same knowledge domain. " +
		"Specialized agents add overhead without adding specialization value.",
	"agents_as_tools": "Some 'agents' here do one thing and return a result. " +
		"Those are tools, not agents. Fold them into a single agent's tool list.",
	"tightly_coupled_pipeline": "Every agent depends on the previous one's full output. " +
		"This is a sequential pipeline — a single agent with structured steps does the same with less overhead.",
	"real_time_requirement": "Real-time requirements conflict with multi-agent coordination. " +
		"Each agent-to-agent handoff adds 1–5 seconds of latency.",
	"simple_classification": "Classification or routing tasks don't need multiple agents. " +
		"A single LLM call with a structured output schema handles this.",
	"single_user_turn": "If the system is stateless (one user turn, one reply), " +
		"multi-agent adds infrastructure complexity for no conversational benefit.",
}

// signalKeywords maps each signal key to keywords that trigger it.
var oeSignalKeywords = map[string][]string{
	"faq_bot":                  {"faq", "frequently asked", "q&a", "q & a", "knowledge base answer"},
	"crud_app":                 {"crud", "create update delete", "database", "form", "record management"},
	"simple_workflow":          {"linear", "step by step", "step-by-step", "one after another", "sequential pipeline", "waterfall"},
	"agents_as_tools":          {"returns a result", "calls an api", "fetch", "lookup", "query the"},
	"tightly_coupled_pipeline": {"each agent", "previous agent", "passes to the next", "chain of agents", "pipeline of agents"},
	"real_time_requirement":    {"real-time", "real time", "low latency", "sub-second", "millisecond", "streaming response"},
	"simple_classification":    {"classify", "route", "intent detection", "categoris", "categoriz", "label"},
	"single_user_turn":         {"one-shot", "single turn", "stateless", "no conversation", "no history", "no memory"},
}

// justificationPatterns are multi-word patterns that JUSTIFY multi-agent use.
var oeJustificationPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)adversarial`),
	regexp.MustCompile(`(?i)debate`),
	regexp.MustCompile(`(?i)multiple domain`),
	regexp.MustCompile(`(?i)different expertise`),
	regexp.MustCompile(`(?i)parallel`),
	regexp.MustCompile(`(?i)independent sub.task`),
	regexp.MustCompile(`(?i)human in the loop`),
	regexp.MustCompile(`(?i)critique`),
	regexp.MustCompile(`(?i)quality check`),
	regexp.MustCompile(`(?i)role specializ`),
	regexp.MustCompile(`(?i)long.running`),
	regexp.MustCompile(`(?i)autonomous`),
}

// agentCountPattern extracts numeric agent count from description text.
var oeAgentCountRE = regexp.MustCompile(`(?i)(\d+)[\s\-]agent|(\d+)\s+agents?|team\s+of\s+(\d+)`)
var oeWordAgentRE = regexp.MustCompile(`(?i)(one|two|three|four|five|six|seven|eight|nine|ten)\s+agents?`)

var oeWordMap = map[string]int{
	"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
	"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

func oeExtractAgentCount(description string) int {
	if m := oeAgentCountRE.FindStringSubmatch(description); m != nil {
		for _, g := range m[1:] {
			if g != "" {
				n := 0
				fmt.Sscanf(g, "%d", &n)
				if n > 0 {
					return n
				}
			}
		}
	}
	if m := oeWordAgentRE.FindStringSubmatch(description); m != nil && len(m) > 1 {
		if n, ok := oeWordMap[strings.ToLower(m[1])]; ok {
			return n
		}
	}
	return 0
}

func oeCheckSignal(description, key string) bool {
	lower := strings.ToLower(description)
	keywords, ok := oeSignalKeywords[key]
	if !ok {
		return false
	}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func oeCheckSingleDomain(description string) bool {
	domains := map[string][]string{
		"customer_support": {"support", "helpdesk", "ticket", "complaint", "inquiry"},
		"data_entry":       {"fill", "form", "entry", "transcrib", "extract fields"},
		"simple_qa":        {"answer", "question", "faq", "lookup"},
		"notification":     {"notify", "alert", "send email", "send message", "push notification"},
	}
	lower := strings.ToLower(description)
	for _, keywords := range domains {
		count := 0
		for _, kw := range keywords {
			if strings.Contains(lower, kw) {
				count++
			}
		}
		if count >= 2 {
			return true
		}
	}
	return false
}

func oeCountJustifications(description string) int {
	count := 0
	for _, re := range oeJustificationPatterns {
		if re.MatchString(description) {
			count++
		}
	}
	return count
}

// Analyze returns an OEReport for the given project description.
func (d *OverEngineeringDetector) Analyze(ctx context.Context, client *openai.Client, description string) OEReport {
	var warnings []string

	// Check each signal
	for key, message := range oeWarningPatterns {
		if key == "single_domain" {
			if oeCheckSingleDomain(description) {
				warnings = append(warnings, message)
			}
		} else {
			if oeCheckSignal(description, key) {
				warnings = append(warnings, message)
			}
		}
	}

	proposed := oeExtractAgentCount(description)
	justifications := oeCountJustifications(description)
	maxJustified := 2 + justifications

	if proposed > 0 && proposed > maxJustified {
		warnings = append(warnings, fmt.Sprintf(
			"The design proposes %d agents but the described task appears to justify at most %d. "+
				"Each extra agent adds token overhead and latency.", proposed, maxJustified))
	}

	// Determine risk level
	nWarnings := len(warnings)
	riskLevel := "low"
	switch {
	case nWarnings == 0:
		riskLevel = "low"
	case nWarnings <= 2:
		riskLevel = "medium"
	default:
		riskLevel = "high"
	}
	if justifications >= 3 && riskLevel == "high" {
		riskLevel = "medium"
	}
	if justifications >= 5 {
		riskLevel = "low"
	}

	// Suggested agent count
	suggested := 1
	if proposed == 0 {
		suggested = 1
	} else if riskLevel == "low" {
		suggested = proposed
	} else if riskLevel == "medium" {
		if proposed-1 > 1 {
			suggested = proposed - 1
		} else {
			suggested = 1
		}
	} else {
		suggested = proposed / 2
		if suggested < 1 {
			suggested = 1
		}
		if suggested > 2 {
			suggested = 2
		}
	}

	suggestions := d.buildSuggestions(description, warnings, proposed, suggested)
	costSavings := d.estimateCostSavings(proposed, suggested)

	report := OEReport{
		RiskLevel:                 riskLevel,
		AgentCount:                proposed,
		SuggestedAgentCount:       suggested,
		Warnings:                  warnings,
		SimplificationSuggestions: suggestions,
		CostSavingsEstimateUSD:    costSavings,
	}

	if d.UseLLMJudge && client != nil {
		report.LLMVerdict = d.llmJudge(ctx, client, description, report)
	}

	return report
}

func (d *OverEngineeringDetector) buildSuggestions(description string, warnings []string, proposed, suggested int) []string {
	var suggestions []string

	joinedWarnings := strings.Join(warnings, " ")

	if strings.Contains(joinedWarnings, "FAQ") || strings.Contains(joinedWarnings, "Q&A") {
		suggestions = append(suggestions, "Replace the multi-agent system with a single agent + RAG pipeline. "+
			"Use a vector database to retrieve relevant docs and a single LLM call to answer.")
	}
	if strings.Contains(joinedWarnings, "CRUD") {
		suggestions = append(suggestions, "Replace agents with deterministic code. Use function tools for "+
			"database operations; reserve the LLM for natural-language interpretation only.")
	}
	if strings.Contains(strings.ToLower(joinedWarnings), "linear") || strings.Contains(strings.ToLower(joinedWarnings), "pipeline") {
		suggestions = append(suggestions, "Use a single Plan-and-Execute agent instead of a chain of agents. "+
			"The agent plans the steps internally and executes them with tools.")
	}
	if strings.Contains(strings.ToLower(joinedWarnings), "tool") {
		suggestions = append(suggestions, "Convert single-purpose agents into tools. A tool is a Go function "+
			"with a description — simpler, faster, and cheaper than a full agent.")
	}
	if strings.Contains(strings.ToLower(joinedWarnings), "real-time") || strings.Contains(strings.ToLower(joinedWarnings), "latency") {
		suggestions = append(suggestions, "For real-time requirements, pre-compute results or use a single fast "+
			"LLM call. Multi-agent coordination typically adds 3–10 seconds of latency.")
	}
	if proposed > 0 && suggested < proposed {
		savingsPct := int((1 - float64(suggested)/float64(proposed)) * 100)
		suggestions = append(suggestions, fmt.Sprintf(
			"Reducing from %d to %d agents saves approximately %d%% of token usage "+
				"and improves reliability (fewer failure points).", proposed, suggested, savingsPct))
	}
	if len(suggestions) == 0 {
		suggestions = append(suggestions, "The design appears reasonable. Proceed with the multi-agent approach, "+
			"but monitor token costs closely in the first two weeks.")
	}
	return suggestions
}

func (d *OverEngineeringDetector) estimateCostSavings(proposed, suggested int) float64 {
	if proposed <= 0 || suggested >= proposed {
		return 0
	}
	proposedTokens := float64(proposed) * oeTokensPerAgentPerCall * oeCallsPerAgentPerDay * oeDaysPerMonth
	suggestedTokens := float64(suggested) * oeTokensPerAgentPerCall * oeCallsPerAgentPerDay * oeDaysPerMonth
	savings := (proposedTokens - suggestedTokens) / 1000 * oeCostPer1KTokens
	// Round to 2 decimal places
	return float64(int(savings*100+0.5)) / 100
}

func (d *OverEngineeringDetector) llmJudge(ctx context.Context, client *openai.Client, description string, report OEReport) string {
	warningsText := strings.Join(report.Warnings, "\n- ")
	if warningsText == "" {
		warningsText = "None"
	}
	prompt := fmt.Sprintf(
		"A developer proposes this multi-agent system:\n\n%s\n\n"+
			"A rule-based checker raised these concerns:\n- %s\n\n"+
			"In one paragraph (max 100 words), give a balanced verdict: "+
			"Is this system over-engineered? What is the key risk? "+
			"What is the simplest version that achieves the same goal?",
		description, warningsText)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: "gpt-4o",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "You are a senior AI systems architect. Give concise, practical advice."},
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		MaxTokens: 200,
	})
	if err != nil {
		return "(LLM judge unavailable)"
	}
	return resp.Choices[0].Message.Content
}

// SuggestSimplification formats the report as a readable string.
func (d *OverEngineeringDetector) SuggestSimplification(report OEReport) string {
	var sb strings.Builder
	sb.WriteString("OVER-ENGINEERING ASSESSMENT\n")
	sb.WriteString(strings.Repeat("─", 40) + "\n")
	sb.WriteString(fmt.Sprintf("Risk level       : %s\n", strings.ToUpper(report.RiskLevel)))
	if report.AgentCount > 0 {
		sb.WriteString(fmt.Sprintf("Proposed agents  : %d\n", report.AgentCount))
	} else {
		sb.WriteString("Proposed agents  : unknown\n")
	}
	sb.WriteString(fmt.Sprintf("Suggested agents : %d\n", report.SuggestedAgentCount))
	sb.WriteString(fmt.Sprintf("Monthly savings  : $%.2f (est.)\n", report.CostSavingsEstimateUSD))

	if len(report.Warnings) > 0 {
		sb.WriteString("\nWARNINGS:\n")
		for _, w := range report.Warnings {
			sb.WriteString(fmt.Sprintf("  ⚠  %s\n", w))
		}
	}

	if len(report.SimplificationSuggestions) > 0 {
		sb.WriteString("\nSIMPLIFICATION SUGGESTIONS:\n")
		for i, s := range report.SimplificationSuggestions {
			sb.WriteString(fmt.Sprintf("  %d. %s\n", i+1, s))
		}
	}

	if report.LLMVerdict != "" {
		sb.WriteString("\nLLM VERDICT:\n")
		sb.WriteString(fmt.Sprintf("  %s\n", report.LLMVerdict))
	}

	return sb.String()
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runOverEngineeringDetectorDemo runs the demo (no main()).
func runOverEngineeringDetectorDemo(ctx context.Context, client *openai.Client) {
	detector := NewOverEngineeringDetector(client != nil)

	testCases := []struct {
		description string
		label       string
	}{
		{
			description: "5-agent system for a customer support chatbot that answers FAQs about our product. " +
				"Agents: Greeter, IntentClassifier, KnowledgeRetriever, ResponseWriter, ResponseReviewer.",
			label: "Customer support FAQ bot",
		},
		{
			description: "A 3-agent research crew: Researcher (web search + database), " +
				"FactChecker (verify sources), Writer (produce structured report). " +
				"Used for daily competitive intelligence reports with adversarial review.",
			label: "Research + analysis + writing team",
		},
		{
			description: "A 4-agent pipeline to fill out insurance claim forms: " +
				"DataExtractor, FormFiller, ValidatorAgent, SubmitterAgent. " +
				"Linear workflow, one-shot per claim, no critique needed.",
			label: "Insurance form filling pipeline",
		},
		{
			description: "An 8-agent content marketing system: IdeaGenerator, TopicResearcher, " +
				"OutlineWriter, DraftWriter, EditorAgent, SEOOptimizer, ImagePromptAgent, SchedulerAgent. " +
				"Each depends on the previous agent's output.",
			label: "Content marketing pipeline",
		},
	}

	for _, tc := range testCases {
		fmt.Printf("\n%s\n  TEST CASE: %s\n%s\n", strings.Repeat("═", 60), tc.label, strings.Repeat("═", 60))
		descPreview := tc.description
		if len(descPreview) > 120 {
			descPreview = descPreview[:120] + "…"
		}
		fmt.Printf("\nDescription:\n  %s\n\n", descPreview)
		report := detector.Analyze(ctx, client, tc.description)
		fmt.Print(detector.SuggestSimplification(report))
	}

	// Cost comparison
	fmt.Printf("\n%s\n  COST COMPARISON: 5-agent vs. 1-agent customer support bot\n%s\n", strings.Repeat("═", 60), strings.Repeat("═", 60))
	proposed := 5
	simplified := 1
	proposedMonthly := float64(proposed) * oeTokensPerAgentPerCall * oeCallsPerAgentPerDay * oeDaysPerMonth / 1000 * oeCostPer1KTokens
	simplifiedMonthly := float64(simplified) * oeTokensPerAgentPerCall * oeCallsPerAgentPerDay * oeDaysPerMonth / 1000 * oeCostPer1KTokens
	fmt.Printf("\n  Assumptions: %d queries/day, %d tokens/agent/call\n", oeCallsPerAgentPerDay, oeTokensPerAgentPerCall)
	fmt.Printf("  5-agent monthly cost  : $%.2f\n", proposedMonthly)
	fmt.Printf("  1-agent monthly cost  : $%.2f\n", simplifiedMonthly)
	fmt.Printf("  Monthly savings       : $%.2f\n", proposedMonthly-simplifiedMonthly)
	fmt.Printf("  Annual savings        : $%.2f\n\n", (proposedMonthly-simplifiedMonthly)*12)
}
