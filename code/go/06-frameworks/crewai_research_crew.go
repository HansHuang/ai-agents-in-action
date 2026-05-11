// crewai_research_crew.go — CrewAI-equivalent sequential research crew in Go.
//
// CrewAI has no official Go port. This file implements the same CONCEPT —
// a "crew" of specialised agents executing tasks sequentially — using the
// Agent / Task / SequentialCrew types already defined in multi_agent_from_scratch.go.
//
// Crew:
//   - ResearchAgent  : gathers information and raw notes
//   - AnalysisAgent  : synthesizes and identifies key findings
//   - WritingAgent   : produces a structured, citation-rich report
//
// The crew runs kickoff(), passing each agent's output to the next
// as context — exactly as CrewAI does.
//
// Run:
//
//	go run crewai_research_crew.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go langsmith_tracer.go \
//	           autogen_design_team.go
//
// See: docs/06-frameworks-in-practice/03-crewai-autogen.md
package main

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

// CrewResearchResult holds the output of a full research crew run.
type CrewResearchResult struct {
	Topic         string
	Report        string
	Sources       []string
	KeyFindings   []string
	AgentOutputs  map[string]string
	ExecutionTime float64
	TokenUsage    int
}

// ---------------------------------------------------------------------------
// Simulated external tools
// ---------------------------------------------------------------------------

// crewWebSearch simulates searching the web for a query.
func crewWebSearch(query string) string {
	results := map[string]string{
		"AI productivity":        "Study: GitHub Copilot increased developer output by 55% (GitHub, 2023). McKinsey: AI boosts knowledge-worker productivity 15–35%.",
		"developer tools":        "Stack Overflow Survey 2024: 77% of developers use or plan to use AI coding tools. Top tools: GitHub Copilot, Cursor, Codeium.",
		"software development":   "AI in SDLC: automated testing reduces manual effort 40%, code review time 30%. Source: Gartner 2024.",
		"impact on productivity": "NBER paper: LLM code assistants cut task completion time 56% for experienced devs. Junior devs: 37% faster.",
		"AI agents in software":  "Emerging trend: autonomous AI agents that can write, test, and commit code. Early adopters report 2–4x throughput gains.",
	}
	lower := strings.ToLower(query)
	for key, val := range results {
		if strings.Contains(lower, key) {
			return val
		}
	}
	return fmt.Sprintf("Web search for %q: multiple recent studies show significant productivity improvements with AI tools in software development.", query)
}

// crewDbLookup simulates a database lookup for structured data.
func crewDbLookup(query string) string {
	data := map[string]string{
		"statistics": "DB: 2024 AI Developer Survey — 68% report 20-40% time savings. 23% report >50% time savings. Sample size: 12,000 devs globally.",
		"roi":        "DB: Average ROI for AI coding tools: 3.5x in first 6 months. Cost: $20-40/dev/month. Benefit: ~10 hours saved per dev per week.",
		"challenges": "DB: Top concerns — code quality (42%), security (38%), job security (31%), over-reliance (27%). Source: IEEE Spectrum 2024.",
		"adoption":   "DB: Fortune 500 adoption of AI coding tools: 87% in 2024 vs 34% in 2022. SMB adoption: 52% in 2024.",
	}
	lower := strings.ToLower(query)
	for key, val := range data {
		if strings.Contains(lower, key) {
			return val
		}
	}
	return fmt.Sprintf("DB lookup for %q: structured data found in enterprise AI adoption database, Q4 2024 report.", query)
}

// crewFactVerify simulates fact-checking a claim.
func crewFactVerify(claim string) string {
	lower := strings.ToLower(claim)
	if strings.Contains(lower, "55%") {
		return "VERIFIED: GitHub Copilot study (Kalliamvakou 2022) — 55% faster task completion. Peer-reviewed."
	}
	if strings.Contains(lower, "56%") || strings.Contains(lower, "37%") {
		return "VERIFIED: NBER Working Paper #31161 (Noy & Zhang 2023). Sample: 453 college-educated professionals."
	}
	if strings.Contains(lower, "3.5x") {
		return "PARTIALLY VERIFIED: ROI figures vary by company size; 3.5x is a commonly cited median. Source: McKinsey, 2023."
	}
	if strings.Contains(lower, "87%") {
		return "VERIFIED: Gartner CIO survey 2024. Fortune 500 cohort, n=247 CIOs."
	}
	return fmt.Sprintf("UNVERIFIED: The claim %q could not be verified against known databases. Recommend citing a primary source.", claim)
}

// ---------------------------------------------------------------------------
// Crew builder
// ---------------------------------------------------------------------------

// crewAgentSpec extends Agent with tool information for crew use.
type crewAgentSpec struct {
	Agent Agent
	Tools []string
}

// crewBuildResearchAgents creates the three-agent research crew with tool access.
func crewBuildResearchAgents() []crewAgentSpec {
	return []crewAgentSpec{
		{
			Agent: Agent{
				Name: "ResearchAgent",
				SystemPrompt: "You are a senior research specialist. Your output MUST include:\n" +
					"1. At least 3 distinct data points with approximate sources.\n" +
					"2. One contrarian perspective or limitation.\n" +
					"3. Use the web_search and db_lookup tools to gather data.\n" +
					"Format: 'Research Notes:\\n...\\n\\nSources:\\n...'",
			},
			Tools: []string{"web_search", "db_lookup"},
		},
		{
			Agent: Agent{
				Name: "AnalysisAgent",
				SystemPrompt: "You are a strategic analysis expert. Using the research notes provided:\n" +
					"1. Extract 3–5 key findings (quantified where possible).\n" +
					"2. Identify one major risk or limitation.\n" +
					"3. Prioritise findings by business impact.\n" +
					"Use the fact_verify tool to validate statistics before including them.\n" +
					"Format: 'Key Findings:\\n...\\n\\nRisks:\\n...'",
			},
			Tools: []string{"fact_verify"},
		},
		{
			Agent: Agent{
				Name: "WritingAgent",
				SystemPrompt: "You are a senior technical writer. Produce a 300–400-word report that:\n" +
					"1. Opens with an executive summary (2 sentences).\n" +
					"2. Organises key findings in a logical flow.\n" +
					"3. Closes with actionable recommendations.\n" +
					"4. Cites sources inline (Author, Year) format.\n" +
					"The report must be self-contained — no bullet-only sections.",
			},
			Tools: nil,
		},
	}
}

// crewDispatchTool executes a crew tool by name.
func crewDispatchTool(toolName, input string) string {
	switch toolName {
	case "web_search":
		return crewWebSearch(input)
	case "db_lookup":
		return crewDbLookup(input)
	case "fact_verify":
		return crewFactVerify(input)
	default:
		return fmt.Sprintf("(unknown tool: %s)", toolName)
	}
}

// crewRunAgentWithTools runs a single agent turn, calling tools when the LLM requests them.
func crewRunAgentWithTools(ctx context.Context, client *openai.Client, spec crewAgentSpec, contextMsg string) (string, int, error) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: spec.Agent.SystemPrompt},
	}
	if contextMsg != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: contextMsg,
		})
	}

	// Build tool spec for the LLM (for documentation purposes)
	_ = spec.Tools
	for _, toolName := range spec.Tools {
		var description, paramName string
		switch toolName {
		case "web_search":
			description = "Search the web for current information on a topic."
			paramName = "query"
		case "db_lookup":
			description = "Look up structured data (statistics, ROI, surveys) from the internal database."
			paramName = "query"
		case "fact_verify":
			description = "Verify whether a specific claim or statistic is supported by primary sources."
			paramName = "claim"
		default:
			description = "Call " + toolName
			paramName = "input"
		}
		tools = append(tools, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        toolName,
				Description: description,
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						paramName: map[string]interface{}{"type": "string", "description": description},
					},
					"required": []string{paramName},
				},
			},
		})
	}

	totalTokens := 0
	// Up to 5 tool-call rounds
	for iter := 0; iter < 5; iter++ {
		req := openai.ChatCompletionRequest{
			Model:     "gpt-4o-mini",
			Messages:  messages,
			MaxTokens: 600,
		}
		if len(tools) > 0 {
			req.Tools = tools
		}

		resp, err := client.CreateChatCompletion(ctx, req)
		if err != nil {
			return "", totalTokens, err
		}
		totalTokens += resp.Usage.TotalTokens
		choice := resp.Choices[0]

		// No tool calls — we have the final answer
		if len(choice.Message.ToolCalls) == 0 {
			return choice.Message.Content, totalTokens, nil
		}

		// Process tool calls
		messages = append(messages, choice.Message)
		for _, tc := range choice.Message.ToolCalls {
			result := crewDispatchTool(tc.Function.Name, tc.Function.Arguments)
			messages = append(messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleUser,
				Content:    result,
				Name:       tc.Function.Name,
				ToolCallID: tc.ID,
			})
		}
	}

	// Fallback if we hit the iter limit
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     "gpt-4o-mini",
		Messages:  messages,
		MaxTokens: 600,
	})
	if err != nil {
		return "", totalTokens, err
	}
	return resp.Choices[0].Message.Content, totalTokens + resp.Usage.TotalTokens, nil
}

// crewResearch runs the full three-agent research crew for the given topic.
func crewResearch(ctx context.Context, client *openai.Client, topic string) (CrewResearchResult, error) {
	t0 := time.Now()
	agents := crewBuildResearchAgents()
	result := CrewResearchResult{
		Topic:        topic,
		AgentOutputs: make(map[string]string),
	}

	contextMsg := fmt.Sprintf("Research topic: %s", topic)

	for _, agent := range agents {
		agentOut, tokens, err := crewRunAgentWithTools(ctx, client, agent, contextMsg)
		if err != nil {
			return result, fmt.Errorf("agent %s: %w", agent.Agent.Name, err)
		}
		result.TokenUsage += tokens
		result.AgentOutputs[agent.Agent.Name] = agentOut

		// Each agent's output becomes context for the next
		contextMsg = fmt.Sprintf("Topic: %s\n\n%s output:\n%s", topic, agent.Agent.Name, agentOut)
	}

	result.Report = result.AgentOutputs["WritingAgent"]
	result.Sources = crewExtractSources(result.AgentOutputs["ResearchAgent"])
	result.KeyFindings = crewExtractKeyFindings(result.AgentOutputs["AnalysisAgent"])
	result.ExecutionTime = time.Since(t0).Seconds()
	return result, nil
}

// crewExtractSources extracts source citations from research agent output.
func crewExtractSources(text string) []string {
	var sources []string
	re := regexp.MustCompile(`(?i)(source[s]?:|from:|reference[s]?:|cited in:)\s*([^\n]+)`)
	matches := re.FindAllStringSubmatch(text, -1)
	for _, m := range matches {
		if len(m) >= 3 {
			s := strings.TrimSpace(m[2])
			if len(s) > 0 {
				sources = append(sources, s)
			}
		}
	}
	// Fallback: look for lines with parenthetical year citations
	yearRE := regexp.MustCompile(`\([A-Z][a-zA-Z\s]+,?\s*(19|20)\d{2}\)`)
	for _, line := range strings.Split(text, "\n") {
		if yearRE.MatchString(line) {
			line = strings.TrimSpace(line)
			if len(line) > 10 {
				sources = append(sources, line)
			}
		}
	}
	if len(sources) == 0 {
		sources = []string{"(no explicit sources extracted — see Research Notes)"}
	}
	return sources
}

// crewExtractKeyFindings extracts bullet key findings from analysis agent output.
func crewExtractKeyFindings(text string) []string {
	var findings []string
	inSection := false
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		lowerLine := strings.ToLower(trimmed)
		if strings.Contains(lowerLine, "key finding") || strings.Contains(lowerLine, "findings:") {
			inSection = true
			continue
		}
		if inSection {
			if strings.HasPrefix(trimmed, "-") || strings.HasPrefix(trimmed, "•") ||
				(len(trimmed) > 0 && trimmed[0] >= '1' && trimmed[0] <= '9') {
				finding := strings.TrimLeft(trimmed, "-•0123456789. ")
				if len(finding) > 10 {
					findings = append(findings, finding)
				}
			} else if strings.Contains(lowerLine, "risk") || strings.Contains(lowerLine, "limitation") {
				inSection = false
			}
		}
	}
	if len(findings) == 0 {
		findings = []string{"(see Analysis Agent output)"}
	}
	return findings
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runCrewAIResearchCrewDemo is the demo entry point (no main()).
func runCrewAIResearchCrewDemo(ctx context.Context, client *openai.Client) {
	topics := []string{
		"The impact of AI on software developer productivity",
		"How multi-agent AI systems change software architecture",
	}

	for i, topic := range topics {
		fmt.Printf("\n%s\n  RESEARCH CREW — TOPIC %d\n%s\n\n", strings.Repeat("═", 60), i+1, strings.Repeat("═", 60))
		fmt.Printf("Topic: %s\n\n", topic)

		result, err := crewResearch(ctx, client, topic)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("Execution time: %.2fs\n", result.ExecutionTime)
		fmt.Printf("Tokens used  : %d\n", result.TokenUsage)
		fmt.Printf("Sources found: %d\n", len(result.Sources))
		fmt.Printf("Key findings : %d\n", len(result.KeyFindings))

		if len(result.KeyFindings) > 0 {
			fmt.Printf("\nKey Findings:\n")
			for j, f := range result.KeyFindings {
				if j >= 5 {
					break
				}
				fmt.Printf("  %d. %s\n", j+1, f)
			}
		}

		fmt.Printf("\nFinal Report (preview):\n")
		report := result.Report
		if len(report) > 600 {
			report = report[:600] + "\n…(truncated)"
		}
		fmt.Printf("%s\n", report)
	}
}
