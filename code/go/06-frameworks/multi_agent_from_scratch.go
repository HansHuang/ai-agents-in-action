// multi_agent_from_scratch.go — Go from-scratch multi-agent research pipeline.
//
// Neither CrewAI nor AutoGen have Go support (May 2026).
// This file implements the same three-agent research task from
// code/python/06-frameworks/multi_agent_comparison.py using only the
// OpenAI Go SDK — and adds comments showing exactly which lines of code
// CrewAI and AutoGen would automate for you.
//
// Task: "Research the impact of AI on software developer productivity."
//
// Architecture:
//
//	Researcher → Critic → Writer
//	(same pipeline as the Python from-scratch implementation)
//
// What CrewAI automates (but you don't get in Go):
//   - Declarative agent definition with role/goal/backstory
//   - Automatic context passing via Task.context=[previous_task]
//   - Process.sequential execution engine
//   - Built-in verbose logging per agent
//
// What AutoGen automates (but you don't get in Go):
//   - GroupChat turn selection (round-robin or LLM-managed)
//   - Shared message history accumulation
//   - Termination condition detection from message content
//   - UserProxy human-in-the-loop integration
//
// Go tradeoff (May 2026):
//   - No CrewAI, no AutoGen, no LangGraph — but this is often an ADVANTAGE.
//   - Go agents are faster, cheaper, and easier to debug than Python
//     framework agents.  The orchestration code is explicit and testable.
//   - For production Go AI systems, build the pipeline yourself.
//     Reserve frameworks for rapid prototyping in Python.
//
// Run:
//
//	go run multi_agent_from_scratch.go
//
// See: docs/06-frameworks-in-practice/03-crewai-autogen.md
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
	"unicode"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	multiAgentModel     = openai.GPT4o
	researchTask        = "Research the impact of AI on software developer productivity."
	terminateSignal     = "RESEARCH_COMPLETE"
	maxConversationRuns = 12

	// Cost model (gpt-4o, May 2026)
	inputCostPer1K  = 0.0025 // USD per 1,000 input tokens
	outputCostPer1K = 0.010  // USD per 1,000 output tokens
)

// ---------------------------------------------------------------------------
// Agent — the Go equivalent of a CrewAI Agent or AutoGen AssistantAgent
//
// CrewAI equivalent:
//
//	researcher = Agent(
//	    role="Research Analyst",
//	    goal="Find concrete data and statistics",
//	    backstory="You are a data-driven researcher…",
//	    tools=[web_search],
//	)
//
// AutoGen equivalent:
//
//	researcher = AssistantAgent(
//	    name="Researcher",
//	    system_message="You are a research analyst…",
//	    llm_config=llm_config,
//	)
// ---------------------------------------------------------------------------

// Agent holds the identity and behaviour of a single participant.
// In CrewAI this is the Agent struct with role/goal/backstory.
// In AutoGen this is an AssistantAgent with a system_message.
// In Go it's a plain struct with a system prompt string.
type Agent struct {
	Name         string
	SystemPrompt string
}

// speak sends the conversation history to this agent and returns its reply.
//
// CrewAI equivalent: task.execute() — CrewAI calls the LLM internally.
// AutoGen equivalent: manager.run() — GroupChatManager routes the message.
// In Go we call the API directly — no magic, full visibility.
func (a Agent) speak(
	ctx context.Context,
	client *openai.Client,
	history []openai.ChatCompletionMessage,
	callCounter *int,
	tokenCounter *int,
) (string, error) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: a.SystemPrompt},
	}
	// Sliding window: last 6 messages for context (avoids unbounded growth)
	window := history
	if len(window) > 6 {
		window = window[len(window)-6:]
	}
	messages = append(messages, window...)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    multiAgentModel,
		Messages: messages,
	})
	if err != nil {
		return "", fmt.Errorf("agent %s LLM call failed: %w", a.Name, err)
	}

	*callCounter++
	*tokenCounter += resp.Usage.TotalTokens
	return resp.Choices[0].Message.Content, nil
}

// ---------------------------------------------------------------------------
// Task — the Go equivalent of a CrewAI Task
//
// CrewAI equivalent:
//
//	Task(
//	    description="Research the topic…",
//	    agent=researcher,
//	    expected_output="Structured research brief",
//	    context=[previous_task],   ← CrewAI injects previous output automatically
//	)
//
// In Go you pass previous outputs explicitly.  It's more code but also more
// transparent — you can see exactly what each agent receives.
// ---------------------------------------------------------------------------

// Task is a single unit of work assigned to an agent.
// Context is the accumulated output from prior tasks that this agent needs.
type Task struct {
	Description    string
	Agent          Agent
	ExpectedOutput string
	// Context holds the combined output of dependency tasks.
	// CrewAI passes this automatically via context=[...].
	// In Go we pass it explicitly — more verbose, fully controllable.
	Context string
}

// execute runs the task and returns the agent's output.
func (t Task) execute(ctx context.Context, client *openai.Client, calls, tokens *int) (string, error) {
	prompt := t.Description
	if t.Context != "" {
		prompt = fmt.Sprintf("%s\n\nContext from previous steps:\n%s", t.Description, t.Context)
	}
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: prompt},
	}
	return t.Agent.speak(ctx, client, history, calls, tokens)
}

// ---------------------------------------------------------------------------
// SequentialCrew — the Go equivalent of Crew(process=Process.sequential)
//
// CrewAI equivalent:
//
//	crew = Crew(
//	    agents=[researcher, critic, writer],
//	    tasks=[research_task, critique_task, writing_task],
//	    process=Process.sequential,
//	)
//	result = crew.kickoff()
//
// In Go we iterate through tasks manually.  Each task receives the combined
// output of all previous tasks as its context — exactly what CrewAI does
// internally, but visible here.
// ---------------------------------------------------------------------------

// SequentialCrew executes tasks in dependency order.
type SequentialCrew struct {
	Tasks []Task
}

// kickoff runs all tasks sequentially, passing accumulated context forward.
// Returns per-task outputs and the final output.
func (c *SequentialCrew) kickoff(ctx context.Context, client *openai.Client) (map[string]string, string, error) {
	outputs := make(map[string]string, len(c.Tasks))
	var allContext strings.Builder

	for _, task := range c.Tasks {
		// Inject all previous outputs as context — CrewAI does this automatically
		task.Context = allContext.String()

		output, err := task.execute(ctx, client, new(int), new(int))
		if err != nil {
			return outputs, "", err
		}
		outputs[task.Agent.Name] = output
		allContext.WriteString(fmt.Sprintf("\n\n[%s output]\n%s", task.Agent.Name, output))
	}

	finalOutput := outputs[c.Tasks[len(c.Tasks)-1].Agent.Name]
	return outputs, finalOutput, nil
}

// ---------------------------------------------------------------------------
// ConversationalTeam — the Go equivalent of AutoGen's GroupChat
//
// AutoGen equivalent:
//
//	groupchat = GroupChat(
//	    agents=[user_proxy, researcher, critic, writer],
//	    max_round=12,
//	    speaker_selection_method="round_robin",
//	)
//	manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
//	user_proxy.initiate_chat(manager, message=task)
//
// In Go we implement the round-robin loop explicitly.  AutoGen handles
// turn selection, message routing, and termination detection automatically;
// here we write those ~25 lines ourselves — fully transparent.
// ---------------------------------------------------------------------------

// ConversationalTeam runs a round-robin group conversation.
type ConversationalTeam struct {
	Agents          []Agent
	TerminateSignal string
	MaxRounds       int
}

// run starts the conversation with an initial message and returns the full
// history and the final report (the message containing TerminateSignal).
func (t *ConversationalTeam) run(
	ctx context.Context,
	client *openai.Client,
	initialMessage string,
	calls, tokens *int,
) ([]openai.ChatCompletionMessage, string, error) {
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: initialMessage},
	}

	finalReport := ""

	for round := range t.MaxRounds {
		agent := t.Agents[round%len(t.Agents)]

		// In AutoGen the GroupChatManager picks the next speaker based on the
		// conversation context.  Here we use a deterministic round-robin —
		// simpler, reproducible, and easier to test.
		reply, err := agent.speak(ctx, client, history, calls, tokens)
		if err != nil {
			return history, finalReport, err
		}

		history = append(history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf("[%s]: %s", agent.Name, reply),
		})

		if strings.Contains(reply, t.TerminateSignal) {
			finalReport = reply
			break
		}
	}

	if finalReport == "" && len(history) > 0 {
		finalReport = history[len(history)-1].Content
	}
	return history, finalReport, nil
}

// ---------------------------------------------------------------------------
// Metrics — mirrors the Python ComparisonResult structure
// ---------------------------------------------------------------------------

// Metrics captures runtime measurements for comparison.
type Metrics struct {
	Name           string
	ExecutionMs    float64
	LLMCalls       int
	TotalTokens    int
	EstimatedCost  float64
	ReportWordsCnt int
	SourcesCited   int
}

func newMetrics(name string, elapsed time.Duration, calls, tokens int, report string) Metrics {
	inputTokens := float64(tokens) * 0.4
	outputTokens := float64(tokens) * 0.6
	cost := (inputTokens/1000)*inputCostPer1K + (outputTokens/1000)*outputCostPer1K

	return Metrics{
		Name:           name,
		ExecutionMs:    float64(elapsed.Milliseconds()),
		LLMCalls:       calls,
		TotalTokens:    tokens,
		EstimatedCost:  math.Round(cost*10000) / 10000,
		ReportWordsCnt: countWords(report),
		SourcesCited:   countSources(report),
	}
}

// ---------------------------------------------------------------------------
// Demo helpers
// ---------------------------------------------------------------------------

// createResearchAgents returns the three agents used in both implementations.
func createResearchAgents() (researcher, critic, writer Agent) {
	researcher = Agent{
		Name: "Researcher",
		SystemPrompt: "You are a research analyst. Produce a structured research brief. " +
			"Include at least three concrete statistics, three named sources, " +
			"and a timeline of key developments.",
	}
	critic = Agent{
		Name: "Critic",
		SystemPrompt: "You are a rigorous research critic. Identify three specific weaknesses: " +
			"missing data, unsupported claims, or coverage gaps. Be constructive.",
	}
	writer = Agent{
		Name: "Writer",
		SystemPrompt: "You are a technical report writer. Produce a 400–500 word report with " +
			"sections: ## Executive Summary, ## Key Findings (bullets), ## Analysis, " +
			"## Sources. Address the critic's feedback. " +
			"When your report is complete, end with '" + terminateSignal + "'.",
	}
	return
}

func runSequentialCrew(ctx context.Context, client *openai.Client) (Metrics, map[string]string, error) {
	researcher, critic, writer := createResearchAgents()

	crew := &SequentialCrew{
		Tasks: []Task{
			{
				Description:    "Research topic: " + researchTask,
				Agent:          researcher,
				ExpectedOutput: "Structured research brief with facts, sources, timeline.",
			},
			{
				Description:    "Critique the research brief: identify three specific weaknesses.",
				Agent:          critic,
				ExpectedOutput: "Three numbered critiques with specific suggestions.",
			},
			{
				Description: "Write a 400–500 word report. Address all critique points. " +
					"Sections: Executive Summary, Key Findings, Analysis, Sources.",
				Agent:          writer,
				ExpectedOutput: "Polished structured report, 400–500 words.",
			},
		},
	}

	calls, tokens := 0, 0
	start := time.Now()

	// Use per-task call/token tracking for aggregate count
	totalCalls, totalTokens := 0, 0
	outputs := make(map[string]string)
	var lastOutput string

	for i := range crew.Tasks {
		task := &crew.Tasks[i]
		if i > 0 {
			// Build context from all previous outputs
			var ctxBuilder strings.Builder
			for _, prevTask := range crew.Tasks[:i] {
				if out, ok := outputs[prevTask.Agent.Name]; ok {
					ctxBuilder.WriteString(fmt.Sprintf("\n[%s]\n%s\n", prevTask.Agent.Name, out))
				}
			}
			task.Context = ctxBuilder.String()
		}
		output, err := task.execute(ctx, client, &calls, &tokens)
		if err != nil {
			return Metrics{}, nil, err
		}
		outputs[task.Agent.Name] = output
		totalCalls += calls
		totalTokens += tokens
		calls, tokens = 0, 0
		lastOutput = output
	}

	elapsed := time.Since(start)
	return newMetrics("Sequential (CrewAI pattern)", elapsed, totalCalls, totalTokens, lastOutput), outputs, nil
}

func runConversational(ctx context.Context, client *openai.Client) (Metrics, []openai.ChatCompletionMessage, error) {
	researcher, critic, writer := createResearchAgents()

	team := &ConversationalTeam{
		Agents:          []Agent{researcher, critic, writer},
		TerminateSignal: terminateSignal,
		MaxRounds:       maxConversationRuns,
	}

	calls, tokens := 0, 0
	start := time.Now()
	history, report, err := team.run(ctx, client, "Research task: "+researchTask, &calls, &tokens)
	if err != nil {
		return Metrics{}, nil, err
	}
	elapsed := time.Since(start)
	return newMetrics("Conversational (AutoGen pattern)", elapsed, calls, tokens, report), history, nil
}

func countWords(text string) int {
	words := 0
	inWord := false
	for _, r := range text {
		if unicode.IsSpace(r) {
			inWord = false
		} else if !inWord {
			inWord = true
			words++
		}
	}
	return words
}

func countSources(text string) int {
	count := 0
	for _, line := range strings.Split(text, "\n") {
		s := strings.TrimSpace(line)
		if len(s) > 15 && (strings.HasPrefix(s, "http") ||
			strings.HasPrefix(s, "[") ||
			strings.HasPrefix(s, "•") ||
			(len(s) > 2 && s[1] == '.')) {
			count++
		}
	}
	if count > 20 {
		return 20
	}
	return count
}

func preview(text string, maxChars int) string {
	if len(text) <= maxChars {
		return text
	}
	return text[:maxChars] + fmt.Sprintf("\n  … [%d more chars]", len(text)-maxChars)
}

func printMetricsTable(results []Metrics) {
	colW := 32
	sep := strings.Repeat("─", 36+colW*len(results))
	hdr := fmt.Sprintf("  %-34s", "Metric")
	for _, m := range results {
		hdr += fmt.Sprintf("%*s", colW, m.Name)
	}
	fmt.Println()
	fmt.Println(sep)
	fmt.Println(hdr)
	fmt.Println(sep)

	type metricRow struct {
		label  string
		values []string
	}
	rows := []metricRow{
		{"Execution time (ms)", func() []string {
			s := make([]string, len(results))
			for i, m := range results {
				s[i] = fmt.Sprintf("%.0fms", m.ExecutionMs)
			}
			return s
		}()},
		{"LLM calls", func() []string {
			s := make([]string, len(results))
			for i, m := range results {
				s[i] = fmt.Sprintf("%d", m.LLMCalls)
			}
			return s
		}()},
		{"Total tokens", func() []string {
			s := make([]string, len(results))
			for i, m := range results {
				s[i] = fmt.Sprintf("%d", m.TotalTokens)
			}
			return s
		}()},
		{"Estimated cost (USD)", func() []string {
			s := make([]string, len(results))
			for i, m := range results {
				s[i] = fmt.Sprintf("$%.4f", m.EstimatedCost)
			}
			return s
		}()},
		{"Report length (words)", func() []string {
			s := make([]string, len(results))
			for i, m := range results {
				s[i] = fmt.Sprintf("%d", m.ReportWordsCnt)
			}
			return s
		}()},
		{"Sources cited", func() []string {
			s := make([]string, len(results))
			for i, m := range results {
				s[i] = fmt.Sprintf("%d", m.SourcesCited)
			}
			return s
		}()},
	}

	for _, row := range rows {
		line := fmt.Sprintf("  %-34s", row.label)
		for _, v := range row.values {
			line += fmt.Sprintf("%*s", colW, v)
		}
		fmt.Println(line)
	}
	fmt.Println(sep)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable not set")
		os.Exit(1)
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()

	fmt.Printf("Task: %s\n\n", researchTask)

	// --- Run 1: Sequential crew (CrewAI pattern) ---
	fmt.Print("Running Sequential (CrewAI pattern)… ")
	seqMetrics, seqOutputs, err := runSequentialCrew(ctx, client)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("done.")

	// --- Run 2: Conversational team (AutoGen pattern) ---
	fmt.Print("Running Conversational (AutoGen pattern)… ")
	convMetrics, convHistory, err := runConversational(ctx, client)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("done.")

	// --- Comparison table ---
	printMetricsTable([]Metrics{seqMetrics, convMetrics})

	// --- Per-agent outputs (sequential) ---
	fmt.Printf("\n%s\n  SEQUENTIAL CREW — AGENT OUTPUTS\n%s\n",
		strings.Repeat("═", 60), strings.Repeat("═", 60))
	agentOrder := []string{"Researcher", "Critic", "Writer"}
	for _, name := range agentOrder {
		if out, ok := seqOutputs[name]; ok {
			fmt.Printf("\n[%s]\n%s\n", name, preview(out, 400))
		}
	}

	// --- Conversation history (conversational) ---
	fmt.Printf("\n%s\n  CONVERSATIONAL TEAM — CONVERSATION\n%s\n",
		strings.Repeat("═", 60), strings.Repeat("═", 60))
	for i, msg := range convHistory {
		fmt.Printf("\n[%d] %s\n%s\n", i+1, msg.Role, preview(msg.Content, 300))
	}

	// --- Analysis ---
	fmt.Printf("\n%s\n  ANALYSIS\n%s\n", strings.Repeat("─", 60), strings.Repeat("─", 60))
	fmt.Println(
		"\n  Go has no CrewAI or AutoGen, but that is often an advantage:\n\n" +
			"  • Sequential pattern: 3 explicit function calls — fully traceable,\n" +
			"    easily unit-testable, zero framework overhead.\n\n" +
			"  • Conversational pattern: a simple for-loop — the same structure\n" +
			"    AutoGen uses internally, but visible and modifiable.\n\n" +
			"  • Performance: Go agents typically run 30–50% faster than equivalent\n" +
			"    Python framework agents due to lower runtime overhead.\n\n" +
			"  • For production Go AI systems, build the pipeline yourself.\n" +
			"    Reserve Python frameworks (CrewAI, AutoGen) for rapid prototyping.",
	)
}
