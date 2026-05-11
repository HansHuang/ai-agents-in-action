// rag_agent.go — RAG integrated as a tool in an agent loop.
//
// The agent decides when to search the knowledge base versus answering directly
// from general knowledge. RAG is exposed as one tool among many; the model
// routes each query appropriately.
//
// See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
//
//	docs/02-the-agent-loop/01-anatomy-of-an-agent.md
package ragpipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const ragAgentSystemPrompt = `You are a helpful assistant with access to a company knowledge base and general knowledge.

DECISION RULES:
1. For general knowledge (math, common facts, science, coding): answer directly.
2. For company-specific information (policies, procedures, products, HR):
   ALWAYS call search_knowledge_base first.
3. If search_knowledge_base returns that it has no information, tell the user
   you couldn't find it in the knowledge base and offer to help another way.
4. Always cite the knowledge base when you use it.`

// ---------------------------------------------------------------------------
// Agent request / response types (separate from chatMessage used elsewhere)
// ---------------------------------------------------------------------------

type agentMsg struct {
	Role       string          `json:"role"`
	Content    *string         `json:"content"`
	ToolCalls  []agentToolCall `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

type agentToolCall struct {
	ID       string            `json:"id"`
	Type     string            `json:"type"`
	Function agentToolFunction `json:"function"`
}

type agentToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type agentReq struct {
	Model      string        `json:"model"`
	Messages   []agentMsg    `json:"messages"`
	Tools      []interface{} `json:"tools,omitempty"`
	ToolChoice interface{}   `json:"tool_choice,omitempty"`
}

type agentChoice struct {
	Message      agentMsg `json:"message"`
	FinishReason string   `json:"finish_reason"`
}

type agentResp struct {
	Choices []agentChoice `json:"choices"`
	Usage   chatUsage     `json:"usage"`
}

// ---------------------------------------------------------------------------
// AgentResponse
// ---------------------------------------------------------------------------

// AgentResponse is the result of a single RAGAgent.Run call.
type AgentResponse struct {
	Answer        string
	UsedRAG       bool
	RAGQueries    []string
	RAGSources    []string
	ToolCallsMade int
	DecisionTrail []string
}

// ---------------------------------------------------------------------------
// RAGAgent
// ---------------------------------------------------------------------------

// RAGAgent is an agent that uses RAG as one of its tools.
type RAGAgent struct {
	RAGPipeline *RAGPipeline
	Model       string
	apiKey      string
	tools       []interface{}
}

// NewRAGAgent creates a RAGAgent with a populated RAGPipeline.
func NewRAGAgent(ragPipeline *RAGPipeline, model string) *RAGAgent {
	if model == "" {
		model = "gpt-4o"
	}
	agent := &RAGAgent{
		RAGPipeline: ragPipeline,
		Model:       model,
		apiKey:      os.Getenv("OPENAI_API_KEY"),
	}
	agent.tools = agent.buildTools()
	return agent
}

func (a *RAGAgent) buildTools() []interface{} {
	return []interface{}{
		map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "search_knowledge_base",
				"description": "Search the company knowledge base for information. " +
					"Use this when the user asks about policies, procedures, products, HR topics, " +
					"or any company-specific information. " +
					"Do NOT use for general knowledge questions like math, common facts, science, or coding.",
				"parameters": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "Specific, targeted search query. Example: 'return policy for damaged electronics'",
						},
					},
					"required": []string{"query"},
				},
			},
		},
	}
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

func (a *RAGAgent) executeTool(ctx context.Context, toolName, arguments string) string {
	if toolName == "search_knowledge_base" {
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return `{"error": "failed to parse arguments"}`
		}
		query, _ := args["query"].(string)
		ragResp, err := a.RAGPipeline.Query(ctx, query, 0, 0)
		if err != nil {
			return fmt.Sprintf(`{"error": %q}`, err.Error())
		}
		result := map[string]interface{}{
			"answer":  ragResp.Answer,
			"sources": ragResp.Sources,
			"scores":  ragResp.SimilarityScores,
		}
		b, _ := json.Marshal(result)
		return string(b)
	}
	return fmt.Sprintf(`{"error": "unknown tool: %s"}`, toolName)
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

func strPtrAgent(s string) *string { return &s }

// Run executes the agent loop for a single user turn.
// The agent may call tools multiple times before producing a final answer.
func (a *RAGAgent) Run(ctx context.Context, userInput string) (*AgentResponse, error) {
	systemContent := ragAgentSystemPrompt
	userContent := userInput
	messages := []agentMsg{
		{Role: "system", Content: &systemContent},
		{Role: "user", Content: &userContent},
	}

	usedRAG := false
	var ragQueries []string
	var ragSources []string
	toolCallsMade := 0
	decisionTrail := []string{fmt.Sprintf("USER: %s", userInput)}

	const maxRounds = 5
	for round := 0; round < maxRounds; round++ {
		reqBody, err := json.Marshal(agentReq{
			Model:      a.Model,
			Messages:   messages,
			Tools:      a.tools,
			ToolChoice: "auto",
		})
		if err != nil {
			return nil, err
		}

		httpCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		httpReq, err := http.NewRequestWithContext(httpCtx, http.MethodPost,
			"https://api.openai.com/v1/chat/completions", bytes.NewReader(reqBody))
		if err != nil {
			cancel()
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+a.apiKey)

		resp, err := (&http.Client{}).Do(httpReq)
		cancel()
		if err != nil {
			return nil, err
		}
		respBytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBytes))
		}

		var ar agentResp
		if err := json.Unmarshal(respBytes, &ar); err != nil {
			return nil, err
		}
		if len(ar.Choices) == 0 {
			return nil, fmt.Errorf("no choices in response")
		}

		choice := ar.Choices[0].Message

		// Append assistant message.
		messages = append(messages, choice)

		// No tool calls → final answer.
		if len(choice.ToolCalls) == 0 {
			answer := ""
			if choice.Content != nil {
				answer = *choice.Content
			}
			trail := "AGENT: answered directly (no RAG)"
			if usedRAG {
				trail = "AGENT: answered using RAG"
			}
			decisionTrail = append(decisionTrail, trail)
			return &AgentResponse{
				Answer:        answer,
				UsedRAG:       usedRAG,
				RAGQueries:    ragQueries,
				RAGSources:    ragSources,
				ToolCallsMade: toolCallsMade,
				DecisionTrail: decisionTrail,
			}, nil
		}

		// Execute each tool call.
		for _, tc := range choice.ToolCalls {
			toolName := tc.Function.Name
			argsRaw := tc.Function.Arguments

			queryStr := argsRaw
			var parsed map[string]interface{}
			if err := json.Unmarshal([]byte(argsRaw), &parsed); err == nil {
				if q, ok := parsed["query"].(string); ok {
					queryStr = q
				}
			}

			decisionTrail = append(decisionTrail, fmt.Sprintf("TOOL CALL: %s(%q)", toolName, queryStr))
			result := a.executeTool(ctx, toolName, argsRaw)
			if len(result) > 120 {
				decisionTrail = append(decisionTrail, fmt.Sprintf("TOOL RESULT: %s", result[:120]))
			} else {
				decisionTrail = append(decisionTrail, fmt.Sprintf("TOOL RESULT: %s", result))
			}

			toolID := tc.ID
			resultContent := result
			messages = append(messages, agentMsg{
				Role:       "tool",
				ToolCallID: toolID,
				Content:    &resultContent,
			})

			toolCallsMade++
			if toolName == "search_knowledge_base" {
				usedRAG = true
				ragQueries = append(ragQueries, queryStr)
				var resultData map[string]interface{}
				if err := json.Unmarshal([]byte(result), &resultData); err == nil {
					if sources, ok := resultData["sources"].([]interface{}); ok {
						for _, s := range sources {
							if str, ok := s.(string); ok {
								ragSources = append(ragSources, str)
							}
						}
					}
				}
			}
		}
	}

	// Safety: return last assistant content if loop exhausted.
	lastContent := "Agent loop limit reached without a final answer."
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "assistant" && messages[i].Content != nil {
			lastContent = *messages[i].Content
			break
		}
	}
	decisionTrail = append(decisionTrail, "AGENT: loop limit reached")
	return &AgentResponse{
		Answer:        lastContent,
		UsedRAG:       usedRAG,
		RAGQueries:    ragQueries,
		RAGSources:    ragSources,
		ToolCallsMade: toolCallsMade,
		DecisionTrail: decisionTrail,
	}, nil
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunRAGAgent demonstrates the RAG agent with a company knowledge base.
func RunRAGAgent() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("RAG AGENT DEMO")
	fmt.Println(strings.Repeat("=", 70))

	embedder := NewEmbeddingGenerator("text-embedding-3-small", 0)
	vectorStore := NewSimpleVectorStore()
	pipeline := NewRAGPipeline(vectorStore, embedder, "gpt-4o", 200, 40, 5, 0.5)

	companyDocs := map[string]string{
		"vacation-policy.md": `# Vacation Policy
Full-time employees accrue 15 days of paid vacation per year.
Vacation must be approved by your manager at least 2 weeks in advance.
Unused vacation of up to 5 days may be rolled over to the following year.`,
		"expense-policy.md": `# Expense Reporting
Employees must submit expense reports within 30 days of the expense.
Receipts are required for all expenses over $25.
Meals: up to $75/person for client entertainment; $30/person for internal meals.`,
		"it-support.md": `# IT Support Procedures
To reset your password: visit https://password.internal.example.com or call IT at ext. 4357.
For new software requests, submit a ticket at helpdesk.internal.example.com.`,
	}

	ctx := context.Background()
	for source, text := range companyDocs {
		n, err := pipeline.IngestText(ctx, text, map[string]interface{}{"source": source})
		if err != nil {
			fmt.Printf("  Error ingesting %s: %v\n", source, err)
			continue
		}
		fmt.Printf("Ingested %s: %d chunks\n", source, n)
	}

	agent := NewRAGAgent(pipeline, "gpt-4o")

	questions := []string{
		"What's 2+2?",
		"How many vacation days do I get per year?",
		"How do I reset my password?",
	}

	for _, q := range questions {
		fmt.Printf("\nQ: %s\n", q)
		resp, err := agent.Run(ctx, q)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		fmt.Printf("A: %s\n", resp.Answer)
		fmt.Printf("   [used_rag=%v, tool_calls=%d]\n", resp.UsedRAG, resp.ToolCallsMade)
	}
}
