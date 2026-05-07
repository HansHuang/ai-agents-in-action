// Context Assembly demo — runs ContextBudget and ContextAssembler.
//
// See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
package main

import (
	"fmt"
	"strings"
)

func main() {
	fmt.Println("=== Context Budget & Assembler Demo (Go) ===\n")

	budget := NewContextBudget(8_000, "gpt-4o")

	fmt.Println("Budget allocations:")
	budgets, _ := budget.GetAllBudgets()
	for zone, tok := range budgets {
		pct := budget.Allocations[zone] * 100
		fmt.Printf("  %-25s  %6d tokens  (%.0f%%)\n", zone, tok, pct)
	}
	fmt.Printf("\nTotal window: %d tokens\n\n", budget.TotalTokens)

	systemPrompt := strings.Repeat(
		"You are a customer support agent. Use the knowledge base to answer. "+
			"If unsure, escalate.\n",
		5,
	)

	var messages []map[string]any
	for i := 1; i <= 10; i++ {
		messages = append(messages,
			map[string]any{
				"role":    "user",
				"content": fmt.Sprintf("Turn %d: %s", i, strings.Repeat("Tell me about context engineering. ", 10)),
			},
			map[string]any{
				"role":    "assistant",
				"content": fmt.Sprintf("Turn %d reply: %s", i, strings.Repeat("Context engineering manages the LLM context window as a finite resource. ", 8)),
			},
		)
	}

	ragContext := strings.Repeat(
		"## Knowledge Base: Context Windows\nContext windows are measured in tokens.\n\n",
		20,
	)

	spTok, _ := budget.MeasureZone("system_prompt", systemPrompt)
	dcTok, _ := budget.MeasureZone("dynamic_context", ragContext)
	fmt.Printf("Before enforcement:\n")
	fmt.Printf("  system_prompt   : %6d tokens\n", spTok)
	fmt.Printf("  dynamic_context : %6d tokens\n\n", dcTok)

	result, err := budget.Enforce(systemPrompt, messages, ragContext, nil)
	if err != nil {
		fmt.Printf("Enforce error: %v\n", err)
		return
	}

	fmt.Println("After enforcement:")
	for zone, audit := range result.Audit {
		fmt.Printf("  %-25s: %6d → %5d  (%s)\n",
			zone, audit.OriginalTokens, audit.FinalTokens, audit.ActionTaken)
	}
	fmt.Printf("\nTotal tokens saved : %d\n", result.TotalTokensSaved())
	fmt.Printf("Total tokens used  : %d\n\n", result.TotalTokensUsed())

	if len(result.Warnings) > 0 {
		fmt.Println("Warnings:")
		for _, w := range result.Warnings {
			fmt.Printf("  ⚠  %s\n", w)
		}
	}

	// Assembler demo
	fmt.Println("\n--- Context Assembler Demo ---")
	assembler := NewContextAssembler(budget, "gpt-4o")
	docs := []Document{
		{
			Text:     "## Cancellation Policy\nCancel any time. Takes effect end of billing period.",
			Metadata: map[string]any{"source": "cancellation.md"},
		},
		{
			Text:     "## Billing\nInvoices at Settings → Billing → Invoice History.",
			Metadata: map[string]any{"source": "billing.md"},
		},
	}
	asmResult, err := assembler.Assemble(AssembleOptions{
		Template:      "You are a $role support agent. Answer using only the documents.",
		Variables:     map[string]string{"role": "customer"},
		RetrievedDocs: docs,
		UserProfile:   map[string]string{"name": "Alice", "plan": "Pro"},
		Query:         "How do I cancel my subscription?",
		Optimize:      true,
	})
	if err != nil {
		fmt.Printf("Assemble error: %v\n", err)
		return
	}

	fmt.Printf("Sources included : %v\n", asmResult.SourcesIncluded)
	fmt.Printf("Total tokens     : %d\n", asmResult.TotalTokens())
}
