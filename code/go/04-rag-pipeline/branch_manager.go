// branch_manager.go — Parallel conversation contexts (branch manager).
//
// Allows an agent to explore hypothetical paths or handle sub-tasks without
// polluting the main conversation history. Each branch is an independent
// MemoryManager instance.
//
// See: docs/03-memory-and-retrieval/01-short-term-memory.md
package ragpipeline

import (
	"fmt"
	"sync"

	"crypto/rand"
	"encoding/hex"
)

// ---------------------------------------------------------------------------
// BranchManager
// ---------------------------------------------------------------------------

// BranchManager manages parallel conversation branches for hypothetical
// exploration. Each branch is a separate MemoryManager with its own history.
type BranchManager struct {
	systemPrompt string
	model        string
	maxTokens    int
	summarizer   *ConversationSummarizer

	mu       sync.Mutex
	branches map[string]*MemoryManager
}

// NewBranchManager creates a BranchManager.
func NewBranchManager(systemPrompt, model string, maxTokens int) *BranchManager {
	if model == "" {
		model = "gpt-4o"
	}
	if maxTokens == 0 {
		maxTokens = 100_000
	}
	return &BranchManager{
		systemPrompt: systemPrompt,
		model:        model,
		maxTokens:    maxTokens,
		summarizer:   NewConversationSummarizer("gpt-4o-mini"),
		branches:     make(map[string]*MemoryManager),
	}
}

// ---------------------------------------------------------------------------
// Branch lifecycle
// ---------------------------------------------------------------------------

func newBranchID(name string) string {
	b := make([]byte, 4)
	rand.Read(b)
	return fmt.Sprintf("%s-%s", name, hex.EncodeToString(b))
}

// CreateBranch creates a new branch and returns its ID.
// contextFrom optionally names a branch whose summary is injected as context.
func (bm *BranchManager) CreateBranch(name, userQuery, contextFrom string) (string, error) {
	branchID := newBranchID(name)

	mem := NewMemoryManager(MemoryManagerOptions{
		Model:        bm.model,
		MaxTokens:    bm.maxTokens,
		SystemPrompt: bm.systemPrompt,
		Summarizer:   bm.summarizer,
	})

	// Inject context summary from a parent branch.
	if contextFrom != "" {
		parent, err := bm.GetBranch(contextFrom)
		if err != nil {
			return "", err
		}
		parentMsgs := parent.Messages[1:] // skip system prompt
		if len(parentMsgs) > 0 {
			summary, _ := bm.summarizer.Summarize(parentMsgs)
			if summary != "" {
				content := fmt.Sprintf("[Context from branch '%s': %s]", contextFrom, summary)
				mem.AddUserMessage(content)
			}
		}
	}

	if userQuery != "" {
		mem.AddUserMessage(userQuery)
	}

	bm.mu.Lock()
	bm.branches[branchID] = mem
	bm.mu.Unlock()
	return branchID, nil
}

// GetBranch returns the MemoryManager for a branch.
func (bm *BranchManager) GetBranch(branchID string) (*MemoryManager, error) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	mem, ok := bm.branches[branchID]
	if !ok {
		return nil, fmt.Errorf("unknown branch: %q", branchID)
	}
	return mem, nil
}

// AddToBranch appends a message to a branch.
func (bm *BranchManager) AddToBranch(branchID string, msg Message) error {
	mem, err := bm.GetBranch(branchID)
	if err != nil {
		return err
	}
	mem.AddMessage(msg)
	return nil
}

// SummarizeBranch returns a text summary of a branch's conversation.
func (bm *BranchManager) SummarizeBranch(branchID string) (string, error) {
	mem, err := bm.GetBranch(branchID)
	if err != nil {
		return "", err
	}
	msgs := mem.Messages[1:] // skip system prompt
	if len(msgs) == 0 {
		return "", nil
	}
	return bm.summarizer.Summarize(msgs)
}

// CloseBranch summarizes and removes a branch, returning its summary.
func (bm *BranchManager) CloseBranch(branchID string) (string, error) {
	summary, err := bm.SummarizeBranch(branchID)
	if err != nil {
		return "", err
	}
	bm.mu.Lock()
	delete(bm.branches, branchID)
	bm.mu.Unlock()
	return summary, nil
}

// GetActiveBranches returns the IDs of all currently open branches.
func (bm *BranchManager) GetActiveBranches() []string {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	ids := make([]string, 0, len(bm.branches))
	for id := range bm.branches {
		ids = append(ids, id)
	}
	return ids
}

// MergeContext injects summaries of source branches into the target branch.
func (bm *BranchManager) MergeContext(targetBranch string, sourceBranches []string) error {
	target, err := bm.GetBranch(targetBranch)
	if err != nil {
		return err
	}
	for _, sourceID := range sourceBranches {
		summary, err := bm.SummarizeBranch(sourceID)
		if err != nil || summary == "" {
			continue
		}
		content := fmt.Sprintf("[Summary from branch '%s': %s]", sourceID, summary)
		target.AddUserMessage(content)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunBranchManager demonstrates parallel conversation branch management.
func RunBranchManager() {
	fmt.Println("BRANCH MANAGER DEMO")
	bm := NewBranchManager("You are a research assistant.", "gpt-4o", 100_000)

	mainID, err := bm.CreateBranch("main", "Research the electric vehicle market.", "")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	content := "The EV market is growing rapidly."
	bm.AddToBranch(mainID, Message{Role: "assistant", Content: &content})

	subID, err := bm.CreateBranch("ev-competitors", "Compare Tesla vs Rivian.", mainID)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	content2 := "Tesla leads in volume; Rivian focuses on adventure trucks."
	bm.AddToBranch(subID, Message{Role: "assistant", Content: &content2})

	fmt.Printf("Active branches: %v\n", bm.GetActiveBranches())

	if err := bm.MergeContext(mainID, []string{subID}); err != nil {
		fmt.Printf("MergeContext error: %v\n", err)
	}
	mainMem, _ := bm.GetBranch(mainID)
	fmt.Printf("Main branch messages after merge: %d\n", len(mainMem.Messages))

	subSummary, _ := bm.CloseBranch(subID)
	fmt.Printf("Sub-branch summary: %q\n", subSummary)
	fmt.Printf("Active branches: %v\n", bm.GetActiveBranches())
}
