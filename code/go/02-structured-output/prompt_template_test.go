package main

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// buildSystemPrompt
// ---------------------------------------------------------------------------

func TestBuildSystemPromptContainsFocusArea(t *testing.T) {
	result, err := buildSystemPrompt("practical implementation details")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "practical implementation details") {
		t.Error("system prompt does not contain the focus area")
	}
}

func TestBuildSystemPromptNoLeftoverBraces(t *testing.T) {
	result, err := buildSystemPrompt("testing")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(result, "{") || strings.Contains(result, "}") {
		t.Errorf("leftover template braces found in: %q", result)
	}
}

func TestBuildSystemPromptEmptyReturnsError(t *testing.T) {
	_, err := buildSystemPrompt("")
	if err == nil {
		t.Error("expected error for empty focusArea, got nil")
	}
}

func TestBuildSystemPromptWhitespaceOnlyReturnsError(t *testing.T) {
	_, err := buildSystemPrompt("   ")
	if err == nil {
		t.Error("expected error for whitespace-only focusArea, got nil")
	}
}

// ---------------------------------------------------------------------------
// buildUserPrompt
// ---------------------------------------------------------------------------

func TestBuildUserPromptContainsArticleText(t *testing.T) {
	result, err := buildUserPrompt("AI agents are fascinating.")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "AI agents are fascinating.") {
		t.Error("user prompt does not contain the article text")
	}
}

func TestBuildUserPromptNoLeftoverBraces(t *testing.T) {
	result, err := buildUserPrompt("Some article text")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(result, "{") || strings.Contains(result, "}") {
		t.Errorf("leftover template braces found in: %q", result)
	}
}

func TestBuildUserPromptEmptyReturnsError(t *testing.T) {
	_, err := buildUserPrompt("")
	if err == nil {
		t.Error("expected error for empty articleText, got nil")
	}
}

func TestBuildUserPromptWhitespaceOnlyReturnsError(t *testing.T) {
	_, err := buildUserPrompt("   ")
	if err == nil {
		t.Error("expected error for whitespace-only articleText, got nil")
	}
}

func TestBuildUserPromptVeryLongInputDoesNotCrash(t *testing.T) {
	longText := strings.Repeat("word ", 2_000)
	result, err := buildUserPrompt(longText)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, longText) {
		t.Error("long text was not included in the user prompt")
	}
}

// ---------------------------------------------------------------------------
// buildMessages
// ---------------------------------------------------------------------------

func TestBuildMessagesReturnsTwoMessages(t *testing.T) {
	msgs, err := buildMessages("focus", "text")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %d", len(msgs))
	}
}

func TestBuildMessagesFirstRoleIsSystem(t *testing.T) {
	msgs, err := buildMessages("focus", "text")
	if err != nil {
		t.Fatal(err)
	}
	if msgs[0].Role != "system" {
		t.Errorf("expected role=system, got %q", msgs[0].Role)
	}
}

func TestBuildMessagesSecondRoleIsUser(t *testing.T) {
	msgs, err := buildMessages("focus", "text")
	if err != nil {
		t.Fatal(err)
	}
	if msgs[1].Role != "user" {
		t.Errorf("expected role=user, got %q", msgs[1].Role)
	}
}

func TestBuildMessagesSystemContainsFocusArea(t *testing.T) {
	msgs, err := buildMessages("unique-focus-area", "text")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(msgs[0].Content, "unique-focus-area") {
		t.Error("system message does not contain the focus area")
	}
}

func TestBuildMessagesUserContainsArticleText(t *testing.T) {
	msgs, err := buildMessages("focus", "unique-article-content")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(msgs[1].Content, "unique-article-content") {
		t.Error("user message does not contain the article text")
	}
}

func TestBuildMessagesNoLeftoverBraces(t *testing.T) {
	msgs, err := buildMessages("focus", "text")
	if err != nil {
		t.Fatal(err)
	}
	for _, msg := range msgs {
		if strings.Contains(msg.Content, "{") || strings.Contains(msg.Content, "}") {
			t.Errorf("leftover braces in message: %q", msg.Content)
		}
	}
}

func TestBuildMessagesEmptyFocusReturnsError(t *testing.T) {
	_, err := buildMessages("", "text")
	if err == nil {
		t.Error("expected error for empty focusArea, got nil")
	}
}

func TestBuildMessagesEmptyArticleReturnsError(t *testing.T) {
	_, err := buildMessages("focus", "")
	if err == nil {
		t.Error("expected error for empty articleText, got nil")
	}
}

// ---------------------------------------------------------------------------
// countTokens
// ---------------------------------------------------------------------------

func TestCountTokensReturnsPositiveInteger(t *testing.T) {
	msgs, err := buildMessages("focus area", "article text here")
	if err != nil {
		t.Fatal(err)
	}
	n, err := countTokens(msgs, model)
	if err != nil {
		t.Fatal(err)
	}
	if n <= 0 {
		t.Errorf("expected positive token count, got %d", n)
	}
}

func TestCountTokensLongerInputYieldsMoreTokens(t *testing.T) {
	short, _ := buildMessages("focus", "short text")
	long, _ := buildMessages("focus", "short text "+strings.Repeat("extra content ", 100))
	nShort, _ := countTokens(short, model)
	nLong, _ := countTokens(long, model)
	if nLong <= nShort {
		t.Errorf("longer input should yield more tokens: short=%d long=%d", nShort, nLong)
	}
}

func TestCountTokensVeryLongInputDoesNotCrash(t *testing.T) {
	msgs, _ := buildMessages("focus", strings.Repeat("word ", 2_000))
	n, err := countTokens(msgs, model)
	if err != nil {
		t.Fatal(err)
	}
	if n <= 0 {
		t.Error("expected positive token count for long input")
	}
}

func TestCountTokensEmptyMessagesReturnsReplyPrimerOverhead(t *testing.T) {
	n, err := countTokens([]message{}, model)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Errorf("expected 3 (reply primer overhead), got %d", n)
	}
}
