import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  buildSystemPrompt,
  buildUserPrompt,
  buildMessages,
  countTokens,
} from "./prompt_template.js";

// ---------------------------------------------------------------------------
// buildSystemPrompt
// ---------------------------------------------------------------------------

describe("buildSystemPrompt", () => {
  it("contains the focus area in the output", () => {
    const result = buildSystemPrompt("practical implementation details");
    assert.ok(result.includes("practical implementation details"));
  });

  it("has no leftover template braces", () => {
    const result = buildSystemPrompt("testing");
    assert.ok(!result.includes("{") && !result.includes("}"));
  });

  it("throws on empty string", () => {
    assert.throws(() => buildSystemPrompt(""), /empty/i);
  });

  it("throws on whitespace-only string", () => {
    assert.throws(() => buildSystemPrompt("   "), /empty/i);
  });
});

// ---------------------------------------------------------------------------
// buildUserPrompt
// ---------------------------------------------------------------------------

describe("buildUserPrompt", () => {
  it("contains the article text in the output", () => {
    const result = buildUserPrompt("AI agents are fascinating.");
    assert.ok(result.includes("AI agents are fascinating."));
  });

  it("has no leftover template braces", () => {
    const result = buildUserPrompt("Some article text");
    assert.ok(!result.includes("{") && !result.includes("}"));
  });

  it("throws on empty string", () => {
    assert.throws(() => buildUserPrompt(""), /empty/i);
  });

  it("throws on whitespace-only string", () => {
    assert.throws(() => buildUserPrompt("   "), /empty/i);
  });

  it("handles very long input without crashing", () => {
    const longText = "word ".repeat(2_000);
    const result = buildUserPrompt(longText);
    assert.ok(result.includes(longText));
  });
});

// ---------------------------------------------------------------------------
// buildMessages
// ---------------------------------------------------------------------------

describe("buildMessages", () => {
  it("returns exactly two messages", () => {
    const msgs = buildMessages("focus", "text");
    assert.equal(msgs.length, 2);
  });

  it("first message role is system", () => {
    const msgs = buildMessages("focus", "text");
    assert.equal(msgs[0].role, "system");
  });

  it("second message role is user", () => {
    const msgs = buildMessages("focus", "text");
    assert.equal(msgs[1].role, "user");
  });

  it("system message contains focus area", () => {
    const msgs = buildMessages("unique-focus-area", "text");
    assert.ok(msgs[0].content.includes("unique-focus-area"));
  });

  it("user message contains article text", () => {
    const msgs = buildMessages("focus", "unique-article-content");
    assert.ok(msgs[1].content.includes("unique-article-content"));
  });

  it("no leftover braces in any message", () => {
    const msgs = buildMessages("focus", "text");
    for (const msg of msgs) {
      assert.ok(!msg.content.includes("{") && !msg.content.includes("}"));
    }
  });

  it("throws on empty focusArea", () => {
    assert.throws(() => buildMessages("", "text"), /empty/i);
  });

  it("throws on empty articleText", () => {
    assert.throws(() => buildMessages("focus", ""), /empty/i);
  });
});

// ---------------------------------------------------------------------------
// countTokens
// ---------------------------------------------------------------------------

describe("countTokens", () => {
  it("returns a positive integer for normal input", () => {
    const msgs = buildMessages("focus area", "article text here");
    assert.ok(countTokens(msgs) > 0);
  });

  it("longer input yields more tokens", () => {
    const short = buildMessages("focus", "short text");
    const long = buildMessages("focus", "short text " + "extra content ".repeat(100));
    assert.ok(countTokens(long) > countTokens(short));
  });

  it("very long input does not crash", () => {
    const msgs = buildMessages("focus", "word ".repeat(2_000));
    assert.ok(countTokens(msgs) > 0);
  });

  it("empty messages list returns integer (reply-primer overhead)", () => {
    const result = countTokens([]);
    assert.equal(typeof result, "number");
    assert.equal(result, 3);
  });
});
