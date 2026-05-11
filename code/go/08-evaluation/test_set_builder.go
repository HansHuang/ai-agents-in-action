package main

import (
	"fmt"
	"math/rand"
)

// ---------------------------------------------------------------------------
// TestSetBuilder
// ---------------------------------------------------------------------------

// TestCase is a single input/expected-output pair for evaluation.
type TestCase struct {
	ID       string
	Input    string
	Expected string
	Tags     []string
	Metadata map[string]string
}

// TestSet is a labelled collection of TestCases.
type TestSet struct {
	Name        string
	Description string
	Cases       []TestCase
}

// TestSetBuilder constructs evaluation test sets from templates and sampling.
type TestSetBuilder struct {
	templates []TestCase
	cases     []TestCase
}

// NewTestSetBuilder creates an empty builder.
func NewTestSetBuilder() *TestSetBuilder { return &TestSetBuilder{} }

// AddTemplate registers a template case that can be expanded.
func (b *TestSetBuilder) AddTemplate(tc TestCase) {
	b.templates = append(b.templates, tc)
}

// Add adds a concrete test case directly.
func (b *TestSetBuilder) Add(tc TestCase) {
	if tc.ID == "" {
		tc.ID = fmt.Sprintf("TC-%04d", len(b.cases)+1)
	}
	b.cases = append(b.cases, tc)
}

// Sample randomly samples n cases from the current set (without replacement).
func (b *TestSetBuilder) Sample(n int) []TestCase {
	pool := make([]TestCase, len(b.cases))
	copy(pool, b.cases)
	rand.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	if n > len(pool) {
		n = len(pool)
	}
	return pool[:n]
}

// Build finalises and returns the TestSet.
func (b *TestSetBuilder) Build(name, description string) TestSet {
	return TestSet{
		Name:        name,
		Description: description,
		Cases:       append([]TestCase(nil), b.cases...),
	}
}

// FilterByTag returns only test cases that have the given tag.
func (ts TestSet) FilterByTag(tag string) TestSet {
	var filtered []TestCase
	for _, c := range ts.Cases {
		for _, t := range c.Tags {
			if t == tag {
				filtered = append(filtered, c)
				break
			}
		}
	}
	return TestSet{Name: ts.Name + "/" + tag, Description: ts.Description, Cases: filtered}
}

// RunTestSetBuilderDemo demonstrates building an evaluation test set.
func RunTestSetBuilderDemo() {
	builder := NewTestSetBuilder()

	questions := []struct {
		input, expected string
		tags            []string
	}{
		{"What is 2+2?", "4", []string{"math", "easy"}},
		{"Capital of France?", "Paris", []string{"geography", "easy"}},
		{"Who wrote Hamlet?", "Shakespeare", []string{"literature", "medium"}},
		{"Explain quantum entanglement", "quantum physics concept involving correlated particles", []string{"science", "hard"}},
		{"What is the Pythagorean theorem?", "a² + b² = c²", []string{"math", "medium"}},
	}

	for i, q := range questions {
		builder.Add(TestCase{
			ID:       fmt.Sprintf("TC-%03d", i+1),
			Input:    q.input,
			Expected: q.expected,
			Tags:     q.tags,
		})
	}

	ts := builder.Build("baseline-v1", "Baseline evaluation test set")
	fmt.Printf("Test set: %s (%d cases)\n", ts.Name, len(ts.Cases))

	mathCases := ts.FilterByTag("math")
	fmt.Printf("Math cases: %d\n", len(mathCases.Cases))

	sample := builder.Sample(3)
	fmt.Printf("Sampled 3 cases: %v\n", func() []string {
		ids := make([]string, len(sample))
		for i, c := range sample {
			ids[i] = c.ID
		}
		return ids
	}())
}
