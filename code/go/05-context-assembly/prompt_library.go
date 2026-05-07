// Prompt Library — version-controlled YAML-based prompt template management.
//
// Go port of code/python/05-context-assembly/prompt_library.py
//
// Same struct names (PromptLibrary, PromptTemplate, RenderedPrompt), same
// PascalCase methods, same {variable} syntax, same YAML template format.
//
// Uses gopkg.in/yaml.v3 for YAML parsing.
//
// See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// ---------------------------------------------------------------------------
// YAML schema types
// ---------------------------------------------------------------------------

type sectionYAML struct {
	Condition string `yaml:"condition"`
	Content   string `yaml:"content"`
}

type templateYAML struct {
	Name        string                 `yaml:"name"`
	Version     string                 `yaml:"version"`
	Description string                 `yaml:"description"`
	Template    string                 `yaml:"template"`
	Sections    map[string]sectionYAML `yaml:"sections"`
	Parent      string                 `yaml:"parent"`
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// RenderedPrompt holds the output of rendering a template.
type RenderedPrompt struct {
	RenderedText     string
	TemplateName     string
	TemplateVersion  string
	SectionsIncluded []string
	VariablesUsed    map[string]any
	TokenCount       int
}

// String returns a human-readable summary of the rendered prompt.
func (r *RenderedPrompt) String() string {
	return fmt.Sprintf("[%s v%s] sections=%v tokens=%d\n%s",
		r.TemplateName, r.TemplateVersion,
		r.SectionsIncluded, r.TokenCount,
		r.RenderedText,
	)
}

// LibraryPromptSection is a conditional section loaded from YAML.
type LibraryPromptSection struct {
	Name      string
	Condition string
	Content   string
}

// PromptTemplate is a loaded YAML prompt template.
type PromptTemplate struct {
	Name         string
	Version      string
	Description  string
	BaseTemplate string
	Sections     map[string]*LibraryPromptSection
	Parent       string
}

// Render renders the template with the provided variables and optional context.
// If activeSections is nil, section conditions are evaluated automatically.
func (t *PromptTemplate) Render(
	variables map[string]any,
	context map[string]any,
	activeSections []string,
) (*RenderedPrompt, error) {
	allVars := make(map[string]any)
	for k, v := range context {
		allVars[k] = v
	}
	for k, v := range variables {
		allVars[k] = v
	}

	// Determine active sections
	var included []string
	if activeSections != nil {
		included = activeSections
	} else {
		// Sorted section names for determinism
		names := make([]string, 0, len(t.Sections))
		for n := range t.Sections {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, name := range names {
			sec := t.Sections[name]
			if sec.Condition != "" && EvaluateCondition(sec.Condition, allVars) {
				included = append(included, name)
			}
		}
	}

	// Build sections block
	var sectionParts []string
	for _, name := range included {
		if sec, ok := t.Sections[name]; ok {
			sectionParts = append(sectionParts, strings.TrimRight(sec.Content, "\n"))
		}
	}
	sectionsBlock := strings.Join(sectionParts, "\n\n")

	// Fill variables — leave missing as {key}
	strVars := map[string]string{"sections": sectionsBlock}
	for k, v := range allVars {
		strVars[k] = fmt.Sprintf("%v", v)
	}
	rendered, _ := fillTemplate(t.BaseTemplate, strVars)
	rendered = strings.TrimRight(rendered, "\n")

	// Append sections if no placeholder
	if !strings.Contains(t.BaseTemplate, "{sections}") && sectionsBlock != "" {
		rendered += "\n\n" + sectionsBlock
	}

	tokenCount := countTokensStr(rendered)
	return &RenderedPrompt{
		RenderedText:     rendered,
		TemplateName:     t.Name,
		TemplateVersion:  t.Version,
		SectionsIncluded: included,
		VariablesUsed:    allVars,
		TokenCount:       tokenCount,
	}, nil
}

// RequiredVariables returns all {variable} names used in the template and sections.
func (t *PromptTemplate) RequiredVariables() []string {
	seen := make(map[string]bool)
	var result []string
	sources := []string{t.BaseTemplate}
	for _, sec := range t.Sections {
		sources = append(sources, sec.Content)
	}
	for _, src := range sources {
		for _, m := range varRE.FindAllStringSubmatch(src, -1) {
			if !seen[m[1]] {
				seen[m[1]] = true
				result = append(result, m[1])
			}
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// PromptLibrary
// ---------------------------------------------------------------------------

// PromptLibrary loads, manages, and renders version-controlled YAML templates.
type PromptLibrary struct {
	PromptsDir string
	Templates  map[string]*PromptTemplate
	// history: name → []*PromptTemplate (all loaded versions)
	history map[string][]*PromptTemplate
	// raw YAML text: "name@version" → raw text
	rawYAML map[string]string
}

// NewPromptLibrary creates a PromptLibrary and loads all templates.
func NewPromptLibrary(promptsDir string) (*PromptLibrary, error) {
	lib := &PromptLibrary{
		PromptsDir: promptsDir,
		Templates:  make(map[string]*PromptTemplate),
		history:    make(map[string][]*PromptTemplate),
		rawYAML:    make(map[string]string),
	}
	return lib, lib.LoadAll()
}

// LoadAll loads all .yaml template files from the prompts directory.
func (lib *PromptLibrary) LoadAll() error {
	if _, err := os.Stat(lib.PromptsDir); os.IsNotExist(err) {
		log.Printf("Prompts directory %s does not exist", lib.PromptsDir)
		return nil
	}
	count := 0
	err := filepath.WalkDir(lib.PromptsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if strings.HasSuffix(path, ".yaml") {
			if loadErr := lib.loadFile(path); loadErr != nil {
				log.Printf("Warning: failed to load %s: %v", path, loadErr)
			} else {
				count++
			}
		}
		return nil
	})
	log.Printf("Loaded %d prompt templates from %s", count, lib.PromptsDir)
	return err
}

// Reload reloads all templates from disk, preserving version history.
func (lib *PromptLibrary) Reload() error {
	lib.Templates = make(map[string]*PromptTemplate)
	return lib.LoadAll()
}

func (lib *PromptLibrary) loadFile(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var data templateYAML
	if err := yaml.Unmarshal(raw, &data); err != nil {
		return fmt.Errorf("YAML parse error in %s: %w", path, err)
	}

	if data.Template == "" {
		log.Printf("Template file %s has no 'template' field — skipping", path)
		return nil
	}

	if data.Name == "" {
		data.Name = strings.TrimSuffix(filepath.Base(path), ".yaml")
	}
	if data.Version == "" {
		data.Version = "0.0.0"
	}

	sections := make(map[string]*LibraryPromptSection)
	for secName, secData := range data.Sections {
		sections[secName] = &LibraryPromptSection{
			Name:      secName,
			Condition: secData.Condition,
			Content:   strings.TrimRight(secData.Content, "\n"),
		}
	}

	tmpl := &PromptTemplate{
		Name:         data.Name,
		Version:      data.Version,
		Description:  data.Description,
		BaseTemplate: strings.TrimRight(data.Template, "\n"),
		Sections:     sections,
		Parent:       data.Parent,
	}

	lib.Templates[data.Name] = tmpl
	histKey := data.Name + "@" + data.Version
	lib.rawYAML[histKey] = string(raw)
	lib.history[data.Name] = append(lib.history[data.Name], tmpl)
	log.Printf("Loaded template %q v%s from %s", data.Name, data.Version, filepath.Base(path))
	return nil
}

// Get returns the template for name, or an error if not found.
func (lib *PromptLibrary) Get(name string) (*PromptTemplate, error) {
	tmpl, ok := lib.Templates[name]
	if !ok {
		keys := make([]string, 0, len(lib.Templates))
		for k := range lib.Templates {
			keys = append(keys, k)
		}
		return nil, fmt.Errorf("template %q not found; available: [%s]",
			name, strings.Join(keys, ", "))
	}
	return tmpl, nil
}

// Render renders a template by name with the given variables.
func (lib *PromptLibrary) Render(
	name string,
	variables map[string]any,
	context map[string]any,
) (*RenderedPrompt, error) {
	tmpl, err := lib.Get(name)
	if err != nil {
		return nil, err
	}
	result, err := tmpl.Render(variables, context, nil)
	if err != nil {
		return nil, err
	}
	log.Printf("Rendered %q v%s | sections=%v | tokens=%d",
		name, result.TemplateVersion, result.SectionsIncluded, result.TokenCount)
	return result, nil
}

// ValidateAll validates all loaded templates and returns a list of issues.
func (lib *PromptLibrary) ValidateAll() []string {
	var issues []string
	for name, tmpl := range lib.Templates {
		if tmpl.Version == "" || tmpl.Version == "0.0.0" {
			issues = append(issues, fmt.Sprintf("[%s] Missing or default version", name))
		}
		for secName, sec := range tmpl.Sections {
			if sec.Condition == "" {
				issues = append(issues, fmt.Sprintf("[%s] Section %q has no condition", name, secName))
			}
		}
		if tmpl.Parent != "" {
			if _, ok := lib.Templates[tmpl.Parent]; !ok {
				issues = append(issues, fmt.Sprintf("[%s] References parent %q which is not loaded",
					name, tmpl.Parent))
			}
		}
	}
	return issues
}

// Diff shows a line-by-line diff between two versions of a template.
func (lib *PromptLibrary) Diff(name, versionA, versionB string) string {
	keyA := name + "@" + versionA
	keyB := name + "@" + versionB

	rawA, okA := lib.rawYAML[keyA]
	rawB, okB := lib.rawYAML[keyB]

	if !okA {
		return fmt.Sprintf("Version %q of %q not found in history.", versionA, name)
	}
	if !okB {
		return fmt.Sprintf("Version %q of %q not found in history.", versionB, name)
	}

	linesA := strings.Split(rawA, "\n")
	linesB := strings.Split(rawB, "\n")

	var diff []string
	diff = append(diff, fmt.Sprintf("--- %s v%s", name, versionA))
	diff = append(diff, fmt.Sprintf("+++ %s v%s", name, versionB))

	maxLen := len(linesA)
	if len(linesB) > maxLen {
		maxLen = len(linesB)
	}
	changed := false
	for i := 0; i < maxLen; i++ {
		var a, b string
		if i < len(linesA) {
			a = linesA[i]
		}
		if i < len(linesB) {
			b = linesB[i]
		}
		if a == b {
			continue
		}
		changed = true
		if i < len(linesA) {
			diff = append(diff, "- "+a)
		}
		if i < len(linesB) {
			diff = append(diff, "+ "+b)
		}
	}
	if !changed {
		return fmt.Sprintf("No differences between v%s and v%s.", versionA, versionB)
	}
	return strings.Join(diff, "\n")
}
