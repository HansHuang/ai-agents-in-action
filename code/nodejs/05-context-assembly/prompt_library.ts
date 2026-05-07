/**
 * Prompt Library — version-controlled YAML-based prompt template management.
 *
 * TypeScript port of code/python/05-context-assembly/prompt_library.py
 *
 * Same class names (PromptLibrary, PromptTemplate, RenderedPrompt),
 * same camelCase methods, same {variable} syntax, same YAML format.
 *
 * Requires js-yaml: npm install js-yaml @types/js-yaml
 *
 * See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
 */

import * as fs   from "fs";
import * as path from "path";
import * as yaml from "js-yaml";

import { countTokens } from "./context_budget.js";
import { evaluateCondition } from "./prompt_assembler.js";

// ---------------------------------------------------------------------------
// YAML schema type
// ---------------------------------------------------------------------------

interface SectionData {
  condition: string;
  content:   string;
}

interface TemplateData {
  name:        string;
  version:     string;
  description?: string;
  template:    string;
  sections?:   Record<string, SectionData>;
  parent?:     string;
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

export interface RenderedPrompt {
  renderedText:     string;
  templateName:     string;
  templateVersion:  string;
  sectionsIncluded: string[];
  variablesUsed:    Record<string, unknown>;
  tokenCount:       number;
  toString():       string;
}

function makeRenderedPrompt(
  renderedText:    string,
  templateName:    string,
  templateVersion: string,
  sectionsIncluded: string[],
  variablesUsed:   Record<string, unknown>,
  tokenCount:      number,
): RenderedPrompt {
  return {
    renderedText,
    templateName,
    templateVersion,
    sectionsIncluded,
    variablesUsed,
    tokenCount,
    toString() {
      return (
        `[${this.templateName} v${this.templateVersion}] ` +
        `sections=${JSON.stringify(this.sectionsIncluded)} ` +
        `tokens=${this.tokenCount}\n${this.renderedText}`
      );
    },
  };
}

interface PromptSection {
  name:      string;
  condition: string;
  content:   string;
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

export class PromptTemplate {
  readonly name:          string;
  readonly version:       string;
  readonly description:   string;
  readonly baseTemplate:  string;
  readonly sections:      Map<string, PromptSection>;
  readonly parent:        string | null;

  constructor(data: TemplateData) {
    this.name         = data.name;
    this.version      = data.version ?? "0.0.0";
    this.description  = data.description ?? "";
    this.baseTemplate = data.template;
    this.parent       = data.parent ?? null;
    this.sections     = new Map();

    for (const [secName, secData] of Object.entries(data.sections ?? {})) {
      this.sections.set(secName, {
        name:      secName,
        condition: secData.condition ?? "",
        content:   (secData.content ?? "").trimEnd(),
      });
    }
  }

  /**
   * Render the template with the given variables and optional context.
   *
   * @param variables       - Variable values for template filling.
   * @param context         - Optional supplementary variables (merged with variables).
   * @param activeSections  - If provided, use these sections instead of evaluating conditions.
   */
  render(
    variables:      Record<string, unknown>,
    context:        Record<string, unknown> = {},
    activeSections: string[] | null = null,
  ): RenderedPrompt {
    const allVars: Record<string, unknown> = { ...context, ...variables };

    // Determine active sections
    let included: string[];
    if (activeSections !== null) {
      included = activeSections;
    } else {
      included = [];
      for (const [name, section] of this.sections) {
        if (section.condition && evaluateCondition(section.condition, allVars)) {
          included.push(name);
        }
      }
    }

    // Build sections block
    const sectionsBlock = included
      .filter(n => this.sections.has(n))
      .map(n => this.sections.get(n)!.content)
      .join("\n\n");

    // Fill variables — leave missing keys as {key}
    const fillVars: Record<string, string> = { sections: sectionsBlock };
    for (const [k, v] of Object.entries(allVars)) fillVars[k] = String(v);

    let rendered = this.baseTemplate.replace(/\{(\w+)\}/g, (match, key: string) => {
      return key in fillVars ? fillVars[key] : match;
    });

    // Append sections if no placeholder
    if (!this.baseTemplate.includes("{sections}") && sectionsBlock) {
      rendered = rendered.trimEnd() + "\n\n" + sectionsBlock;
    }

    const tokenCount = countTokens(rendered);
    return makeRenderedPrompt(rendered, this.name, this.version, included, allVars, tokenCount);
  }

  /** Return all {variable} names in the template and section contents. */
  requiredVariables(): string[] {
    const seen   = new Set<string>();
    const result: string[] = [];
    const sources = [
      this.baseTemplate,
      ...[...this.sections.values()].map(s => s.content),
    ];
    for (const src of sources) {
      for (const m of src.matchAll(/\{(\w+)\}/g)) {
        const name = m[1];
        if (!seen.has(name)) { seen.add(name); result.push(name); }
      }
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// PromptLibrary
// ---------------------------------------------------------------------------

export class PromptLibrary {
  readonly promptsDir: string;
  readonly templates:  Map<string, PromptTemplate> = new Map();

  // Version history: name → PromptTemplate[]
  private readonly _history: Map<string, PromptTemplate[]> = new Map();
  // Raw YAML text: `${name}@${version}` → raw text
  private readonly _raw:     Map<string, string>            = new Map();

  constructor(promptsDir: string = "prompts/") {
    this.promptsDir = promptsDir;
    this.loadAll();
  }

  // ------------------------------------------------------------------
  // Loading
  // ------------------------------------------------------------------

  loadAll(): void {
    if (!fs.existsSync(this.promptsDir)) {
      console.warn(`Prompts directory ${this.promptsDir} does not exist`);
      return;
    }
    let count = 0;
    for (const file of this._findYamlFiles(this.promptsDir)) {
      this._loadFile(file);
      count++;
    }
    console.debug(`Loaded ${count} prompt templates from ${this.promptsDir}`);
  }

  reload(): void {
    this.templates.clear();
    this.loadAll();
    console.debug("Prompt library reloaded");
  }

  private _findYamlFiles(dir: string): string[] {
    const results: string[] = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        results.push(...this._findYamlFiles(fullPath));
      } else if (entry.isFile() && entry.name.endsWith(".yaml")) {
        results.push(fullPath);
      }
    }
    return results.sort();
  }

  private _loadFile(filePath: string): void {
    const raw  = fs.readFileSync(filePath, "utf-8");
    const data = yaml.load(raw) as TemplateData;

    if (!data?.template) {
      console.warn(`Template file ${filePath} has no 'template' field — skipping`);
      return;
    }

    const name    = data.name ?? path.basename(filePath, ".yaml");
    const version = String(data.version ?? "0.0.0");
    data.name    = name;
    data.version = version;

    const tmpl = new PromptTemplate(data);
    this.templates.set(name, tmpl);

    const historyKey = `${name}@${version}`;
    this._raw.set(historyKey, raw);
    const hist = this._history.get(name) ?? [];
    hist.push(tmpl);
    this._history.set(name, hist);
  }

  // ------------------------------------------------------------------
  // Access
  // ------------------------------------------------------------------

  get(name: string): PromptTemplate {
    const tmpl = this.templates.get(name);
    if (!tmpl) {
      throw new Error(
        `Template '${name}' not found. Available: [${[...this.templates.keys()].join(", ")}]`,
      );
    }
    return tmpl;
  }

  render(
    name:      string,
    variables: Record<string, unknown>,
    context:   Record<string, unknown> = {},
  ): RenderedPrompt {
    const tmpl   = this.get(name);
    const result = tmpl.render(variables, context);
    console.debug(
      `Rendered '${name}' v${result.templateVersion} | ` +
      `sections=${JSON.stringify(result.sectionsIncluded)} | ` +
      `tokens=${result.tokenCount}`,
    );
    return result;
  }

  // ------------------------------------------------------------------
  // Validation
  // ------------------------------------------------------------------

  validateAll(): string[] {
    const issues: string[] = [];

    for (const [name, tmpl] of this.templates) {
      if (!tmpl.version || tmpl.version === "0.0.0") {
        issues.push(`[${name}] Missing or default version`);
      }
      for (const [secName, section] of tmpl.sections) {
        if (!section.condition) {
          issues.push(`[${name}] Section '${secName}' has no condition`);
        } else {
          try {
            evaluateCondition(section.condition, {});
          } catch (err) {
            issues.push(
              `[${name}] Section '${secName}' has invalid condition ` +
              `'${section.condition}': ${(err as Error).message}`,
            );
          }
        }
      }
      if (tmpl.parent && !this.templates.has(tmpl.parent)) {
        issues.push(`[${name}] References parent '${tmpl.parent}' which is not loaded`);
      }
    }
    return issues;
  }

  // ------------------------------------------------------------------
  // Diff
  // ------------------------------------------------------------------

  diff(name: string, versionA: string, versionB: string): string {
    const rawA = this._raw.get(`${name}@${versionA}`);
    const rawB = this._raw.get(`${name}@${versionB}`);

    if (!rawA) return `Version '${versionA}' of '${name}' not found in history.`;
    if (!rawB) return `Version '${versionB}' of '${name}' not found in history.`;

    const linesA = rawA.split("\n");
    const linesB = rawB.split("\n");
    const diff: string[] = [`--- ${name} v${versionA}`, `+++ ${name} v${versionB}`];

    let maxLen = Math.max(linesA.length, linesB.length);
    for (let i = 0; i < maxLen; i++) {
      const a = linesA[i];
      const b = linesB[i];
      if (a === b) continue;
      if (a !== undefined) diff.push(`- ${a}`);
      if (b !== undefined) diff.push(`+ ${b}`);
    }

    return diff.length > 2 ? diff.join("\n") : `No differences between v${versionA} and v${versionB}.`;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function demo(promptsDir: string = "prompts/"): void {
  const library = new PromptLibrary(promptsDir);
  console.log("Loaded templates:", [...library.templates.keys()]);

  const baseVars = {
    role:            "customer support agent",
    company_name:    "Acme Corp",
    customer_name:   "Alice",
    customer_plan:   "premium",
    customer_region: "EU",
    guidelines:      "Be concise and cite sources.",
    context:         "(RAG results would appear here)",
    is_multi_turn:   "false",
  };

  console.log("\n--- support_base: premium EU customer ---");
  const r1 = library.render("support_base", baseVars);
  console.log(r1.toString());

  console.log("\n--- Validation ---");
  const issues = library.validateAll();
  if (issues.length) {
    issues.forEach(i => console.log(`  WARN: ${i}`));
  } else {
    console.log("  All templates pass validation.");
  }
}

// Uncomment to run: demo();
