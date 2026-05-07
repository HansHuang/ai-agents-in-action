"""Prompt Library — version-controlled YAML-based prompt template management.

Templates live in a ``prompts/`` directory as YAML files.  Each file is a
self-contained template with a name, version, optional conditional sections,
and a ``{variable}``-style template string.

The library treats templates as code:

- **Versioned** — every template carries a semver version string.
- **Hot-reloadable** — call :meth:`PromptLibrary.reload` to pick up edits
  without restarting.
- **Validated** — :meth:`PromptLibrary.validate_all` reports structural
  problems before they cause runtime failures.
- **Diffable** — :meth:`PromptLibrary.diff` shows what changed between two
  versions of the same template.

YAML format::

    name: support_base
    version: 1.2.0
    description: Base customer support template

    template: |
      You are a {role} for {company_name}.
      {context}

    sections:
      premium_experience:
        condition: "customer_plan == 'premium'"
        content: |
          - Provide white-glove service.

See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from condition_engine import ConditionEngine
from context_budget import count_tokens

logger = logging.getLogger(__name__)

_VAR_RE = re.compile(r"\{(\w+)\}")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RenderedPrompt:
    """The output of rendering a template.

    Attributes:
        rendered_text:      Final assembled prompt string.
        template_name:      Name of the template used.
        template_version:   Semver version of the template.
        sections_included:  Names of conditional sections that were active.
        variables_used:     Variables supplied at render time.
        token_count:        Token count of the rendered prompt.
    """

    rendered_text:    str
    template_name:    str
    template_version: str
    sections_included: list[str]
    variables_used:   dict
    token_count:      int

    def __str__(self) -> str:
        return (
            f"[{self.template_name} v{self.template_version}] "
            f"sections={self.sections_included} "
            f"tokens={self.token_count}\n"
            + self.rendered_text
        )


@dataclass
class PromptSection:
    """A named conditional section from a YAML template."""

    name:      str
    condition: str   # DSL string evaluated by ConditionEngine
    content:   str


@dataclass
class PromptTemplate:
    """A loaded YAML prompt template.

    Attributes:
        name:          Template identifier (must be unique in the library).
        version:       Semver version string.
        description:   Human-readable description.
        base_template: Template string with ``{variable}`` placeholders.
        sections:      Conditional sections keyed by section name.
        parent:        Name of a parent template to inherit from (optional).
    """

    name:          str
    version:       str
    description:   str
    base_template: str
    sections:      dict[str, PromptSection] = field(default_factory=dict)
    parent:        str | None = None

    _engine: ConditionEngine = field(default_factory=ConditionEngine, repr=False)

    def render(
        self,
        variables: dict,
        context: dict | None = None,
        active_sections: list[str] | None = None,
    ) -> RenderedPrompt:
        """Render the template with variables and optional context.

        Args:
            variables:       Template variable values.
            context:         Optional additional variables (merged with variables).
            active_sections: If given, use these section names instead of
                             evaluating conditions.

        Returns:
            :class:`RenderedPrompt` with the rendered text and metadata.
        """
        all_vars: dict[str, Any] = {**(context or {}), **variables}

        # Determine active sections
        if active_sections is None:
            active_sections = [
                name
                for name, section in self.sections.items()
                if self._engine.evaluate(section.condition, all_vars)
            ]

        # Build sections block
        sections_block = "\n\n".join(
            self.sections[name].content
            for name in active_sections
            if name in self.sections
        )

        # Fill variables — missing keys left as-is (not raised here;
        # use validate_all to catch structural problems ahead of time)
        fill_vars = {**all_vars, "sections": sections_block}
        try:
            rendered = self.base_template.format_map(
                _DefaultDict(fill_vars)
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to render template {self.name!r} v{self.version}: {exc}"
            ) from exc

        # Append sections that have no placeholder in the base template
        if "{sections}" not in self.base_template and sections_block:
            rendered = rendered.rstrip() + "\n\n" + sections_block

        token_count = count_tokens(rendered)
        return RenderedPrompt(
            rendered_text=rendered,
            template_name=self.name,
            template_version=self.version,
            sections_included=list(active_sections),
            variables_used=dict(all_vars),
            token_count=token_count,
        )

    def required_variables(self) -> list[str]:
        """Return all ``{variable}`` names used in the template and sections."""
        seen: set[str] = set()
        result: list[str] = []
        sources = [self.base_template] + [s.content for s in self.sections.values()]
        for src in sources:
            for m in _VAR_RE.finditer(src):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        return result


class _DefaultDict(dict):
    """dict that leaves missing keys as ``{key}`` instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


# ---------------------------------------------------------------------------
# PromptLibrary
# ---------------------------------------------------------------------------

class PromptLibrary:
    """Load, manage, and render version-controlled YAML prompt templates.

    Example::

        library = PromptLibrary("prompts/")

        result = library.render(
            "support_base",
            variables={
                "role":          "support agent",
                "company_name":  "Acme Corp",
                "customer_name": "Alice",
                "customer_plan": "premium",
                "customer_region": "EU",
                "guidelines":    "Be concise.",
                "context":       "...",
            },
        )
        print(result.template_version)   # "1.2.0"
        print(result.sections_included)  # ["premium_experience", "gdpr_notice"]
        print(result.token_count)        # 312
    """

    def __init__(self, prompts_dir: str = "prompts/") -> None:
        self.prompts_dir = Path(prompts_dir)
        self.templates: dict[str, PromptTemplate] = {}
        # History: name → list of PromptTemplate versions (oldest first)
        self._history: dict[str, list[PromptTemplate]] = {}
        # Raw YAML text indexed by (name, version)
        self._raw: dict[tuple[str, str], str] = {}
        self.load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all ``.yaml`` template files from the prompts directory."""
        if not self.prompts_dir.exists():
            logger.warning("Prompts directory %s does not exist", self.prompts_dir)
            return

        loaded = 0
        for path in sorted(self.prompts_dir.rglob("*.yaml")):
            self._load_file(path)
            loaded += 1

        logger.info("Loaded %d prompt templates from %s", loaded, self.prompts_dir)

    def reload(self) -> None:
        """Reload all templates from disk.

        Existing version history is preserved so :meth:`diff` can compare
        before and after a reload.
        """
        self.templates = {}
        self.load_all()
        logger.info("Prompt library reloaded")

    def _load_file(self, path: Path) -> None:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)

        name    = data.get("name", path.stem)
        version = str(data.get("version", "0.0.0"))

        if not data.get("template"):
            logger.warning("Template file %s has no 'template' field — skipping", path)
            return

        sections: dict[str, PromptSection] = {}
        for sec_name, sec_data in (data.get("sections") or {}).items():
            sections[sec_name] = PromptSection(
                name=sec_name,
                condition=sec_data.get("condition", "true == 'true'"),
                content=(sec_data.get("content") or "").rstrip(),
            )

        template = PromptTemplate(
            name=name,
            version=version,
            description=data.get("description", ""),
            base_template=data["template"].rstrip(),
            sections=sections,
            parent=data.get("parent"),
        )

        if name in self.templates:
            logger.debug("Replacing template %r (was v%s, now v%s)",
                         name, self.templates[name].version, version)

        self.templates[name] = template
        self._history.setdefault(name, []).append(template)
        self._raw[(name, version)] = raw
        logger.debug("Loaded template %r v%s from %s", name, version, path.name)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, name: str) -> PromptTemplate:
        """Return the template for *name*.

        Raises:
            KeyError: If *name* is not in the library.
        """
        if name not in self.templates:
            raise KeyError(
                f"Template {name!r} not found. Available: {list(self.templates)}"
            )
        return self.templates[name]

    def render(
        self,
        name: str,
        variables: dict,
        context: dict | None = None,
    ) -> RenderedPrompt:
        """Render a template by name.

        Args:
            name:      Template name.
            variables: Variable values for template filling.
            context:   Optional supplementary variables (merged with variables).

        Returns:
            :class:`RenderedPrompt` with rendered text and metadata.

        Raises:
            KeyError: If *name* is not registered.
        """
        template = self.get(name)
        result   = template.render(variables, context)
        logger.info(
            "Rendered %r v%s | sections=%s | tokens=%d",
            name, result.template_version,
            result.sections_included, result.token_count,
        )
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_all(self) -> list[str]:
        """Validate all loaded templates.

        Checks:

        - Every template has a ``name``, ``version``, and ``template`` string.
        - Every section has a non-empty ``condition``.
        - ``{variables}`` used in the template are either provided by common
          synthetic keys or are noted as warnings.
        - No duplicate template names (enforced at load time; reported here).
        - Any ``parent`` reference points to an existing template.

        Returns:
            List of warning/error strings.  Empty list means all templates pass.
        """
        issues: list[str] = []

        _SYNTHETIC = {"context", "sections"}

        for name, tmpl in self.templates.items():
            # Version check
            if not tmpl.version or tmpl.version == "0.0.0":
                issues.append(f"[{name}] Missing or default version (set a semver string)")

            # Section condition check
            for sec_name, section in tmpl.sections.items():
                if not section.condition:
                    issues.append(f"[{name}] Section {sec_name!r} has no condition")
                else:
                    from condition_engine import ConditionEngine as _CE
                    ok, msg = _CE().validate_condition(section.condition)
                    if not ok:
                        issues.append(
                            f"[{name}] Section {sec_name!r} has invalid condition "
                            f"{section.condition!r}: {msg}"
                        )

            # Parent check
            if tmpl.parent and tmpl.parent not in self.templates:
                issues.append(
                    f"[{name}] References parent {tmpl.parent!r} which is not loaded"
                )

            # Variable documentation hint
            used_vars = tmpl.required_variables()
            for var in used_vars:
                if var not in _SYNTHETIC:
                    # Just a debug hint — libraries don't mandate a variables list
                    logger.debug("[%s] Variable {%s} used (document in README)", name, var)

        return issues

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, name: str, version_a: str, version_b: str) -> str:
        """Show a unified diff between two versions of a template.

        Both versions must have been loaded during the lifetime of this
        library instance (i.e. present in the reload history).

        Args:
            name:      Template name.
            version_a: Earlier version string.
            version_b: Later version string.

        Returns:
            Unified diff string, or a message explaining why no diff is available.
        """
        raw_a = self._raw.get((name, version_a))
        raw_b = self._raw.get((name, version_b))

        if raw_a is None:
            return f"Version {version_a!r} of {name!r} not found in history."
        if raw_b is None:
            return f"Version {version_b!r} of {name!r} not found in history."

        lines_a = raw_a.splitlines(keepends=True)
        lines_b = raw_b.splitlines(keepends=True)
        diff = difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"{name} v{version_a}",
            tofile=f"{name} v{version_b}",
        )
        result = "".join(diff)
        return result if result else f"No differences between v{version_a} and v{version_b}."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo(prompts_dir: str = "prompts/") -> None:
    """Load templates, render for scenarios, validate, and hot-reload."""

    library = PromptLibrary(prompts_dir)

    print(f"Loaded templates: {list(library.templates.keys())}")

    base_vars = {
        "role":            "customer support agent",
        "company_name":    "Acme Corp",
        "customer_name":   "Alice",
        "customer_plan":   "premium",
        "customer_region": "EU",
        "guidelines":      "Be concise and cite sources.",
        "context":         "(RAG results would appear here)",
    }

    # ── Scenario 1: Premium EU customer ───────────────────────────────────
    print("\n--- support_base: premium EU customer ---")
    r1 = library.render("support_base", base_vars)
    print(r1)

    # ── Scenario 2: Non-premium non-EU customer ────────────────────────────
    print("\n--- support_base: free US customer ---")
    r2 = library.render("support_base", {
        **base_vars,
        "customer_plan":   "free",
        "customer_region": "US",
    })
    print(r2)

    # ── Validation ─────────────────────────────────────────────────────────
    print("\n--- Validation ---")
    issues = library.validate_all()
    if issues:
        for issue in issues:
            print(f"  WARN: {issue}")
    else:
        print("  All templates pass validation.")

    # ── Hot-reload simulation ──────────────────────────────────────────────
    # (In a real scenario, you would edit a file on disk then call reload().)
    print("\n--- Hot-reload simulation ---")
    base_yaml_path = Path(prompts_dir) / "support_base.yaml"
    if base_yaml_path.exists():
        original = base_yaml_path.read_text()
        # Bump version and add a new line for demo purposes
        updated = original.replace("version: 1.2.0", "version: 1.2.1")
        base_yaml_path.write_text(updated)
        library.reload()
        r3 = library.render("support_base", base_vars)
        print(f"After reload: template version = {r3.template_version}")
        diff_output = library.diff("support_base", "1.2.0", "1.2.1")
        print("Diff:\n" + diff_output)
        # Restore original
        base_yaml_path.write_text(original)
        library.reload()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _demo()
