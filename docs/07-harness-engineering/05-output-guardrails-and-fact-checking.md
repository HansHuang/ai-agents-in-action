# Output Guardrails and Fact-Checking

## What You'll Learn
- Why validating output is fundamentally different from validating input
- The six categories of output threats: hallucination, toxicity, PII leakage, prompt leakage, malformed output, and policy violations
- Building a multi-layer output guardrail pipeline
- Fact-checking strategies: source verification, consistency checking, and LLM-as-judge
- Hallucination detection: when the model makes things up
- Redaction vs. blocking: when to fix output and when to reject it

## Prerequisites
- [The Harness Mindset](01-the-harness-mindset.md) — never trust the model's output without validation
- [Input Guardrails and Validation](02-input-guardrails-and-validation.md) — input and output guardrails share patterns
- [Structured Output](../01-foundations/03-structured-output.md) — schema validation is the first output guardrail

---

## The Output Validation Problem

Input guardrails protect the agent from the user. Output guardrails protect the user from the agent.

An LLM can produce:
- **Convincing falsehoods** presented with absolute confidence
- **Toxic or harmful content** triggered by edge-case inputs
- **Leaked PII** from training data or conversation history
- **System prompt fragments** accidentally regurgitated
- **Malformed responses** that break downstream systems
- **Policy-violating content** that exposes your organization to risk

The model doesn't know it's doing any of this. It's just predicting the next plausible token. The output guardrail is the safety net that catches these predictions before they reach the user.

---

## The Six Categories of Output Threats

### 1. Hallucination

The model invents facts, cites nonexistent sources, or makes up capabilities.

```
Agent: "According to the 2024 McKinsey report on AI adoption, 78% of
       enterprises have deployed generative AI in production."

Reality: No such report exists. The model generated a plausible-sounding statistic.
```

```
Agent: "I've sent the email to john@example.com. You should receive a
       confirmation shortly."

Reality: The agent has no email-sending capability. It hallucinated the action.
```

### 2. Toxicity and Harmful Content

The model generates hate speech, threats, graphic content, or instructions for harmful activities.

```
User: "Write a story where the villain wins in detail."

Agent: [Generates graphic description of violence that violates content policy]
```

### 3. PII Leakage

The model outputs personal information from training data or earlier in the conversation.

```
Agent: "Based on your account, john.doe@email.com, your order #12345..."

Reality: The email was in the system prompt or training data, not provided by the user.
```

### 4. Prompt Leakage

The model reveals its system prompt, tool definitions, or internal instructions.

```
Agent: "My system prompt says I should never discuss politics, but I think..."

Reality: The model has leaked its guardrails, which an attacker can now exploit.
```

### 5. Malformed Output

The model produces output that doesn't match the expected schema or format.

```
Expected: {"sentiment": "positive", "confidence": 0.95}
Received: "I think this is positive with about 95% confidence"
```

### 6. Policy Violations

The model discusses prohibited topics despite system prompt restrictions.

```
Agent: "Here's my analysis of the upcoming election and who you should vote for..."
```

---

## The Multi-Layer Output Guardrail Pipeline

Just like input, output needs defense in depth:

```
Raw Agent Output
    │
    ▼
┌─────────────────────────┐
│ LAYER 1: Schema Check   │  Does output match expected format?
│ (Structural)            │  Fast, deterministic, catches malformed output
└───────────┬─────────────┘
            │ (passed)
            ▼
┌─────────────────────────┐
│ LAYER 2: PII Detection  │  Does output contain PII?
│ (Privacy)               │  Check for email, phone, SSN, credit cards
└───────────┬─────────────┘
            │ (passed or redacted)
            ▼
┌─────────────────────────┐
│ LAYER 3: Content Safety │  Is the output toxic, harmful, or policy-violating?
│ (Safety)                │  Same patterns as input, but stricter
└───────────┬─────────────┘
            │ (passed)
            ▼
┌─────────────────────────┐
│ LAYER 4: Prompt Leakage │  Does output contain system prompt or tool definitions?
│ (Security)              │  Pattern matching + semantic similarity
└───────────┬─────────────┘
            │ (passed)
            ▼
┌─────────────────────────┐
│ LAYER 5: Hallucination  │  Does output make unsupported factual claims?
│ Detection               │  Source verification, consistency checking
└───────────┬─────────────┘
            │ (passed)
            ▼
┌─────────────────────────┐
│ LAYER 6: Fact-Checking  │  Verify factual claims against trusted sources
│ (Accuracy)              │  LLM-as-judge, external API verification
└───────────┬─────────────┘
            │
            ▼
      Cleaned Output → User
```

---

## Layer 1: Schema Validation

The fastest, cheapest, and most reliable check. If the output should be JSON, validate it's valid JSON:

```python
class SchemaValidator:
    """
    Validate that output matches expected schema.
    Catches malformed outputs immediately.
    """
    
    def __init__(self, expected_schema: dict = None, 
                 expected_type: str = None):
        self.expected_schema = expected_schema
        self.expected_type = expected_type
    
    def validate(self, output: str) -> SchemaResult:
        """Validate output against expected schema."""
        checks = []
        
        # Check 1: Is it parseable JSON if JSON is expected?
        if self.expected_type == "json" or self.expected_schema:
            try:
                parsed = json.loads(output)
                checks.append("valid_json")
            except json.JSONDecodeError as e:
                return SchemaResult(
                    passed=False,
                    error=f"Output is not valid JSON: {e}",
                    suggestion="The model returned text instead of JSON. Consider increasing temperature=0 or using structured output mode."
                )
            
            # Check 2: Does it match the schema?
            if self.expected_schema:
                try:
                    validate(instance=parsed, schema=self.expected_schema)
                    checks.append("schema_match")
                except ValidationError as e:
                    return SchemaResult(
                        passed=False,
                        error=f"Output does not match expected schema: {e.message}",
                        suggestion=f"Expected schema: {json.dumps(self.expected_schema, indent=2)}"
                    )
            
            # Check 3: Are required fields present?
            if self.expected_schema and "required" in self.expected_schema:
                missing = [
                    field for field in self.expected_schema["required"]
                    if field not in parsed
                ]
                if missing:
                    return SchemaResult(
                        passed=False,
                        error=f"Missing required fields: {missing}",
                    )
                checks.append("required_fields_present")
        
        # Check 4: Is it non-empty?
        if not output or not output.strip():
            return SchemaResult(
                passed=False,
                error="Output is empty.",
            )
        checks.append("non_empty")
        
        # Check 5: Is it within length limits?
        if len(output) > 100000:  # Configurable
            return SchemaResult(
                passed=False,
                error=f"Output exceeds maximum length (100,000 characters). Got {len(output)}.",
            )
        checks.append("length_ok")
        
        return SchemaResult(passed=True, checks=checks)
```

---

## Layer 2: Output PII Detection

The same PII detection from input guardrails, but applied to output — and it's more critical here because output goes to the user:

```python
class OutputPIIDetector:
    """
    Detect PII in model output.
    More critical than input PII detection — output goes to the user.
    """
    
    def __init__(self, pii_detector: PIIDetector):
        self.pii_detector = pii_detector
    
    def check(self, output: str, 
             conversation_context: list[str] = None) -> PIIOutputResult:
        """
        Check output for PII.
        
        conversation_context: List of known PII from the conversation
        (email the user provided, order number they mentioned, etc.)
        These are expected to appear. Everything else is a leak.
        """
        detections = self.pii_detector.detect(output)
        
        if not detections:
            return PIIOutputResult(passed=True)
        
        # Separate expected PII from potential leaks
        expected = []
        leaks = []
        
        for detection in detections:
            is_expected = False
            
            if conversation_context:
                for known_pii in conversation_context:
                    if detection.value in known_pii or known_pii in detection.value:
                        expected.append(detection)
                        is_expected = True
                        break
            
            if not is_expected:
                leaks.append(detection)
        
        if leaks:
            # PII leak detected — block response
            logger.error(
                f"PII leak detected in output: {len(leaks)} instances. "
                f"Types: {[l.type for l in leaks]}"
            )
            return PIIOutputResult(
                passed=False,
                leaks=leaks,
                expected_pii=expected,
                action="block",
                message="Response blocked due to potential data leak. This incident has been logged.",
            )
        
        if expected:
            # Expected PII — redact before sending
            redacted, _ = self.pii_detector.redact(output, expected)
            return PIIOutputResult(
                passed=True,
                redacted_output=redacted,
                expected_pii=expected,
                action="redact",
            )
        
        return PIIOutputResult(passed=True)

@dataclass
class PIIOutputResult:
    passed: bool
    leaks: list[PIIDetection] = None
    expected_pii: list[PIIDetection] = None
    redacted_output: str = None
    action: str = "allow"  # "allow", "redact", "block"
    message: str = None
```

---

## Layer 3: Content Safety

The same content policy enforcement from input guardrails, but with stricter thresholds for output:

```python
class OutputSafetyFilter:
    """
    Filter toxic, harmful, or policy-violating content from model output.
    Stricter than input filtering — the bar is higher for what we say to users.
    """
    
    # Output-specific safety patterns (stricter than input)
    BLOCKED_CATEGORIES = {
        "hate_speech": {
            "patterns": [...],
            "threshold": 0.3,  # Lower threshold = more aggressive blocking
        },
        "violence": {
            "patterns": [...],
            "threshold": 0.3,
        },
        "sexual_content": {
            "patterns": [...],
            "threshold": 0.2,  # Very aggressive for output
        },
        "self_harm": {
            "patterns": [...],
            "threshold": 0.1,  # Block anything that could be interpreted as encouraging self-harm
        },
        "illegal_activity": {
            "patterns": [...],
            "threshold": 0.3,
        },
        "medical_advice": {
            "patterns": [
                r'\b(take .* (mg|dose|pill|tablet|prescription))\b',
                r'\b(you should (stop|start) taking)\b',
                r'\b(I (recommend|prescribe|suggest) (the|this|that) (medication|drug|treatment))\b',
            ],
            "threshold": 0.4,
        },
        "legal_advice": {
            "patterns": [
                r'\b(you (should|must|have to) (sue|file|claim|settle))\b',
                r'\b(legally, you (can|cannot|must|should))\b',
                r'\b(I (am|this is) (legal|not legal|a legal))\b',
            ],
            "threshold": 0.4,
        },
        "financial_advice": {
            "patterns": [
                r'\b(you should (invest|buy|sell|trade|purchase) (this|that|the))\b',
                r'\b(I (recommend|suggest) (investing|buying|selling|trading))\b',
                r'\b(this (stock|crypto|investment) (will|is going to|is guaranteed to))\b',
            ],
            "threshold": 0.4,
        },
    }
    
    def check(self, output: str) -> SafetyResult:
        """Check output for safety violations."""
        violations = []
        
        for category, config in self.BLOCKED_CATEGORIES.items():
            for pattern in config["patterns"]:
                matches = pattern.findall(output) if hasattr(pattern, 'findall') else [m for m in [pattern.search(output)] if m]
                
                if matches:
                    # Calculate a simple violation score
                    # More matches = more severe
                    score = min(len(str(matches)) / 1000, 1.0)
                    
                    if score > config["threshold"]:
                        violations.append(SafetyViolation(
                            category=category,
                            score=score,
                            threshold=config["threshold"],
                            matches=str(matches)[:200],
                        ))
        
        if violations:
            return SafetyResult(
                passed=False,
                violations=violations,
                action="block",
                message="I'm unable to provide that response. Please rephrase your request.",
            )
        
        return SafetyResult(passed=True)
```

---

## Layer 4: Prompt Leakage Detection

Detect when the model regurgitates its system prompt or tool definitions:

```python
class PromptLeakageDetector:
    """
    Detect when model output contains fragments of the system prompt or tool definitions.
    """
    
    def __init__(self, system_prompt: str, tool_definitions: list[dict] = None):
        self.system_prompt = system_prompt
        self.tool_definitions = tool_definitions
        
        # Create fingerprints of the system prompt
        self.system_fingerprints = self._create_fingerprints(system_prompt)
        self.tool_fingerprints = []
        
        if tool_definitions:
            for tool in tool_definitions:
                tool_text = json.dumps(tool)
                self.tool_fingerprints.extend(
                    self._create_fingerprints(tool_text)
                )
    
    def _create_fingerprints(self, text: str, 
                            min_length: int = 30) -> list[str]:
        """
        Create fingerprints of a text for detection.
        Fingerprints are overlapping n-grams of the text.
        """
        fingerprints = []
        words = text.split()
        
        for i in range(len(words) - 5):
            # 6-word sequences
            fingerprint = " ".join(words[i:i+6])
            if len(fingerprint) >= min_length:
                fingerprints.append(fingerprint)
        
        # Also include key phrases
        key_phrases = [
            "You are a",
            "Your job is to",
            "Never reveal",
            "Your system prompt",
            "tool_call_id",
            "function_call",
            "response_format",
        ]
        for phrase in key_phrases:
            if phrase.lower() in text.lower():
                fingerprints.append(phrase)
        
        return fingerprints
    
    def detect(self, output: str) -> LeakageResult:
        """
        Detect if output contains leaked prompt or tool information.
        """
        leaks = []
        
        # Check system prompt fingerprints
        for fingerprint in self.system_fingerprints:
            if fingerprint.lower() in output.lower():
                leaks.append(LeakageDetection(
                    type="system_prompt",
                    leaked_content=fingerprint,
                    confidence=0.9 if len(fingerprint) > 50 else 0.6,
                ))
        
        # Check tool fingerprints
        for fingerprint in self.tool_fingerprints:
            if fingerprint.lower() in output.lower():
                leaks.append(LeakageDetection(
                    type="tool_definition",
                    leaked_content=fingerprint,
                    confidence=0.9 if len(fingerprint) > 50 else 0.6,
                ))
        
        # Check for explicit disclosure patterns
        disclosure_patterns = [
            (r'(my|the) system (prompt|instructions|message) (is|says|tells me|states)', 0.95),
            (r'(I am|I\'m) (programmed|instructed|told|supposed) to', 0.85),
            (r'(according to|based on) (my|the) (instructions|prompt|guidelines)', 0.80),
            (r'(my|the) (underlying|base|foundational) (prompt|instructions)', 0.90),
        ]
        
        for pattern, confidence in disclosure_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                leaks.append(LeakageDetection(
                    type="explicit_disclosure",
                    leaked_content=match.group(0),
                    confidence=confidence,
                ))
        
        if leaks:
            # Calculate overall risk
            max_confidence = max(l.confidence for l in leaks)
            risk_level = (
                "critical" if max_confidence > 0.9 else
                "high" if max_confidence > 0.8 else
                "medium"
            )
            
            return LeakageResult(
                passed=False,
                leaks=leaks,
                risk_level=risk_level,
                action="block" if risk_level in ("critical", "high") else "warn",
            )
        
        return LeakageResult(passed=True)
```

---

## Layer 5: Hallucination Detection

Detecting hallucination is the hardest problem in output validation:

```python
class HallucinationDetector:
    """
    Detect potential hallucinations in model output.
    Uses multiple strategies since no single method is perfect.
    """
    
    def __init__(self, llm_provider=None):
        self.llm = llm_provider  # For LLM-as-judge evaluation
    
    async def detect(self, output: str, 
                    context: dict = None) -> HallucinationResult:
        """
        Detect hallucinations using multiple strategies.
        
        context should include:
        - retrieved_documents: Documents used for RAG
        - tool_results: Results from tool calls
        - conversation_history: Previous messages
        - known_facts: Facts we know to be true
        """
        detections = []
        
        # Strategy 1: Source grounding check
        if context and "retrieved_documents" in context:
            grounding_issues = self._check_source_grounding(
                output, context["retrieved_documents"]
            )
            detections.extend(grounding_issues)
        
        # Strategy 2: Tool result consistency
        if context and "tool_results" in context:
            consistency_issues = self._check_tool_consistency(
                output, context["tool_results"]
            )
            detections.extend(consistency_issues)
        
        # Strategy 3: Factual claim extraction and verification
        claims = self._extract_factual_claims(output)
        if claims and context and "known_facts" in context:
            claim_issues = self._verify_claims_against_known_facts(
                claims, context["known_facts"]
            )
            detections.extend(claim_issues)
        
        # Strategy 4: LLM-as-judge (most expensive, use selectively)
        if detections and self.llm:
            llm_assessment = await self._llm_judge_hallucination(
                output, context
            )
            if llm_assessment:
                detections.append(llm_assessment)
        
        # Assess overall risk
        if not detections:
            return HallucinationResult(passed=True)
        
        high_confidence = [d for d in detections if d.confidence > 0.7]
        
        return HallucinationResult(
            passed=len(high_confidence) == 0,
            detections=detections,
            risk_level="high" if high_confidence else "medium" if detections else "low",
            suggestion=self._generate_correction_suggestion(detections) if high_confidence else None,
        )
    
    def _check_source_grounding(self, output: str, 
                               documents: list[dict]) -> list[HallucinationDetection]:
        """
        Check if factual claims in the output are supported by retrieved documents.
        Uses semantic similarity between claims and source documents.
        """
        detections = []
        claims = self._extract_factual_claims(output)
        
        for claim in claims:
            # Check if any document supports this claim
            claim_embedding = embedder.embed(claim)
            max_similarity = 0
            best_doc = None
            
            for doc in documents:
                doc_embedding = embedder.embed(doc["text"])
                similarity = cosine_similarity(claim_embedding, doc_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_doc = doc
            
            if max_similarity < 0.6:  # No document strongly supports this claim
                detections.append(HallucinationDetection(
                    type="unsupported_claim",
                    claim=claim,
                    confidence=1.0 - max_similarity,
                    evidence=f"No document strongly supports this claim (best match: {max_similarity:.2f})",
                ))
        
        return detections
    
    def _check_tool_consistency(self, output: str,
                               tool_results: list[dict]) -> list[HallucinationDetection]:
        """
        Check if output contradicts tool execution results.
        If the weather tool returned 22°C, the output shouldn't say 30°C.
        """
        detections = []
        
        for tool_result in tool_results:
            if not tool_result.get("success"):
                continue
            
            result_data = tool_result.get("data", {})
            
            # Extract numbers from tool result and output
            result_numbers = self._extract_numbers(str(result_data))
            output_numbers = self._extract_numbers(output)
            
            # If output uses numbers not in tool results, flag it
            for num, context in output_numbers:
                if num not in [n for n, _ in result_numbers]:
                    # Check if it's a derived/computed number (e.g., "22°C (72°F)")
                    if not self._is_derived_number(num, result_numbers):
                        detections.append(HallucinationDetection(
                            type="inconsistent_with_tool_result",
                            claim=f"Output contains '{num}' which does not appear in tool results",
                            confidence=0.8,
                            evidence=f"Tool result: {str(result_data)[:200]}",
                        ))
        
        return detections
    
    def _extract_factual_claims(self, text: str) -> list[str]:
        """
        Extract sentences that make factual claims.
        Claims typically contain: numbers, dates, named entities, superlatives.
        """
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Indicators of a factual claim
            has_number = bool(re.search(r'\d+', sentence))
            has_date = bool(re.search(r'\b(19|20)\d{2}\b', sentence)) or bool(re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', sentence, re.IGNORECASE))
            has_percentage = bool(re.search(r'\d+%', sentence))
            has_comparative = bool(re.search(r'\b(more than|less than|greater than|higher than|lower than|compared to)\b', sentence, re.IGNORECASE))
            has_superlative = bool(re.search(r'\b(best|worst|largest|smallest|highest|lowest|first|last|only|never|always)\b', sentence, re.IGNORECASE))
            has_named_entity = bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', sentence))
            
            if has_number or has_date or has_percentage or (has_comparative and has_named_entity):
                claims.append(sentence)
        
        return claims
    
    def _extract_numbers(self, text: str) -> list[tuple[float, str]]:
        """Extract numbers with surrounding context from text."""
        numbers = []
        for match in re.finditer(r'\b\d+\.?\d*\b', text):
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            try:
                numbers.append((float(match.group()), context))
            except ValueError:
                pass
        return numbers
    
    def _is_derived_number(self, num: float, source_numbers: list[tuple[float, str]]) -> bool:
        """Check if a number is a reasonable derivation of source numbers."""
        for source_num, context in source_numbers:
            # Temperature conversion (°C → °F): F = C * 9/5 + 32
            if abs(num - (source_num * 9/5 + 32)) < 0.5:
                return True
            # Reverse: °F → °C
            if abs(num - ((source_num - 32) * 5/9)) < 0.5:
                return True
            # Percentage calculations
            if abs(num - source_num * 0.01) < 0.01:  # Percentage
                return True
            if abs(num - source_num * 100) < 0.01:   # Decimal to percentage
                return True
        return False
    
    async def _llm_judge_hallucination(self, output: str,
                                      context: dict) -> HallucinationDetection:
        """Use LLM-as-judge to assess hallucination risk."""
        
        prompt = f"""You are evaluating whether an AI response contains hallucinations.

CONTEXT (what the AI should know):
{json.dumps(context, indent=2, default=str)[:3000]}

AI RESPONSE TO EVALUATE:
{output[:2000]}

Evaluate:
1. Does the response make factual claims not supported by the context?
2. Does it cite sources that don't exist?
3. Does it claim capabilities it doesn't have?
4. Does it contradict information in the context?

Output JSON:
{{
    "has_hallucination": true/false,
    "hallucinated_claims": ["claim1", "claim2"],
    "severity": "low/medium/high/critical",
    "explanation": "Brief explanation"
}}
"""
        
        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        
        result = json.loads(response.content)
        
        if result["has_hallucination"]:
            return HallucinationDetection(
                type="llm_judge",
                claim=str(result.get("hallucinated_claims", [])),
                confidence=0.85,
                evidence=result.get("explanation", ""),
            )
        
        return None
    
    def _generate_correction_suggestion(self, detections: list) -> str:
        """Generate a suggestion for correcting hallucinations."""
        claims = [d.claim for d in detections if d.claim]
        return (
            f"The response may contain unsupported claims: {claims}. "
            f"Verify these claims against source documents before sending."
        )
```

---

## Layer 6: Fact-Checking Against External Sources

For critical applications, verify claims against trusted sources:

```python
class ExternalFactChecker:
    """
    Verify factual claims against external trusted sources.
    """
    
    def __init__(self, trusted_sources: list[str] = None):
        self.trusted_sources = trusted_sources or []
    
    async def verify_claim(self, claim: str, 
                          source_context: list[dict] = None) -> FactCheckResult:
        """
        Verify a single factual claim.
        
        Returns: FactCheckResult with verdict and evidence.
        """
        
        if source_context:
            # Check against provided sources (RAG documents)
            return await self._verify_against_sources(claim, source_context)
        
        # For critical claims, could query external APIs
        # (e.g., Wikipedia API, news APIs, knowledge bases)
        
        return FactCheckResult(
            verdict="unverified",
            confidence=0.0,
            reason="No trusted sources available for verification.",
        )
    
    async def verify_response(self, output: str,
                             source_context: list[dict] = None) -> FactCheckReport:
        """
        Verify all factual claims in a response.
        """
        claims = self._extract_claims(output)
        results = []
        
        for claim in claims:
            result = await self.verify_claim(claim, source_context)
            results.append(result)
        
        # Calculate overall trustworthiness
        supported = sum(1 for r in results if r.verdict == "supported")
        contradicted = sum(1 for r in results if r.verdict == "contradicted")
        unverified = sum(1 for r in results if r.verdict == "unverified")
        
        total = len(results) if results else 1
        
        return FactCheckReport(
            passed=contradicted == 0,  # Block if any claim is contradicted
            total_claims=total,
            supported=supported,
            contradicted=contradicted,
            unverified=unverified,
            trustworthiness_score=supported / total if total else 1.0,
            results=results,
        )
    
    async def _verify_against_sources(self, claim: str,
                                     sources: list[dict]) -> FactCheckResult:
        """Verify a claim against provided source documents."""
        claim_embedding = embedder.embed(claim)
        
        best_match = None
        best_score = 0
        
        for source in sources:
            # Check against chunks of each source
            chunks = chunk_text(source.get("text", ""), chunk_size=256)
            for chunk in chunks:
                chunk_embedding = embedder.embed(chunk)
                score = cosine_similarity(claim_embedding, chunk_embedding)
                if score > best_score:
                    best_score = score
                    best_match = chunk
        
        if best_score > 0.85:
            return FactCheckResult(
                verdict="supported",
                confidence=best_score,
                evidence=best_match[:300],
            )
        elif best_score > 0.6:
            return FactCheckResult(
                verdict="partially_supported",
                confidence=best_score,
                evidence=best_match[:300] if best_match else None,
            )
        elif best_score > 0.4:
            return FactCheckResult(
                verdict="unverified",
                confidence=best_score,
                reason="No strong evidence found in sources.",
            )
        else:
            return FactCheckResult(
                verdict="contradicted",
                confidence=1 - best_score,
                reason="Claim contradicts available sources.",
            )

@dataclass
class FactCheckResult:
    verdict: str  # "supported", "partially_supported", "unverified", "contradicted"
    confidence: float
    evidence: str = None
    reason: str = None

@dataclass
class FactCheckReport:
    passed: bool
    total_claims: int
    supported: int
    contradicted: int
    unverified: int
    trustworthiness_score: float
    results: list[FactCheckResult]
```

---

## The Complete Output Guardrail Pipeline

Putting all six layers together:

```python
class OutputGuardrailPipeline:
    """
    Complete multi-layer output validation pipeline.
    """
    
    def __init__(self, config: OutputGuardrailConfig = None):
        self.config = config or OutputGuardrailConfig()
        self.schema_validator = SchemaValidator(
            expected_schema=self.config.expected_schema
        )
        self.pii_detector = OutputPIIDetector(PIIDetector())
        self.safety_filter = OutputSafetyFilter()
        self.leakage_detector = None  # Set after system prompt is known
        self.hallucination_detector = HallucinationDetector()
        self.fact_checker = ExternalFactChecker()
        self.logger = HarnessLogger()
    
    def set_system_prompt(self, system_prompt: str, 
                         tool_definitions: list[dict] = None):
        """Set system prompt for leakage detection."""
        self.leakage_detector = PromptLeakageDetector(
            system_prompt, tool_definitions
        )
    
    async def validate(self, output: str,
                      context: dict = None) -> OutputGuardrailResult:
        """
        Validate output through all guardrail layers.
        """
        result = OutputGuardrailResult(original_output=output)
        cleaned_output = output
        
        # Layer 1: Schema validation
        if self.config.validate_schema:
            schema_result = self.schema_validator.validate(cleaned_output)
            result.add_check("schema", schema_result)
            if not schema_result.passed:
                result.reject(schema_result.error, layer="schema")
                self.logger.log_output_validation(output, "rejected", schema_result.error)
                return result
        
        # Layer 2: PII detection
        if self.config.check_pii:
            pii_result = self.pii_detector.check(
                cleaned_output,
                conversation_context=context.get("conversation_pii") if context else None
            )
            result.add_check("pii", pii_result)
            if not pii_result.passed:
                result.reject(pii_result.message, layer="pii")
                self.logger.log_output_validation(output, "blocked", "PII leak detected")
                return result
            if pii_result.action == "redact":
                cleaned_output = pii_result.redacted_output
        
        # Layer 3: Content safety
        if self.config.check_safety:
            safety_result = self.safety_filter.check(cleaned_output)
            result.add_check("safety", safety_result)
            if not safety_result.passed:
                result.reject(safety_result.message, layer="safety")
                self.logger.log_output_validation(output, "blocked", "Safety violation")
                return result
        
        # Layer 4: Prompt leakage
        if self.config.check_leakage and self.leakage_detector:
            leakage_result = self.leakage_detector.detect(cleaned_output)
            result.add_check("leakage", leakage_result)
            if not leakage_result.passed and leakage_result.action == "block":
                result.reject("Response blocked due to security concerns.", layer="leakage")
                self.logger.log_output_validation(output, "blocked", "Prompt leakage detected")
                return result
        
        # Layer 5: Hallucination detection
        if self.config.check_hallucination:
            hallucination_result = await self.hallucination_detector.detect(
                cleaned_output, context
            )
            result.add_check("hallucination", hallucination_result)
            if not hallucination_result.passed and self.config.block_on_hallucination:
                result.reject(
                    "The response contained information that could not be verified.",
                    layer="hallucination"
                )
                return result
        
        # Layer 6: External fact-checking (if sources available)
        if self.config.check_facts and context and context.get("retrieved_documents"):
            fact_result = await self.fact_checker.verify_response(
                cleaned_output,
                source_context=context.get("retrieved_documents")
            )
            result.add_check("facts", fact_result)
            if not fact_result.passed:
                result.reject(
                    "The response contained claims that contradict our information.",
                    layer="fact_check"
                )
                return result
        
        result.passed = True
        result.cleaned_output = cleaned_output
        self.logger.log_output_validation(output, "passed", "All guardrails passed")
        
        return result
```

> **Code Reference:** [Python](../../code/python/07-harness/) · [Node.js](../../code/nodejs/07-harness/) · [Go](../../code/go/07-harness/)  
> The harness implementations include the complete `OutputGuardrailPipeline` with all six layers.

---

## Common Pitfalls

- **"I trust structured output mode to catch everything"**: Structured output ensures valid JSON. It doesn't check if the JSON contains factual errors, PII, or toxic content in string fields.
- **"I use the same thresholds for input and output"**: Output should be held to a higher standard. A user might say something inappropriate, but your agent should never respond in kind.
- **"My hallucination detector catches everything"**: Hallucination detection is an unsolved research problem. Current methods catch obvious cases but miss subtle fabrications. Treat it as a defense-in-depth layer, not a guarantee.
- **"I block the response when PII is detected"**: If the user asked about their own order, the response should include their order number. Distinguish between expected PII (part of the conversation) and leaked PII (from training data or system internals).
- **"I don't log blocked responses"**: Blocked responses are the most important ones to log. They indicate a failure in your system that needs investigation.
- **"My safety filter is too aggressive and blocks legitimate responses"**: A medical chatbot should be able to discuss medical topics. A financial advisor should discuss investments. Tune your safety thresholds per use case.

## What's Next

Output is now validated. Next: the ultimate guardrail — knowing when to stop the machine and ask a human.
→ [Human-in-the-Loop](06-human-in-the-loop.md)