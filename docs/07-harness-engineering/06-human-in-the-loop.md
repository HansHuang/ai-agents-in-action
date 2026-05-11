# Human-in-the-Loop

## What You'll Learn
- Why human-in-the-loop is not a failure mode — it's a design pattern
- The four types of human intervention: approval, correction, escalation, and oversight
- Designing approval workflows: when to pause, what to show, how to resume
- Conditional approval policies: only interrupt for high-stakes decisions
- Timeout and fallback for human reviewers: humans can fail too
- The `approve_with_edits` pattern: accept an action but change its parameters
- Tracking metrics to optimize how often approvals are required

## Prerequisites
- [The Harness Mindset](01-the-harness-mindset.md) — the harness orchestrates human intervention
- [Retry, Fallback, and Circuit Breakers](04-retry-fallback-and-circuit-breakers.md) — same resilience patterns apply to approval timeouts
- [Output Guardrails and Fact-Checking](05-output-guardrails-and-fact-checking.md) — sometimes validation needs a human
- [Routing and Intent Classification](03-routing-and-intent-classification.md) — routing can escalate to humans

---

## Why Human-in-the-Loop Is Not a Failure Mode

A common misconception: *"If our AI was good enough, we wouldn't need humans."*

Reality: Human-in-the-loop is a design pattern, not a fallback. You don't add human review because your AI is bad. You add it because some decisions shouldn't be made by machines.

| Decision Type | Should AI Decide? | Should Human Review? |
|:---|:---|:---|
| **Reversible, low-stakes** (song recommendation) | Yes | No |
| **Reversible, medium-stakes** (order lookup) | Yes | No (log for audit) |
| **Reversible, high-stakes** (send marketing email to 50k users) | Yes, with approval | Yes |
| **Irreversible, low-stakes** (delete a draft playlist) | Yes, with user confirmation | Optional |
| **Irreversible, medium-stakes** (cancel subscription) | No | Yes |
| **Irreversible, high-stakes** (issue refund over $1 000) | No | Yes (require two approvers) |
| **Life-critical** (medical diagnosis) | No | Yes (AI assists, human decides) |

Note the distinction between *user* confirmation ("Are you sure you want to delete this?") and *reviewer* approval. User confirmation is a UX pattern. Reviewer approval is a governance pattern. HITL is about the latter.

The line between "AI can handle this" and "get a human" depends on your domain, risk tolerance, and regulatory requirements.

---

## The Four Types of Human Intervention

### 1. Approval

The agent proposes an action. A human approves or rejects it before execution.

```
Agent: "I've drafted an email to the customer. Here's the preview.
       Should I send it?"

Human: [Reviews email] → Approve / Reject / Edit then Approve

Agent: "Email sent." or "Email cancelled per your request."
```

### 2. Correction

The agent produces an output. A human corrects it before it reaches the user.

```
Agent: "Here's the response: 'Your order will arrive on March 15th.'"

Human: [Checks order system] "Actually, it's March 17th." → Corrects date

Agent: "Your order will arrive on March 17th." [Corrected version sent]
```

### 3. Escalation

The agent encounters a situation it can't handle. It escalates to a human.

```
Agent: "The customer is asking about a billing error that I can't resolve
       with my available tools. Escalating to billing specialist."

Human: [Reviews conversation history] → Takes over the interaction
```

### 4. Oversight

The agent operates autonomously. A human monitors a dashboard of agent decisions and intervenes when patterns look wrong.

```
Dashboard: "Agent processed 1,247 refunds today. Average: $45.
           3 refunds over $1,000 flagged for review."

Human: [Reviews the 3 flagged refunds] → Approves 2, investigates 1
```

---

## Designing Approval Workflows

An approval workflow has five stages:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ PROPOSE  │───▶│  PENDING │───▶│  REVIEW  │───▶│  DECIDE  │───▶│ EXECUTE  │
│ (Agent)  │    │ (Queued) │    │ (Human)  │    │ (Human)  │    │ (Agent)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │                                                │
               ┌─────┴──────┐      approved ──────────────────────▶ run action
               │  TIMEOUT   │      approved_with_edits ────────────▶ run with edits
               │ (t seconds)│      rejected ──────────────────────▶ notify user
               └─────┬──────┘
                     │ auto-reject (safe default)
                     ▼
               notify user of timeout

               ─────────────────────────────
               Full audit trail logged at every transition
               ─────────────────────────────
```

### Stage 1: Propose (Agent)

The agent prepares what it wants to do and why:

```python
@dataclass
class ApprovalRequest:
    """An action that requires human approval before execution."""
    
    # What the agent wants to do
    request_id: str
    agent_id: str
    session_id: str
    proposed_action: str          # "send_email", "issue_refund", "update_database"
    proposed_params: dict         # The exact parameters for the action
    
    # Why the agent wants to do it
    reasoning: str                # "Customer requested refund for damaged item"
    conversation_summary: str     # Relevant conversation context
    evidence: list[dict]          # Supporting evidence (tool results, documents)
    
    # Impact assessment
    risk_level: str               # "low", "medium", "high", "critical"
    estimated_cost: float         # Financial impact if applicable
    affected_systems: list[str]   # What systems will be changed
    
    # Urgency
    created_at: float
    deadline: float = None        # Epoch time when this request expires (None = use policy timeout)
    
    def to_human_readable(self) -> str:
        """Format the approval request for human review."""
        return f"""
═══════════════════════════════════════
APPROVAL REQUIRED
═══════════════════════════════════════

Action: {self.proposed_action}
Risk Level: {self.risk_level.upper()}
Estimated Cost: ${self.estimated_cost:.2f}

What the agent wants to do:
{json.dumps(self.proposed_params, indent=2)}

Why:
{self.reasoning}

Conversation context:
{self.conversation_summary[:500]}

Supporting evidence:
{chr(10).join(f'- {e}' for e in self.evidence[:5])}

═══════════════════════════════════════
Approve? [Y/N/Edit] Timeout in 5:00
═══════════════════════════════════════
"""
```

### Stage 2: Review (Human)

Present the request to a human reviewer:

```python
class ApprovalInterface:
    """
    Present approval requests to human reviewers.
    Can be integrated with Slack, email, dashboard, or CLI.
    """
    
    def __init__(self, channels: list[str] = None):
        self.channels = channels or ["dashboard"]
        self.pending_requests: dict[str, ApprovalRequest] = {}
        self.reviewers: dict[str, Reviewer] = {}
    
    async def request_approval(self, 
                              request: ApprovalRequest,
                              timeout_seconds: float = 300) -> ApprovalResponse:
        """
        Send an approval request to the appropriate human reviewer.
        
        Returns: ApprovalResponse with approved, rejected, or timeout
        """
        # Find the right reviewer
        reviewer = await self._assign_reviewer(request)
        
        # Send the request
        await self._send_to_reviewer(reviewer, request)
        
        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(
                self._wait_for_reviewer_response(reviewer, request.request_id),
                timeout=timeout_seconds
            )
            return response
        
        except asyncio.TimeoutError:
            # Auto-reject on timeout (safe default)
            logger.warning(
                f"Approval request {request.request_id} timed out "
                f"after {timeout_seconds}s. Auto-rejecting."
            )
            return ApprovalResponse(
                request_id=request.request_id,
                decision="rejected",
                reason="Approval timeout — automatically rejected for safety.",
                reviewer_id=reviewer.id,
                automated=True,
            )
    
    async def _assign_reviewer(self, request: ApprovalRequest) -> "Reviewer":
        """
        Assign the request to the appropriate reviewer.
        
        Routing logic:
        - Low risk → Any available reviewer
        - Medium risk → Reviewer with relevant expertise
        - High risk → Senior reviewer
        - Critical risk → Two senior reviewers
        """
        if request.risk_level == "critical":
            reviewers = await self._get_senior_reviewers(count=2)
            return ReviewerGroup(reviewers)
        elif request.risk_level == "high":
            return await self._get_senior_reviewer(request.affected_systems)
        elif request.risk_level == "medium":
            return await self._get_reviewer_with_expertise(request.proposed_action)
        else:
            return await self._get_any_available_reviewer()
    
    async def _send_to_reviewer(self, reviewer: "Reviewer", 
                               request: ApprovalRequest):
        """Send the approval request to the reviewer's preferred channel."""
        message = request.to_human_readable()
        
        if "slack" in self.channels:
            await self._send_slack_message(reviewer.slack_id, message)
        
        if "dashboard" in self.channels:
            self.pending_requests[request.request_id] = request
        
        if "email" in self.channels and request.risk_level in ("high", "critical"):
            await self._send_email(reviewer.email, f"URGENT: Approval Required - {request.proposed_action}", message)
    
    async def _wait_for_reviewer_response(self, reviewer: "Reviewer",
                                         request_id: str) -> ApprovalResponse:
        """Wait for the reviewer to respond."""
        # In production: listen for webhook, message queue, or poll database
        # For demo: use an asyncio.Event
        event = asyncio.Event()
        self._response_events[request_id] = event
        await event.wait()
        return self._responses.pop(request_id)
```

### Stage 3: Decide (Human)

The human makes a decision:

```python
@dataclass
class ApprovalResponse:
    """A human reviewer's decision on an approval request."""
    request_id: str
    decision: str          # "approved", "rejected", "approved_with_edits"
    reviewer_id: str
    reviewer_notes: str = None
    edited_params: dict = None  # If "approved_with_edits", the modified parameters
    automated: bool = False     # True if auto-rejected on timeout
    decided_at: float = None
    reason: str = None
    
    def __post_init__(self):
        self.decided_at = time.time()

class HumanReviewerInterface:
    """Interface for human reviewers to respond to approval requests."""
    
    def approve(self, request_id: str, notes: str = None) -> ApprovalResponse:
        """Approve the request as-is."""
        return ApprovalResponse(
            request_id=request_id,
            decision="approved",
            reviewer_id=self.reviewer_id,
            reviewer_notes=notes,
        )
    
    def reject(self, request_id: str, reason: str) -> ApprovalResponse:
        """Reject the request with a reason."""
        return ApprovalResponse(
            request_id=request_id,
            decision="rejected",
            reviewer_id=self.reviewer_id,
            reason=reason,
        )
    
    def approve_with_edits(self, request_id: str, 
                          edited_params: dict,
                          notes: str = None) -> ApprovalResponse:
        """
        Approve the action but with modified parameters.
        Example: Agent wants to refund $100, human changes to $75.
        """
        return ApprovalResponse(
            request_id=request_id,
            decision="approved_with_edits",
            reviewer_id=self.reviewer_id,
            edited_params=edited_params,
            reviewer_notes=notes,
        )
```

### Stage 4: Execute (Agent)

The agent carries out the decision:

```python
class ApprovalExecutor:
    """Execute actions based on human approval decisions."""
    
    def __init__(self, agent):
        self.agent = agent
        self.approved_actions = []
        self.rejected_actions = []
    
    async def execute(self, request: ApprovalRequest,
                     response: ApprovalResponse) -> ExecutionResult:
        """
        Execute based on the human's decision.
        """
        
        if response.decision == "approved":
            return await self._execute_approved(request)
        
        elif response.decision == "approved_with_edits":
            return await self._execute_edited(request, response.edited_params)
        
        elif response.decision == "rejected":
            return self._handle_rejection(request, response)
        
        else:
            raise ValueError(f"Unknown decision: {response.decision}")
    
    async def _execute_approved(self, 
                               request: ApprovalRequest) -> ExecutionResult:
        """Execute the action as originally proposed."""
        logger.info(f"Executing approved action: {request.proposed_action}")
        
        try:
            result = await self.agent.execute_tool(
                tool_name=request.proposed_action,
                params=request.proposed_params,
            )
            
            # Notify the user
            await self.agent.send_message(
                f"I've completed your request: {request.proposed_action}"
            )
            
            self.approved_actions.append({
                "request": request,
                "result": result,
                "timestamp": time.time(),
            })
            
            return ExecutionResult(success=True, result=result)
        
        except Exception as e:
            logger.error(f"Approved action failed: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _execute_edited(self, request: ApprovalRequest,
                             edited_params: dict) -> ExecutionResult:
        """Execute the action with human-modified parameters."""
        logger.info(
            f"Executing edited action: {request.proposed_action} "
            f"with params: {edited_params}"
        )
        
        try:
            result = await self.agent.execute_tool(
                tool_name=request.proposed_action,
                params=edited_params,
            )
            
            await self.agent.send_message(
                f"I've completed your request with the adjustments you specified."
            )
            
            return ExecutionResult(success=True, result=result)
        
        except Exception as e:
            logger.error(f"Edited action failed: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    def _handle_rejection(self, request: ApprovalRequest,
                         response: ApprovalResponse) -> ExecutionResult:
        """Handle a rejected approval request."""
        logger.info(
            f"Action rejected: {request.proposed_action}. "
            f"Reason: {response.reason}"
        )
        
        # Inform the user
        rejection_message = (
            f"I wasn't able to complete '{request.proposed_action}'. "
            f"{response.reason if response.reason else 'This action requires additional review.'}"
        )
        
        self.agent.send_message(rejection_message)
        self.rejected_actions.append({
            "request": request,
            "response": response,
            "timestamp": time.time(),
        })
        
        return ExecutionResult(
            success=False,
            error=f"Rejected by reviewer: {response.reason}",
        )
```

---

## Conditional Human-in-the-Loop

Don't ask for approval on every action. Only interrupt when it matters:

```python
class ApprovalPolicy:
    """
    Define when human approval is required.
    Policies are evaluated in order. First match wins.
    """
    
    def __init__(self):
        self.rules: list[ApprovalRule] = []
    
    def add_rule(self, rule: "ApprovalRule"):
        """Add an approval rule. Rules are evaluated in priority order."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def requires_approval(self, action: str, params: dict,
                         context: dict) -> ApprovalDecision:
        """
        Determine if an action requires human approval.
        """
        for rule in self.rules:
            if rule.matches(action, params, context):
                return ApprovalDecision(
                    requires_approval=True,
                    rule=rule.name,
                    risk_level=rule.risk_level,
                    timeout_seconds=rule.timeout_seconds,
                )
        
        # No rule matched — action is auto-approved
        return ApprovalDecision(
            requires_approval=False,
            rule="default_allow",
            risk_level="none",
        )

@dataclass
class ApprovalRule:
    """A rule for when human approval is needed."""
    name: str
    description: str
    priority: int                    # Higher = evaluated first
    risk_level: str                  # "low", "medium", "high", "critical"
    timeout_seconds: float = 300     # How long to wait for human response
    
    # Matching conditions
    actions: list[str] = None        # Specific actions that require approval
    min_cost: float = None           # Require approval above this cost
    affected_systems: list[str] = None  # Systems that trigger approval
    user_roles: list[str] = None     # User roles that require approval
    
    def matches(self, action: str, params: dict, 
               context: dict) -> bool:
        """Check if this rule matches the current action."""
        # Check if action matches
        if self.actions and action not in self.actions:
            return False
        
        # Check if cost exceeds threshold
        if self.min_cost is not None:
            action_cost = self._estimate_cost(action, params)
            if action_cost < self.min_cost:
                return False
        
        # Check if affected systems match
        if self.affected_systems:
            action_systems = self._get_affected_systems(action, params)
            if not any(s in self.affected_systems for s in action_systems):
                return False
        
        return True
    
    def _estimate_cost(self, action: str, params: dict) -> float:
        """Estimate the financial impact of an action."""
        if action == "issue_refund":
            return float(params.get("amount", 0))
        if action == "send_email":
            return 0.0  # Reversible, low cost
        if action == "update_database":
            return 50.0  # Moderate risk
        return 10.0  # Default moderate
    
    def _get_affected_systems(self, action: str, params: dict) -> list[str]:
        """Determine which systems an action affects."""
        system_map = {
            "send_email": ["email_service"],
            "issue_refund": ["payment_processor", "order_database"],
            "update_database": ["database"],
            "create_ticket": ["support_system"],
            "cancel_subscription": ["billing_system", "subscription_service"],
        }
        return system_map.get(action, ["unknown"])

@dataclass
class ApprovalDecision:
    requires_approval: bool
    rule: str
    risk_level: str
    timeout_seconds: float = 300

# Configure approval policies
policy = ApprovalPolicy()

policy.add_rule(ApprovalRule(
    name="high_value_refund",
    description="Refunds over $500 require human approval",
    priority=100,
    risk_level="high",
    actions=["issue_refund"],
    min_cost=500.0,
    timeout_seconds=600,  # 10 minutes for high-value decisions
))

policy.add_rule(ApprovalRule(
    name="external_communication",
    description="Sending email to customers requires approval",
    priority=90,
    risk_level="medium",
    actions=["send_email"],
    timeout_seconds=300,
))

policy.add_rule(ApprovalRule(
    name="database_modification",
    description="Any database write requires approval",
    priority=80,
    risk_level="medium",
    actions=["update_database", "delete_record"],
    timeout_seconds=300,
))

policy.add_rule(ApprovalRule(
    name="subscription_cancellation",
    description="Cancelling subscriptions requires approval",
    priority=85,
    risk_level="high",
    actions=["cancel_subscription"],
    timeout_seconds=600,
))
```

---

## Integrating Human-in-the-Loop into the Harness

The harness state machine gets a new state:

```python
class HarnessWithHumanApproval(HarnessStateMachine):
    """
    Harness with human-in-the-loop integration.
    Extends the base harness with an approval state.
    """
    
    def __init__(self, approval_policy: ApprovalPolicy,
                approval_interface: ApprovalInterface):
        super().__init__()
        self.approval_policy = approval_policy
        self.approval_interface = approval_interface
        self.approval_executor = ApprovalExecutor(self)
    
    async def process(self, user_input: str,
                     user_context: dict = None) -> HarnessResponse:
        """Process a request with optional human approval."""
        
        # ... input guardrails, routing, handler execution ...
        # (same as base harness)
        
        # After agent produces output with tool calls
        if handler_response.tool_calls:
            for tool_call in handler_response.tool_calls:
                # Check if this action requires approval
                decision = self.approval_policy.requires_approval(
                    action=tool_call.tool_name,
                    params=tool_call.params,
                    context={
                        "user_role": user_context.get("role", "user"),
                        "session_id": user_context.get("session_id"),
                    }
                )
                
                if decision.requires_approval:
                    # Pause and request human approval
                    self._transition("awaiting_approval")
                    
                    approval_request = ApprovalRequest(
                        request_id=str(uuid.uuid4()),
                        agent_id=self.agent_id,
                        session_id=user_context.get("session_id"),
                        proposed_action=tool_call.tool_name,
                        proposed_params=tool_call.params,
                        reasoning=handler_response.reasoning,
                        conversation_summary=self._get_conversation_summary(),
                        evidence=self._get_evidence(),
                        risk_level=decision.risk_level,
                        estimated_cost=self._estimate_cost(tool_call),
                        affected_systems=self._get_affected_systems(tool_call),
                        created_at=time.time(),
                    )
                    
                    # Wait for human decision
                    try:
                        response = await self.approval_interface.request_approval(
                            approval_request,
                            timeout_seconds=decision.timeout_seconds,
                        )
                    except Exception as e:
                        logger.error(f"Approval request failed: {e}")
                        self._transition("approval_failed")
                        return self._handle_approval_failure(tool_call)
                    
                    # Execute based on decision
                    if response.decision == "rejected":
                        self._transition("action_rejected")
                        return self.approval_executor._handle_rejection(
                            approval_request, response
                        )
                    
                    # Execute approved action
                    result = await self.approval_executor.execute(
                        approval_request, response
                    )
                    
                    if result.success:
                        self._transition("action_executed")
                    else:
                        self._transition("execution_failed")
                        return self._handle_execution_failure(result)
        
        # All actions processed
        self._transition("responding")
        return self._build_final_response()
    
    def _transition(self, new_state: str, reason: str = None):
        logger.info(f"Harness state: {self.state} → {new_state}")
        self.state = new_state
    
    def _handle_approval_failure(self, tool_call) -> HarnessResponse:
        """Handle when the approval system itself fails."""
        return HarnessResponse(
            content="I apologize, but I'm unable to complete this action right now. "
                   "A team member will follow up with you shortly.",
            status="approval_system_error",
        )
    
    def _handle_execution_failure(self, result: ExecutionResult) -> HarnessResponse:
        """Handle when an approved action fails to execute."""
        return HarnessResponse(
            content=f"I apologize, but the action could not be completed. "
                   f"Error: {result.error}. A team member has been notified.",
            status="execution_error",
        )
```

---

## Human-in-the-Loop Best Practices

### 1. Default to Safety

If the human doesn't respond within the timeout, the safest default is **reject**, not approve. A missed approval delays a legitimate action. A missed rejection allows a potentially harmful one.

```python
# Always default to the safest option on timeout
DEFAULT_TIMEOUT_DECISION = "rejected"   # ← safe default
# NEVER: DEFAULT_TIMEOUT_DECISION = "approved"  # ← unsafe
```

Auto-rejection on timeout is not a failure — it is the system working as designed. The action can always be re-submitted once a reviewer becomes available.

### 2. Give the Human Context

Don't just show the action. Show:
- What the user actually asked for
- What the agent did to get there
- What evidence supports this action
- What happens if approved
- What happens if rejected

### 3. Make It Easy to Say "Yes, But..."

Allow the human to modify the proposed action rather than forcing a binary choice:

```python
# Agent proposes: refund $100
# Human can: approve $100, reject entirely, or approve with edit: refund $75
response = reviewer.approve_with_edits(
    request_id=request.id,
    edited_params={"amount": 75.00},
    notes="Standard refund for this product category is $75, not $100."
)
```

The `approve_with_edits` path is often the most important one. It lets a reviewer correct an almost-right proposal without blocking the user entirely. Track your edit rate — a high edit rate is a signal that the agent has learned the right *actions* but not the right *parameters*.

### 4. Track Every Decision

Every approval and rejection should be logged with:
- Who made the decision
- When they made it
- What they decided
- Why (if they provided a reason)

This is your audit trail for compliance and improvement.

### 5. Measure and Optimize

Track metrics to improve your approval policies:

```python
class ApprovalMetrics:
    """Track human-in-the-loop metrics."""
    
    def __init__(self):
        self.total_requests = 0
        self.approved = 0
        self.rejected = 0
        self.approved_with_edits = 0
        self.timed_out = 0
        self.avg_response_time = 0.0
        self.by_risk_level = defaultdict(lambda: {"approved": 0, "rejected": 0})
    
    def summary(self) -> dict:
        return {
            "total_approval_requests": self.total_requests,
            "approval_rate": self.approved / max(self.total_requests, 1),
            "rejection_rate": self.rejected / max(self.total_requests, 1),
            "edit_rate": self.approved_with_edits / max(self.total_requests, 1),
            "timeout_rate": self.timed_out / max(self.total_requests, 1),
            "avg_response_time_seconds": self.avg_response_time,
            "by_risk_level": dict(self.by_risk_level),
        }
```

If your rejection rate is 40%, your agent is proposing too many bad actions. If your timeout rate is 20%, your reviewers are overloaded. If your edit rate is high, your agent is close but not quite right. Use these signals to improve.

---

## Common Pitfalls

- **"I require human approval for everything"**: If every action needs approval, the human becomes a bottleneck and the agent provides no value. Use conditional approval policies.
- **"My approval timeout is too long"**: A 30-minute timeout means the user waits 30 minutes. Set timeouts appropriate for the user experience. Most approvals should be under 5 minutes.
- **"I auto-approve on timeout because the reviewer is probably fine with it"**: This defeats the purpose. An unanswered request means the reviewer did not see it, not that they agreed. Default to reject and re-queue if needed.
- **"I don't give reviewers enough context"**: Showing just the action without the reasoning, conversation history, or evidence forces the reviewer to investigate from scratch. Give them everything they need.
- **"I don't have an escalation path when the reviewer disagrees"**: What if the reviewer rejects, but the user insists? Have a process for contested decisions.
- **"I treat human review as a one-time setup"**: Review your approval policies regularly. If certain actions are always approved, remove the approval requirement. If certain actions are frequently rejected, improve the agent.

## What's Next

You've built the complete harness: input guardrails, routing, resilience, output guardrails, and human-in-the-loop. Next: putting it all together into a single, reliable system.
→ [Building a Reliable Harness](07-building-a-reliable-harness.md)

---

## Reference Implementation

The code for this chapter lives at:

- Python: [code/python/07-harness/human_in_the_loop.py](../../code/python/07-harness/human_in_the_loop.py) — core system
- Python: [code/python/07-harness/approval_dashboard.py](../../code/python/07-harness/approval_dashboard.py) — Rich CLI reviewer dashboard
- Python: [code/python/07-harness/approval_policy_optimizer.py](../../code/python/07-harness/approval_policy_optimizer.py) — policy optimizer with metrics analysis
- Node.js: [code/nodejs/07-harness/human_in_the_loop.ts](../../code/nodejs/07-harness/human_in_the_loop.ts) — TypeScript port
- Go: [code/go/07-harness/human_in_the_loop.go](../../code/go/07-harness/human_in_the_loop.go) — Go port