"""
Deployment System Tests
=======================
Pytest test suite for code/python/09-deployment/deployment_manager.py

Tests cover:
  - DeploymentManager  (rollout, promotion, halt)
  - CanaryDeployer     (evaluation, thresholds, recommendations)
  - ProductionCostController (budgets, alerts, reset)
  - RollbackManager    (items, history, dry-run)
  - MultiRegionDeployer (routing, GDPR, fallback)
  - AgentLoadTester    (metrics, reports, distribution)
  - DeploymentHealthChecker (aggregation, degraded/unhealthy)

Run with: pytest test_deployment.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deployment_manager import (
    CanaryDeployer,
    CanaryEvaluation,
    CostConfig,
    DeploymentHealthChecker,
    DeploymentManager,
    FeatureFlagService,
    HarnessMetrics,
    MultiRegionDeployer,
    ProductionCostController,
    ProductionHarness,
    RollbackManager,
    StreamingDeployment,
)
from load_tester import AgentLoadTester, LoadTestMetrics, LoadTestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flags():
    return FeatureFlagService()


@pytest.fixture
def dm(flags):
    return DeploymentManager(flags)


@pytest.fixture
def metrics():
    return HarnessMetrics()


@pytest.fixture
def stable_harness():
    return ProductionHarness("v3.1.0")


@pytest.fixture
def canary_harness():
    return ProductionHarness("v3.2.1")


@pytest.fixture
def deployer(stable_harness, canary_harness, metrics, dm):
    return CanaryDeployer(stable_harness, canary_harness, metrics, dm)


@pytest.fixture
def cost_cfg():
    return CostConfig(
        user_daily_budget=10.0,
        total_daily_budget=100.0,
        max_cost_per_request=1.0,
        free_tier_daily_budget=0.50,
        enterprise_daily_budget=50.0,
    )


@pytest.fixture
def cost_ctrl(cost_cfg):
    return ProductionCostController(cost_cfg)


@pytest.fixture
def multi_region():
    return MultiRegionDeployer()


@pytest.fixture
def rollback_mgr(dm):
    return RollbackManager(dm)


# ---------------------------------------------------------------------------
# 1. DeploymentManager tests
# ---------------------------------------------------------------------------

class TestDeploymentManager:
    def test_internal_users_always_get_new_version(self, dm):
        """Internal users should always receive the canary/internal version."""
        version = dm.get_agent_version("internal-001")
        assert version == "v3.2.1"

    def test_external_user_gets_stable_at_zero_rollout(self, dm):
        """With 0% rollout, external users should get stable."""
        version = dm.get_agent_version("external-user-9999")
        assert version == "v3.1.0"

    def test_1pct_rollout_affects_approximately_1pct(self, dm):
        """1% rollout → approximately 1% of a large user set gets canary."""
        dm.promote_rollout(0, 1)
        canary_count = sum(
            1 for i in range(10000)
            if dm.get_agent_version(f"user-{i:05d}") == "v3.2.1"
        )
        # Expect between 0.5% and 2% (ample tolerance for MD5 distribution)
        assert 50 <= canary_count <= 200

    def test_same_user_always_same_version(self, dm):
        """Deterministic assignment: the same user always gets the same version."""
        dm.promote_rollout(0, 5)
        versions = [dm.get_agent_version("user-abc123") for _ in range(50)]
        assert len(set(versions)) == 1

    def test_promote_rollout_increases_percentage(self, dm):
        """Promoting from 1% to 5% should increase the canary user population."""
        dm.promote_rollout(0, 1)
        at_1pct = sum(
            1 for i in range(1000)
            if dm.get_agent_version(f"user-{i:05d}") == "v3.2.1"
        )
        dm.promote_rollout(1, 5)
        at_5pct = sum(
            1 for i in range(1000)
            if dm.get_agent_version(f"user-{i:05d}") == "v3.2.1"
        )
        assert at_5pct > at_1pct

    def test_halt_rollout_returns_all_to_stable(self, dm):
        """After halt, all external users should get stable."""
        dm.promote_rollout(0, 25)
        dm.halt_rollout("test halt")
        versions = [dm.get_agent_version(f"user-{i:05d}") for i in range(100)]
        assert all(v == "v3.1.0" for v in versions)

    def test_rollout_status_reflects_current_state(self, dm):
        """get_rollout_status should reflect the current rollout percentage."""
        dm.promote_rollout(0, 5)
        status = dm.get_rollout_status()
        assert status["canary_rollout_pct"] == 5
        assert status["halted"] is False

    def test_promote_to_invalid_stage_raises(self, dm):
        """Promoting to a non-stage percentage should raise ValueError."""
        with pytest.raises(ValueError):
            dm.promote_rollout(0, 7)

    def test_promote_from_wrong_state_raises(self, dm):
        """Promoting from a wrong 'from' value should raise ValueError."""
        with pytest.raises(ValueError):
            dm.promote_rollout(5, 25)  # currently at 0%, not 5%


# ---------------------------------------------------------------------------
# 2. CanaryDeployer tests
# ---------------------------------------------------------------------------

class TestCanaryDeployer:
    def _fill_metrics(self, metrics, version, count=50, *, error_rate=0.0,
                      latency=1.0, cost=0.02):
        for i in range(count):
            error = i < int(count * error_rate)
            metrics.record(version, error=error, latency=latency, cost=cost)

    def test_canary_healthy_when_metrics_similar(self, deployer, metrics, dm):
        """All metrics within thresholds → no issues reported."""
        self._fill_metrics(metrics, "stable")
        self._fill_metrics(metrics, "canary")
        dm.promote_rollout(0, 1)
        evaluation = deployer.evaluate_canary()
        assert not evaluation.has_issues
        assert len(evaluation.issues) == 0

    def test_canary_error_rate_spike_detected(self, deployer, metrics, dm):
        """Canary error rate > 1.5× stable should be flagged."""
        self._fill_metrics(metrics, "stable", count=100, error_rate=0.02)
        self._fill_metrics(metrics, "canary", count=100, error_rate=0.10)
        dm.promote_rollout(0, 1)
        evaluation = deployer.evaluate_canary()
        assert evaluation.has_issues
        assert any("Error rate" in issue for issue in evaluation.issues)

    def test_canary_latency_spike_detected(self, deployer, metrics, dm):
        """Canary P95 latency > 1.2× stable should be flagged."""
        self._fill_metrics(metrics, "stable", count=100, latency=1.0)
        self._fill_metrics(metrics, "canary", count=100, latency=5.0)
        dm.promote_rollout(0, 1)
        evaluation = deployer.evaluate_canary()
        assert evaluation.has_issues
        assert any("latency" in issue.lower() for issue in evaluation.issues)

    def test_canary_cost_spike_detected(self, deployer, metrics, dm):
        """Canary avg cost > 1.2× stable should be flagged."""
        self._fill_metrics(metrics, "stable", count=50, cost=0.02)
        self._fill_metrics(metrics, "canary", count=50, cost=0.10)
        dm.promote_rollout(0, 1)
        evaluation = deployer.evaluate_canary()
        assert evaluation.has_issues
        assert any("cost" in issue.lower() for issue in evaluation.issues)

    def test_recommendation_promote_when_healthy(self, deployer, metrics, dm):
        """No issues → recommendation should mention increasing rollout."""
        self._fill_metrics(metrics, "stable")
        self._fill_metrics(metrics, "canary")
        dm.promote_rollout(0, 1)
        evaluation = deployer.evaluate_canary()
        assert "increasing" in evaluation.recommendation.lower() or \
               "healthy" in evaluation.recommendation.lower()

    def test_recommendation_halt_when_unhealthy(self, deployer, metrics, dm):
        """Multiple issues → recommendation should mention halting."""
        self._fill_metrics(metrics, "stable", error_rate=0.01, latency=1.0, cost=0.02)
        self._fill_metrics(metrics, "canary", error_rate=0.10, latency=5.0, cost=0.10)
        dm.promote_rollout(0, 1)
        evaluation = deployer.evaluate_canary()
        assert "halt" in evaluation.recommendation.lower()

    @pytest.mark.asyncio
    async def test_canary_users_get_canary_version(self, deployer, dm, metrics):
        """Users in the rollout group should have their request served by canary."""
        dm.promote_rollout(0, 100)  # 100% canary
        response = await deployer.process("Hello", user_id="external-user-99")
        assert response.metadata["version"] == "canary"


# ---------------------------------------------------------------------------
# 3. ProductionCostController tests
# ---------------------------------------------------------------------------

class TestProductionCostController:
    def test_request_within_budget_allowed(self, cost_ctrl):
        """A cheap request for a fresh user should be allowed."""
        check = cost_ctrl.check_budget("user-1", 0.01)
        assert check.allowed

    def test_user_budget_exceeded_rejected(self, cost_ctrl):
        """Spending over the per-user limit should be rejected."""
        cost_ctrl.record_cost("user-1", 9.95)
        check = cost_ctrl.check_budget("user-1", 0.10)
        assert not check.allowed
        assert "budget" in check.reason.lower()

    def test_total_budget_exceeded_rejected(self, cost_ctrl):
        """Spending over the total daily limit should be rejected."""
        cost_ctrl.record_cost("user-1", 99.95)  # Nearly hit $100 cap
        check = cost_ctrl.check_budget("user-2", 0.10)
        assert not check.allowed

    def test_warning_at_70pct(self, cost_ctrl):
        """70% of daily budget consumed → info-level alert should fire."""
        with patch.object(cost_ctrl, "_trigger_alert") as mock_alert:
            cost_ctrl.record_cost("user-1", 70.0)  # 70% of $100
            mock_alert.assert_called_once()
            assert mock_alert.call_args[0][0] == "info"

    def test_critical_at_90pct(self, cost_ctrl):
        """90% of daily budget → warning-level alert should fire."""
        cost_ctrl.record_cost("user-1", 70.0)  # pass 70% mark silently
        cost_ctrl._last_alert_level = None  # reset so next fires
        with patch.object(cost_ctrl, "_trigger_alert") as mock_alert:
            cost_ctrl.record_cost("user-1", 20.0)  # now at 90%
            mock_alert.assert_called_once()
            level = mock_alert.call_args[0][0]
            assert level in ("warning", "critical")

    def test_free_tier_lower_budget(self, cost_ctrl):
        """Free-tier users should have a lower daily budget than premium users."""
        cost_ctrl.register_free_tier("free-user")
        cost_ctrl.record_cost("free-user", 0.45)
        check = cost_ctrl.check_budget("free-user", 0.10)  # would exceed $0.50
        assert not check.allowed
        # Premium user with same spend is still within budget
        cost_ctrl.record_cost("premium-user", 0.45)
        check2 = cost_ctrl.check_budget("premium-user", 0.10)
        assert check2.allowed

    def test_cost_reset_daily(self, cost_ctrl):
        """After reset_daily_costs, all counters should be zero."""
        cost_ctrl.record_cost("user-1", 5.0)
        cost_ctrl.reset_daily_costs()
        report = cost_ctrl.get_cost_report()
        assert report["total_daily_cost"] == 0.0
        assert report["user_count"] == 0

    def test_per_request_limit_rejected(self, cost_ctrl):
        """A single request exceeding max_cost_per_request should be rejected."""
        check = cost_ctrl.check_budget("user-1", 2.0)  # limit is $1.00
        assert not check.allowed

    def test_enterprise_user_higher_budget(self, cost_ctrl):
        """Enterprise users should have the highest daily budget."""
        cost_ctrl.register_enterprise("enterprise-user")
        cost_ctrl.record_cost("enterprise-user", 45.0)
        check = cost_ctrl.check_budget("enterprise-user", 4.0)  # < $50 limit
        assert check.allowed


# ---------------------------------------------------------------------------
# 4. RollbackManager tests
# ---------------------------------------------------------------------------

class TestRollbackManager:
    @pytest.mark.asyncio
    async def test_rollback_code_successful(self, rollback_mgr):
        """Rolling back 'code' alone should succeed."""
        result = await rollback_mgr.rollback("test reason", items=["code"])
        assert result.success
        code_result = next(r for r in result.items if r.name == "code")
        assert code_result.success

    @pytest.mark.asyncio
    async def test_rollback_all_items(self, rollback_mgr):
        """Rolling back with items=None should process all 6 items."""
        result = await rollback_mgr.rollback("full rollback")
        assert len(result.items) == 6
        assert result.success

    @pytest.mark.asyncio
    async def test_rollback_specific_items_only(self, rollback_mgr):
        """Rolling back only 'config' should not touch other items."""
        result = await rollback_mgr.rollback("config only", items=["config"])
        assert len(result.items) == 1
        assert result.items[0].name == "config"

    @pytest.mark.asyncio
    async def test_rollback_time_estimate_reasonable(self, rollback_mgr):
        """Total time estimate for config + prompt should be 20s."""
        result = await rollback_mgr.rollback("timing test", items=["config", "prompt"])
        assert result.total_time_seconds == 20

    @pytest.mark.asyncio
    async def test_failed_rollback_item_reported(self, rollback_mgr):
        """If one rollback method raises, the result should show success=False."""
        async def bad_rollback():
            raise RuntimeError("Simulated rollback failure")

        # Patch the _rollback_code method to fail
        rollback_mgr._rollback_code = bad_rollback
        # Re-setup items to pick up the patched method
        rollback_mgr._setup_items()
        # Assign the bad method directly
        for item in rollback_mgr._rollback_items:
            if item.name == "code":
                import dataclasses
                object.__setattr__(item, "method", bad_rollback)

        result = await rollback_mgr.rollback("failure test", items=["code"])
        assert not result.success
        code_result = next(r for r in result.items if r.name == "code")
        assert not code_result.success
        assert "Simulated" in code_result.error

    @pytest.mark.asyncio
    async def test_rollback_history_recorded(self, rollback_mgr):
        """After a rollback, the history should contain one entry."""
        await rollback_mgr.rollback("first rollback", items=["config"])
        await rollback_mgr.rollback("second rollback", items=["prompt"])
        history = rollback_mgr.get_rollback_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_test_rollback_dry_run_succeeds(self, rollback_mgr):
        """Dry-run should return True when all stubs succeed."""
        ok = await rollback_mgr.test_rollback()
        assert ok is True

    @pytest.mark.asyncio
    async def test_rollback_halts_rollout(self, rollback_mgr, dm):
        """Rollback should halt the active rollout."""
        dm.promote_rollout(0, 5)
        assert not dm._halted
        await rollback_mgr.rollback("halt check", items=["config"])
        assert dm._halted


# ---------------------------------------------------------------------------
# 5. MultiRegionDeployer tests
# ---------------------------------------------------------------------------

class TestMultiRegionDeployer:
    def test_eu_user_routed_to_eu(self, multi_region):
        """An EU IP address should be routed to eu-west."""
        region = multi_region.get_region("195.50.10.1")
        assert region == "eu-west"

    def test_us_user_routed_to_us(self, multi_region):
        """A US IP address should be routed to us-east."""
        region = multi_region.get_region("52.86.1.1")
        assert region == "us-east"

    def test_ap_user_routed_to_ap(self, multi_region):
        """An AP IP address should be routed to ap-southeast."""
        region = multi_region.get_region("13.250.1.1")
        assert region == "ap-southeast"

    def test_unknown_ip_defaults_to_us_east(self, multi_region):
        """An unknown IP should default to us-east."""
        region = multi_region.get_region("1.2.3.4")
        assert region == "us-east"

    def test_unhealthy_region_triggers_fallback(self, multi_region):
        """If the routed region is unhealthy, the nearest healthy region is used."""
        multi_region.open_circuit_breaker("us-east")
        region = multi_region.get_region("52.86.1.1")  # US user
        assert region != "us-east"
        assert multi_region._is_region_healthy(region)

    def test_user_preference_respected_when_healthy(self, multi_region):
        """A user's preferred region should be honoured if it is healthy."""
        region = multi_region.get_region("52.86.1.1", user_preferences={"region": "eu-west"})
        assert region == "eu-west"

    def test_user_preference_ignored_when_unhealthy(self, multi_region):
        """If the preferred region is unhealthy, fall back to geographic routing."""
        multi_region.open_circuit_breaker("eu-west")
        region = multi_region.get_region("52.86.1.1", user_preferences={"region": "eu-west"})
        assert region != "eu-west"

    def test_get_region_config_returns_dict(self, multi_region):
        """get_region_config should return a config dict with required keys."""
        cfg = multi_region.get_region_config("us-east")
        assert "llm_provider" in cfg
        assert "latency_to_provider_ms" in cfg


# ---------------------------------------------------------------------------
# 6. AgentLoadTester tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestAgentLoadTester:
    def _make_tester(self):
        return AgentLoadTester("http://localhost:8000")

    def test_generate_queries_respects_count(self):
        """generate_test_queries should return exactly count queries."""
        tester = self._make_tester()
        queries = tester.generate_test_queries(100)
        assert len(queries) == 100

    def test_generate_queries_distribution_approx(self):
        """Query type distribution should approximate the target fractions."""
        tester = self._make_tester()
        queries = tester.generate_test_queries(1000)
        type_counts = {}
        for q in queries:
            type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1
        # simple_chat target is 35% → expect at least 25% and at most 45%
        simple_frac = type_counts.get("simple_chat", 0) / 1000
        assert 0.25 <= simple_frac <= 0.45

    def test_latency_percentiles_calculated(self):
        """LoadTestMetrics.percentile should return correct percentiles."""
        m = LoadTestMetrics()
        for i in range(1, 101):
            r = LoadTestResult(
                request_id=str(i), query="q", query_type="simple_chat",
                status_code=200, latency_ms=float(i), tokens_used=10,
                cost=0.001, success=True,
            )
            m.record(r)
        assert abs(m.percentile(0.50) - 50.0) <= 5
        assert abs(m.percentile(0.95) - 95.0) <= 5

    def test_error_rate_calculated_correctly(self):
        """Error rate should equal failed / total."""
        m = LoadTestMetrics()
        for i in range(95):
            m.record(LoadTestResult(str(i), "q", "simple_chat", 200, 100.0, 10, 0.001, True))
        for i in range(5):
            m.record(LoadTestResult(f"e-{i}", "q", "simple_chat", 500, 50.0, 0, 0.0, False))
        assert abs(m.error_rate - 0.05) < 0.001

    def test_generate_report_pass_when_targets_met(self):
        """Report status should be 'pass' when all SLA targets are met."""
        tester = self._make_tester()
        from load_tester import LoadTestReport
        report = LoadTestReport(
            scenario="Test",
            duration_seconds=60.0,
            target_rps=10.0,
            actual_rps=9.9,
            p50_latency=800.0,
            p95_latency=2000.0,
            p99_latency=4000.0,
            error_rate=0.005,
            avg_cost=0.01,
            total_cost=1.0,
            status="pass",
            targets_met={k: True for k in ["p50_latency", "p95_latency", "p99_latency",
                                           "error_rate", "avg_cost", "actual_rps"]},
        )
        text = tester.generate_report(report)
        assert "ALL TARGETS MET" in text

    def test_generate_report_fail_when_targets_missed(self):
        """Report status should be 'fail' when any SLA target is missed."""
        tester = self._make_tester()
        from load_tester import LoadTestReport
        report = LoadTestReport(
            scenario="Test",
            duration_seconds=60.0,
            target_rps=10.0,
            actual_rps=9.9,
            p50_latency=5000.0,
            p95_latency=12000.0,
            p99_latency=25000.0,
            error_rate=0.05,
            avg_cost=0.10,
            total_cost=10.0,
            status="fail",
            targets_met={k: False for k in ["p50_latency", "p95_latency", "p99_latency",
                                            "error_rate", "avg_cost", "actual_rps"]},
        )
        text = tester.generate_report(report)
        assert "TARGETS MISSED" in text


# ---------------------------------------------------------------------------
# 7. DeploymentHealthChecker tests
# ---------------------------------------------------------------------------

class TestDeploymentHealthChecker:
    def _make_checker(self, cost_ctrl=None, multi_region=None):
        if cost_ctrl is None:
            cost_ctrl = ProductionCostController(CostConfig())
        if multi_region is None:
            multi_region = MultiRegionDeployer()
        return DeploymentHealthChecker(cost_ctrl, multi_region)

    def test_health_all_ok_when_healthy(self):
        """All checks passing → overall status should be 'healthy'."""
        checker = self._make_checker()
        report = checker.check_all()
        assert report.overall == "healthy"
        assert report.checks["llm_connectivity"] is True
        assert report.checks["vector_db_connectivity"] is True

    def test_health_degraded_when_one_non_critical_check_fails(self):
        """One non-critical failure → overall should be 'degraded'."""
        checker = self._make_checker()
        with patch.object(checker, "check_latency", return_value=False):
            report = checker.check_all()
        assert report.overall in ("degraded", "unhealthy")

    def test_health_unhealthy_when_critical_check_fails(self):
        """LLM connectivity failure → overall should be 'unhealthy'."""
        checker = self._make_checker()
        with patch.object(checker, "check_llm_connectivity", return_value=False):
            report = checker.check_all()
        assert report.overall == "unhealthy"

    def test_health_unhealthy_when_multiple_failures(self):
        """Three or more check failures → overall should be 'unhealthy'."""
        checker = self._make_checker()
        with patch.object(checker, "check_llm_connectivity", return_value=False), \
             patch.object(checker, "check_vector_db_connectivity", return_value=False), \
             patch.object(checker, "check_latency", return_value=False):
            report = checker.check_all()
        assert report.overall == "unhealthy"

    def test_health_report_contains_timestamp(self):
        """HealthReport should always include a timestamp."""
        checker = self._make_checker()
        report = checker.check_all()
        assert isinstance(report.timestamp, float)
        assert report.timestamp > 0

    def test_cost_within_budget_fails_when_over_budget(self):
        """check_cost_within_budget should fail once budget is exhausted."""
        cost_ctrl = ProductionCostController(CostConfig(total_daily_budget=10.0))
        cost_ctrl.record_cost("user-1", 11.0)  # Exceed budget
        checker = self._make_checker(cost_ctrl=cost_ctrl)
        assert checker.check_cost_within_budget() is False


# ---------------------------------------------------------------------------
# 8. StreamingDeployment tests
# ---------------------------------------------------------------------------

class TestStreamingDeployment:
    def test_configure_streaming_response_headers(self):
        """SSE headers should include all required fields."""
        sd = StreamingDeployment()
        headers = sd.configure_streaming_response(None)
        assert "Cache-Control" in headers
        assert "Connection" in headers
        assert "X-Accel-Buffering" in headers
        assert headers["X-Accel-Buffering"] == "no"

    def test_register_and_complete_stream(self):
        """Registering and completing a stream should update metrics correctly."""
        sd = StreamingDeployment()
        sd.register_stream("conn-1")
        sd.register_stream("conn-2")
        sd.complete_stream("conn-1")
        metrics = sd.get_streaming_metrics()
        assert metrics["active_streams"] == 1
        assert metrics["completed_connections"] == 1

    def test_handle_disconnect(self):
        """Disconnecting a stream should decrement active and increment dropped."""
        sd = StreamingDeployment()
        sd.register_stream("conn-x")
        sd.handle_disconnect("conn-x")
        metrics = sd.get_streaming_metrics()
        assert metrics["active_streams"] == 0
        assert metrics["dropped_connections"] == 1

    @pytest.mark.asyncio
    async def test_handle_backpressure_allows_below_max(self):
        """Backpressure should not block when queue is below max_size."""
        import asyncio
        sd = StreamingDeployment()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        # No items in queue → should return immediately
        done = asyncio.Event()

        async def run():
            await sd.handle_backpressure(q, max_size=10)
            done.set()

        await asyncio.wait_for(run(), timeout=1.0)
        assert done.is_set()
