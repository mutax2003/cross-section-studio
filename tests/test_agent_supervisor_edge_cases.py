"""Edge-case tests for hierarchical agent orchestration (supervisor + routing)."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SUPERVISOR = SCRIPTS / "agent_supervisor.py"
AGENTS_MD = ROOT / "AGENTS.md"
PROMPT_NAMES = ("scout", "implement", "verify", "review", "summary", "pm", "architect", "security", "ci_triage")


@pytest.fixture
def sup():
    sys.path.insert(0, str(SCRIPTS))
    import agent_supervisor as module

    return module


class TestPromptLoading:
    def test_all_prompt_templates_substitute(self, sup) -> None:
        for name in PROMPT_NAMES:
            text = sup._load_prompt(
                name,
                task="edge task",
                modules="- `pipeline.py`",
                scout_output="scout context",
                diff_summary="diff text",
                diff_stat="diff stat",
                implement_output="impl",
                verify_log="verify",
                review_output="review",
                pm_spec="(none)",
            )
            assert "edge task" in text
            assert "{task}" not in text

    def test_load_prompt_missing_template_raises(self, sup) -> None:
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            sup._load_prompt("nonexistent", task="x", modules="y")

    def test_load_prompt_leaves_unknown_placeholders(self, sup) -> None:
        text = sup._load_prompt("scout", task="only-task", modules="only-modules")
        assert "{scout_output}" not in text  # scout template has no scout_output


class TestModuleFormatting:
    def test_format_modules_empty_uses_routing_hint(self, sup) -> None:
        assert "AGENTS.md" in sup._format_modules([])

    def test_parse_modules_strips_whitespace_and_skips_blanks(self, sup) -> None:
        assert sup._parse_modules(" renderer.py , , pipeline.py ") == [
            "renderer.py",
            "pipeline.py",
        ]

    def test_parse_modules_empty_string(self, sup) -> None:
        assert sup._parse_modules("") == []
        assert sup._parse_modules("  , , ") == []


class TestModuleBoundaryWarnings:
    def test_empty_modules_no_warnings(self, sup) -> None:
        assert sup.module_boundary_warnings([]) == []

    def test_forbidden_core_triple(self, sup) -> None:
        warnings = sup.module_boundary_warnings(
            ["app.py", "pipeline.py", "stratigraphy.py"],
        )
        assert any("app.py + pipeline.py + stratigraphy.py" in item for item in warnings)

    def test_stratigraphy_and_renderer(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["stratigraphy.py", "renderer.py"])
        assert any("stratigraphy.py and renderer" in item for item in warnings)

    def test_ui_and_engine_core(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app.py", "pipeline.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_app_build_and_engine_core(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app_build.py", "pipeline.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_ui_presets_and_renderer(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["ui_output_presets.py", "renderer.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_projection_and_renderer(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["projection.py", "renderer_common.py"])
        assert any("projection.py and renderer" in item for item in warnings)

    def test_ops_and_engine(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["ops_auth.py", "pipeline.py"])
        assert any("ops_" in item and "engine core" in item for item in warnings)

    def test_ai_assistant_and_engine_core(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["ai_assistant.py", "pipeline.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_section_build_request_and_ui(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app_build.py", "section_build_request.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_transect_planner_and_ui(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app_configure.py", "transect_planner.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_ai_quality_and_ui(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app_validate.py", "ai_quality.py"])
        assert any("UI" in item and "engine core" in item for item in warnings)

    def test_svg_first_triad_app_services_and_pipeline(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app_services.py", "pipeline.py"])
        assert not any("UI" in item and "engine core" in item for item in warnings)

    def test_svg_first_triad_app_services_and_section_build_request(self, sup) -> None:
        warnings = sup.module_boundary_warnings(["app_services.py", "section_build_request.py"])
        assert not any("UI" in item and "engine core" in item for item in warnings)

    def test_svg_first_triad_all_three(self, sup) -> None:
        warnings = sup.module_boundary_warnings(
            ["app_services.py", "pipeline.py", "section_build_request.py"],
        )
        assert not any("UI" in item and "engine core" in item for item in warnings)

    def test_orchestrate_prints_boundary_warning(self, sup, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        args = argparse.Namespace(
            api_key=None,
            modules="stratigraphy.py,renderer.py",
            task="risky scope",
            runtime="local",
            model="composer-2.5",
            skip_scout=True,
            skip_implement=True,
            review=False,
            summary_agent=False,
        )
        client = _mock_async_client(phases=[])

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=client)
            with patch.object(sup, "run_verify_with_log", return_value=(0, "- **overall**: PASS")):
                asyncio.run(sup._orchestrate_async(args))

        captured = capsys.readouterr()
        assert "BOUNDARY WARNING" in captured.out
        assert "stratigraphy.py and renderer" in captured.out


class TestApiKeyResolution:
    def test_resolve_api_key_from_explicit(self, sup) -> None:
        assert sup._resolve_api_key("cursor_test_key") == "cursor_test_key"

    def test_resolve_api_key_from_env(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "  env_key  ")
        assert sup._resolve_api_key(None) == "env_key"

    def test_resolve_api_key_missing_exits(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="CURSOR_API_KEY"):
            sup._resolve_api_key(None)


class TestAgentOptions:
    def test_build_local_options_sets_cwd(self, sup) -> None:
        options = sup._build_agent_options(
            api_key="k",
            runtime="local",
            model="composer-2.5",
            name="test-agent",
        )
        assert options.local is not None
        assert Path(options.local.cwd) == sup.ROOT

    def test_build_cloud_options_requires_origin(self, sup) -> None:
        with patch.object(sup.subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")
            with pytest.raises(SystemExit, match="origin remote"):
                sup._build_agent_options(
                    api_key="k",
                    runtime="cloud",
                    model="composer-2.5",
                    name="test-agent",
                )

    def test_build_cloud_options_uses_git_remote_and_branch(self, sup) -> None:
        with patch.object(sup.subprocess, "run") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(returncode=0, stdout="https://github.com/org/repo.git\n"),
                SimpleNamespace(returncode=0, stdout="feature/orch\n"),
            ]
            options = sup._build_agent_options(
                api_key="k",
                runtime="cloud",
                model="composer-2.5",
                name="test-agent",
            )
        assert options.cloud is not None
        assert options.cloud.repos[0].url == "https://github.com/org/repo.git"
        assert options.cloud.repos[0].starting_ref == "feature/orch"

    def test_build_unknown_runtime_exits(self, sup) -> None:
        with pytest.raises(SystemExit, match="Unknown runtime"):
            sup._build_agent_options(
                api_key="k",
                runtime="invalid",
                model="composer-2.5",
                name="test-agent",
            )


class TestRunResultStatus:
    @pytest.mark.parametrize(
        ("status", "failed"),
        [
            ("error", True),
            ("finished", False),
            ("ERROR", False),
            ("RunResultStatus.error", True),
        ],
    )
    def test_run_result_failed_edge_cases(self, sup, status, failed: bool) -> None:
        result = SimpleNamespace(status=status)
        assert sup._run_result_failed(result) is failed


class TestGitDiffSummary:
    def test_git_diff_prefers_worktree(self, sup) -> None:
        with patch.object(sup.subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="diff chunk")
            assert sup._git_diff_summary() == "diff chunk"
            mock_run.assert_called_once()

    def test_git_diff_falls_back_to_staged(self, sup) -> None:
        with patch.object(sup.subprocess, "run") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(returncode=0, stdout=""),
                SimpleNamespace(returncode=0, stdout="staged diff"),
            ]
            assert sup._git_diff_summary() == "staged diff"

    def test_git_diff_clean_tree_message(self, sup) -> None:
        with patch.object(sup.subprocess, "run") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(returncode=0, stdout=""),
                SimpleNamespace(returncode=0, stdout=""),
            ]
            assert "no git diff" in sup._git_diff_summary()

    def test_git_diff_truncates_large_output(self, sup) -> None:
        huge = "x" * 20000
        with patch.object(sup.subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout=huge)
            summary = sup._git_diff_summary()
        assert len(summary) < len(huge)
        assert "truncated" in summary


class TestVerifyLocal:
    def test_run_verify_local_stops_on_first_failure(self, sup) -> None:
        calls: list[list[str]] = []

        def fake_run(command, cwd, check):  # noqa: ANN001
            calls.append(command)
            return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

        with patch.object(sup.subprocess, "run", side_effect=fake_run):
            code = sup.run_verify_local()

        assert code == 1
        assert len(calls) == 2

    def test_run_verify_local_runs_all_three_on_success(self, sup) -> None:
        with patch.object(
            sup.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as mock_run:
            assert sup.run_verify_local() == 0
        assert mock_run.call_count == 3


class TestCliEdgeCases:
    def test_run_verify_only_skips_api_key(self, sup) -> None:
        args = argparse.Namespace(
            verify_only=True,
            report=None,
            task="verify task",
            modules="",
            runtime="local",
            model=sup.DEFAULT_MODEL,
        )
        with patch.object(sup, "run_verify_with_log", return_value=(0, "- **overall**: PASS")) as mock_verify:
            with patch.object(sup.asyncio, "run") as mock_async:
                assert sup._cmd_run(args) == 0
        mock_verify.assert_called_once()
        mock_async.assert_not_called()

    def test_run_without_verify_only_uses_async_orchestrator(self, sup) -> None:
        args = argparse.Namespace(
            verify_only=False,
            report=None,
            task="t",
            modules="",
            runtime="local",
            model=sup.DEFAULT_MODEL,
            skip_scout=True,
            skip_implement=True,
            review=False,
            summary_agent=False,
            api_key="k",
        )

        def _consume_coro(coro):
            coro.close()
            return (
                0,
                sup.RunReport(task="t", modules=[], runtime="local", model=sup.DEFAULT_MODEL),
            )

        with patch.object(sup.asyncio, "run", side_effect=_consume_coro) as mock_async:
            assert sup._cmd_run(args) == 0
        mock_async.assert_called_once()


def _mock_async_client(*, phases: list[str]) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None

    async def create_agent(_options):  # noqa: ANN001
        agent = AsyncMock()
        agent.agent_id = f"agent-{phases.pop(0) if phases else 'done'}"
        run = AsyncMock()
        run.id = f"run-{agent.agent_id}"
        result = SimpleNamespace(
            status="finished",
            agent_id=agent.agent_id,
            id=run.id,
            result=f"output-{agent.agent_id}",
        )

        async def wait():
            return result

        run.wait = wait
        agent.send = AsyncMock(return_value=run)
        agent.close = AsyncMock()
        return agent

    client.create_agent.side_effect = create_agent
    return client


class TestAsyncOrchestration:
    def test_orchestrate_runs_scout_implement_verify(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        args = argparse.Namespace(
            api_key=None,
            modules="pipeline.py",
            task="test orchestration",
            runtime="local",
            model="composer-2.5",
            skip_scout=False,
            skip_implement=False,
            review=False,
            summary_agent=False,
        )
        client = _mock_async_client(phases=["scout", "implement"])

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=client)
            with patch.object(sup, "run_verify_with_log", return_value=(0, "- **pytest**: PASS\n- **overall**: PASS")):
                exit_code, report = asyncio.run(sup._orchestrate_async(args))

        assert exit_code == 0
        assert report.verify_exit_code == 0
        assert client.create_agent.await_count == 2

    def test_orchestrate_verify_failure_skips_review(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        args = argparse.Namespace(
            api_key=None,
            modules="",
            task="review should not run",
            runtime="local",
            model="composer-2.5",
            skip_scout=True,
            skip_implement=True,
            review=True,
            summary_agent=False,
        )
        client = _mock_async_client(phases=[])

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=client)
            with patch.object(sup, "run_verify_with_log", return_value=(2, "- **pytest**: FAIL (exit 2)")):
                exit_code, report = asyncio.run(sup._orchestrate_async(args))

        assert exit_code == 2
        assert report.verify_exit_code == 2
        client.create_agent.assert_not_awaited()

    def test_orchestrate_review_after_verify_pass(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        args = argparse.Namespace(
            api_key=None,
            modules="",
            task="post-verify review",
            runtime="local",
            model="composer-2.5",
            skip_scout=True,
            skip_implement=True,
            review=True,
            summary_agent=False,
        )
        client = _mock_async_client(phases=["review"])

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=client)
            with patch.object(sup, "run_verify_with_log", return_value=(0, "- **overall**: PASS")):
                with patch.object(sup, "_git_diff_summary", return_value="tiny diff"):
                    exit_code, report = asyncio.run(sup._orchestrate_async(args))

        assert exit_code == 0
        assert report.phases[-1].phase == "review"
        assert client.create_agent.await_count == 1

    def test_orchestrate_summary_agent_after_verify(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        args = argparse.Namespace(
            api_key=None,
            modules="renderer.py",
            task="summarize run",
            runtime="local",
            model="composer-2.5",
            skip_scout=True,
            skip_implement=True,
            review=False,
            summary_agent=True,
        )
        client = _mock_async_client(phases=["summary"])

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=client)
            with patch.object(sup, "run_verify_with_log", return_value=(0, "- **overall**: PASS")):
                exit_code, report = asyncio.run(sup._orchestrate_async(args))

        assert exit_code == 0
        assert report.summary_body.startswith("output-agent-summary")
        assert report.phases[-1].phase == "summary"
        assert client.create_agent.await_count == 1

    def test_orchestrate_agent_run_error_exits(self, sup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        args = argparse.Namespace(
            api_key=None,
            modules="",
            task="failing scout",
            runtime="local",
            model="composer-2.5",
            skip_scout=False,
            skip_implement=True,
            review=False,
            summary_agent=False,
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None

        agent = AsyncMock()
        agent.agent_id = "agent-scout"
        run = AsyncMock()
        run.id = "run-scout"
        run.wait = AsyncMock(
            return_value=SimpleNamespace(
                status="error",
                agent_id="agent-scout",
                id="run-scout",
                result="",
            )
        )
        agent.send = AsyncMock(return_value=run)
        agent.close = AsyncMock()
        client.create_agent = AsyncMock(return_value=agent)

        with patch("cursor_sdk.AsyncClient") as mock_client_cls:
            mock_client_cls.launch_bridge = AsyncMock(return_value=client)
            with pytest.raises(SystemExit, match="phase=scout run failed"):
                asyncio.run(sup._orchestrate_async(args))

        agent.close.assert_awaited_once()


class TestAgentsRoutingTable:
    """AGENTS.md routing entries should reference real modules/tests."""

    @staticmethod
    def _routing_tokens() -> list[str]:
        text = AGENTS_MD.read_text(encoding="utf-8")
        section = text.split("## Task routing", 1)[1].split("## E2E quality gate", 1)[0]
        tokens: list[str] = []
        for raw in section.replace("`", " ").replace("*", " ").split():
            token = raw.strip().rstrip(",.")
            if not token or token == ".py" or "*" in token:
                continue
            if token.endswith(".py") or token.startswith("tests/") or token.startswith("docs/"):
                tokens.append(token)
        return sorted(set(tokens))

    def test_routing_table_targets_exist(self) -> None:
        missing: list[str] = []
        for token in self._routing_tokens():
            path = ROOT / token.replace("/", os.sep)
            if not path.exists():
                missing.append(token)
        assert not missing, f"Missing routing targets: {missing}"

    def test_verify_commands_list_matches_supervisor(self, sup) -> None:
        labels = [label for label, _ in sup.VERIFY_COMMANDS]
        assert labels == ["pytest", "e2e_smoke_direct", "smoke_test"]
