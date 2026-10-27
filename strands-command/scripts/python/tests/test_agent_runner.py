"""Tests for the agent runner's model and reasoning effort configuration."""
import os
import sys

import pytest

# agent_runner imports its sibling modules by absolute name (as it does when the
# action runs it as a script), so put its directory on the path first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_runner  # noqa: E402


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Start each test with neither override set."""
    monkeypatch.delenv("STRANDS_MODEL_ID", raising=False)
    monkeypatch.delenv("STRANDS_REASONING_EFFORT", raising=False)


def test_defaults_when_unset():
    """No override gives the default model and effort."""
    assert agent_runner._resolve_model_config() == ("global.anthropic.claude-opus-4-8", "high")


def test_override(monkeypatch):
    """Both env vars override the defaults."""
    monkeypatch.setenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-5")
    monkeypatch.setenv("STRANDS_REASONING_EFFORT", "medium")
    assert agent_runner._resolve_model_config() == ("us.anthropic.claude-sonnet-4-5", "medium")


def test_empty_string_falls_back_to_default(monkeypatch):
    """An empty value, as an unset action input or repo variable gives, means the default."""
    monkeypatch.setenv("STRANDS_MODEL_ID", "")
    monkeypatch.setenv("STRANDS_REASONING_EFFORT", "")
    assert agent_runner._resolve_model_config() == ("global.anthropic.claude-opus-4-8", "high")


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_each_known_effort_is_accepted(monkeypatch, effort):
    """Each of the four efforts is accepted."""
    monkeypatch.setenv("STRANDS_REASONING_EFFORT", effort)
    assert agent_runner._resolve_model_config()[1] == effort


def test_unknown_effort_is_refused(monkeypatch):
    """An unknown effort fails with a message that names the allowed values."""
    monkeypatch.setenv("STRANDS_REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError, match="Unknown STRANDS_REASONING_EFFORT 'extreme'. Use one of: low, medium, high, max"):
        agent_runner._resolve_model_config()


def test_unknown_effort_stops_main_before_the_agent_runs(monkeypatch, capsys):
    """main() exits non-zero on an unknown effort and never starts the agent."""
    monkeypatch.setenv("STRANDS_REASONING_EFFORT", "extreme")
    monkeypatch.setattr(sys, "argv", ["agent_runner.py", "review"])
    monkeypatch.setattr(agent_runner, "run_agent", lambda *a: pytest.fail("agent must not run"))
    with pytest.raises(SystemExit) as exit_info:
        agent_runner.main()
    assert exit_info.value.code == 1
    assert "Unknown STRANDS_REASONING_EFFORT" in capsys.readouterr().out


def test_main_prints_the_resolved_model_and_effort(monkeypatch, capsys):
    """main() prints the model and effort once, then runs the agent with them."""
    monkeypatch.setenv("STRANDS_REASONING_EFFORT", "medium")
    monkeypatch.setattr(sys, "argv", ["agent_runner.py", "review"])
    calls = []
    monkeypatch.setattr(agent_runner, "run_agent", lambda *a: calls.append(a))
    agent_runner.main()
    out = capsys.readouterr().out
    assert out.count("model=global.anthropic.claude-opus-4-8 effort=medium") == 1
    assert calls == [("review", "global.anthropic.claude-opus-4-8", "medium")]
