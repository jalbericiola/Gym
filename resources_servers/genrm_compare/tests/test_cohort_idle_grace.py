"""Cohort completion behaviour for GenRM verify().

Cohorts routinely never reach num_rollouts_per_prompt: peers fail upstream and
are replaced by placeholders that never call verify(), and prompt_key hashes only
(prompt, principle) so two concurrent groups on the same prompt share a buffer and
the second is left with unfillable leftovers. Before the idle grace, every such
cohort blocked its waiters for the full cohort_timeout_s (1800s in production).
"""
import asyncio
import time

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

from resources_servers.genrm_compare import app as genrm_app


class _Harness:
    """Drives the real verify() cohort path with scoring stubbed out."""

    def __init__(self, monkeypatch, *, expected=16, grace=0.2, hard=30.0):
        self.scored = []
        genrm_app._cohort_buffers.clear()
        genrm_app._cohort_last_arrival.clear()
        genrm_app._cohort_expected.clear()
        genrm_app._cohort_decrements.clear()

        async def fake_score_cohort(_self, cohort_buf, principle):
            self.scored.append(len(cohort_buf))
            for _, fut in cohort_buf:
                if not fut.done():
                    fut.set_result(1.0)

        monkeypatch.setattr(
            genrm_app.GenRMCompareResourcesServer, "_score_cohort", fake_score_cohort
        )

        # The cohort mechanics are what is under test; bypass the response schema
        # so the stub bodies below do not have to satisfy NeMoGymResponse.
        class _Resp:
            def __init__(self, **kw):
                self.reward = kw.get("reward")

        monkeypatch.setattr(genrm_app, "BaseVerifyResponse", _Resp)
        self.cfg = genrm_app.GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[], max_output_tokens=1024
            ),
            num_rollouts_per_prompt=expected,
            cohort_idle_grace_s=grace,
            cohort_timeout_s=hard,
        )


def _body(text="hello", group_id=None, expected=None):
    class _B:
        principle = None
        response = {"output_text": text}

        class responses_create_params:
            input = [{"role": "user", "content": "same-prompt"}]

    if group_id is not None:
        _B.cohort_group_id = group_id
    if expected is not None:
        _B.cohort_expected_size = expected
    return _B()


@pytest.mark.asyncio
async def test_full_cohort_scores_immediately(monkeypatch):
    """All N arrivals -> scored at once, without paying the idle grace."""
    h = _Harness(monkeypatch, expected=4, grace=5.0)
    srv = genrm_app.GenRMCompareResourcesServer.__new__(genrm_app.GenRMCompareResourcesServer)
    object.__setattr__(srv, "config", h.cfg)

    t0 = time.monotonic()
    await asyncio.gather(*[srv.verify(_body(f"r{i}")) for i in range(4)])
    elapsed = time.monotonic() - t0

    assert h.scored == [4], f"expected one 4-way scoring, got {h.scored}"
    assert elapsed < 4.0, f"full cohort waited {elapsed:.1f}s; should not pay the grace"


@pytest.mark.asyncio
async def test_partial_cohort_scores_after_idle_grace_not_hard_timeout(monkeypatch):
    """k < N arrivals -> scored ~grace later, NOT after cohort_timeout_s."""
    h = _Harness(monkeypatch, expected=16, grace=0.3, hard=30.0)
    srv = genrm_app.GenRMCompareResourcesServer.__new__(genrm_app.GenRMCompareResourcesServer)
    object.__setattr__(srv, "config", h.cfg)

    t0 = time.monotonic()
    await asyncio.gather(*[srv.verify(_body(f"r{i}")) for i in range(3)])
    elapsed = time.monotonic() - t0

    assert h.scored == [3], f"expected one 3-way scoring, got {h.scored}"
    assert elapsed < 5.0, f"partial cohort took {elapsed:.1f}s; idle grace did not fire"
    assert elapsed >= 0.3, "scored before the grace elapsed"


@pytest.mark.asyncio
async def test_cohort_scored_exactly_once(monkeypatch):
    """Concurrent waiters must not double-score the same cohort."""
    h = _Harness(monkeypatch, expected=16, grace=0.2, hard=30.0)
    srv = genrm_app.GenRMCompareResourcesServer.__new__(genrm_app.GenRMCompareResourcesServer)
    object.__setattr__(srv, "config", h.cfg)

    await asyncio.gather(*[srv.verify(_body(f"r{i}")) for i in range(8)])

    assert len(h.scored) == 1, f"cohort scored {len(h.scored)} times: {h.scored}"
    assert genrm_app._cohort_buffers == {}, "cohort buffer leaked"
    assert genrm_app._cohort_last_arrival == {}, "arrival timestamps leaked"


@pytest.mark.asyncio
async def test_group_ids_separate_cohorts_on_same_prompt(monkeypatch):
    """Two stamped groups sharing a prompt must not mix (prompt_key would)."""
    h = _Harness(monkeypatch, expected=16, grace=0.0, hard=30.0)
    srv = genrm_app.GenRMCompareResourcesServer.__new__(genrm_app.GenRMCompareResourcesServer)
    object.__setattr__(srv, "config", h.cfg)

    await asyncio.gather(
        *[srv.verify(_body(f"a{i}", group_id="gA", expected=3)) for i in range(3)],
        *[srv.verify(_body(f"b{i}", group_id="gB", expected=3)) for i in range(3)],
    )
    assert sorted(h.scored) == [3, 3], f"expected two 3-way scorings, got {h.scored}"


@pytest.mark.asyncio
async def test_decrement_completes_partial_group(monkeypatch):
    """expected=4, 3 arrive, 1 decrement -> scores the 3 immediately (no grace)."""
    h = _Harness(monkeypatch, expected=16, grace=0.0, hard=30.0)
    srv = genrm_app.GenRMCompareResourcesServer.__new__(genrm_app.GenRMCompareResourcesServer)
    object.__setattr__(srv, "config", h.cfg)

    waiters = [
        asyncio.create_task(srv.verify(_body(f"r{i}", group_id="gD", expected=4)))
        for i in range(3)
    ]
    await asyncio.sleep(0.1)
    assert h.scored == [], "must not score before the decrement"

    t0 = time.monotonic()
    result = await srv.cohort_decrement(
        genrm_app.CohortDecrementRequest(cohort_group_id="gD", count=1)
    )
    await asyncio.gather(*waiters)
    elapsed = time.monotonic() - t0

    assert result["completed"] is True
    assert h.scored == [3], f"expected one 3-way scoring, got {h.scored}"
    assert elapsed < 2.0, f"decrement completion took {elapsed:.1f}s"
    assert genrm_app._cohort_buffers == {} and genrm_app._cohort_decrements == {}


@pytest.mark.asyncio
async def test_decrement_banked_before_first_arrival(monkeypatch):
    """Decrement arriving first is banked; group completes at expected-1."""
    h = _Harness(monkeypatch, expected=16, grace=0.0, hard=30.0)
    srv = genrm_app.GenRMCompareResourcesServer.__new__(genrm_app.GenRMCompareResourcesServer)
    object.__setattr__(srv, "config", h.cfg)

    result = await srv.cohort_decrement(
        genrm_app.CohortDecrementRequest(cohort_group_id="gE", count=1)
    )
    assert result["completed"] is False

    await asyncio.gather(
        *[srv.verify(_body(f"r{i}", group_id="gE", expected=3)) for i in range(2)]
    )
    assert h.scored == [2], f"expected one 2-way scoring, got {h.scored}"
