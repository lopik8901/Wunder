"""Run one evaluated MLEvolve node, generate its next node, then stop."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MLEvolve"))

from config import load_cfg, load_task_desc, prep_agent_workspace, save_run  # noqa: E402
from connectome.mlevolve_adapter import adapt_execution_result  # noqa: E402
from engine.agent_search import AgentSearch  # noqa: E402
from engine.executor import Interpreter  # noqa: E402
from engine.search_node import Journal  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402
from utils.seed import set_global_seed  # noqa: E402


def _executed_node(returned, journal):
    return journal.nodes[-1] if returned.stage == "root" and len(journal.nodes) > 1 else returned


def _expose_repository(root: Path) -> None:
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")


def main() -> None:
    cfg = load_cfg()
    if not cfg.connectome_mode or cfg.llm_backend != "codex":
        raise ValueError("smoke runner requires connectome_mode=true and llm_backend=codex")
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    _expose_repository(ROOT)
    prep_agent_workspace(cfg)

    journal = Journal()
    agent = AgentSearch(load_task_desc(cfg), cfg, journal)
    interpreter = Interpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg)

    def execute(code, node_id, reset_session=True):
        return adapt_execution_result(interpreter.run(code, node_id, reset_session))

    try:
        first = agent.step(node=None, exec_callback=execute)
        save_run(cfg, journal)
        while first.is_buggy and first.debug_depth < cfg.agent.search.max_debug_depth:
            logger.info("Candidate #1 execution failed; invoking MLEvolve's native debug action.")
            first = _executed_node(agent.step(node=first, exec_callback=execute), journal)
            save_run(cfg, journal)
        if first.is_buggy or first.metric.value is None:
            raise RuntimeError(f"candidate #1 failed after native debugging: {first.analysis}")

        second = agent.step(node=first, exec_callback=execute, execute_immediately=False)
        proposal = {
            "id": second.id,
            "stage": second.stage,
            "parent_id": second.parent.id if second.parent else None,
            "branch_id": second.branch_id,
            "plan": second.plan,
            "code": second.code,
            "prompt_input": second.prompt_input,
            "pending_execution": bool(getattr(second, "pending_execution", False)),
        }
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        (cfg.log_dir / "candidate_2_proposal.json").write_text(
            json.dumps(proposal, indent=2), encoding="utf-8"
        )
        (cfg.log_dir / "candidate_2.py").write_text(second.code, encoding="utf-8")
        logger.info(
            "Acceptance boundary reached: candidate #1 evaluated as %s; candidate #2 %s generated and not executed.",
            first.metric.value, second.id,
        )
        print(json.dumps({
            "run_dir": str(cfg.log_dir.parent),
            "candidate_1_id": first.id,
            "candidate_1_metric": first.metric.value,
            "candidate_1_connectome_metrics": first.connectome_metrics,
            "candidate_1_runtime": first.exec_time,
            "candidate_2_id": second.id,
            "candidate_2_stage": second.stage,
            "candidate_2_parent_id": proposal["parent_id"],
            "candidate_2_pending_execution": proposal["pending_execution"],
        }, indent=2))
    finally:
        interpreter.terminate_all_subprocesses()
        interpreter.cleanup_session(-1)


if __name__ == "__main__":
    main()
