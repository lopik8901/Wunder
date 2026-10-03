import logging
from typing import Any, List, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.prompts import prompt_resp_fmt, get_impl_guideline_from_agent
from agents.planner import build_chat_prompt_for_model
from agents.coder import plan_and_code_query

from engine.conditions import should_trigger_branch_fusion  # noqa: F401
from agents.triggers import register_node

logger = logging.getLogger("MLEvolve")


def _collect_branch_representatives(agent) -> List[SearchNode]:
    representatives = []

    for branch_id, successful_nodes in agent.branch_successful_nodes.items():
        if not successful_nodes or len(successful_nodes) == 0:
            logger.debug(f"Branch {branch_id} has no successful nodes, skipping")
            continue

        maximize = agent.metric_maximize if agent.metric_maximize is not None else True
        branch_best = max(
            successful_nodes,
            key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
                float("-inf") if maximize else float("inf")
            ),
        )

        if not branch_best.metric or branch_best.metric.value is None:
            logger.debug(f"Branch {branch_id} best node has no valid metric, skipping")
            continue

        representatives.append(branch_best)

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True
    representatives.sort(
        key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
            float("-inf") if maximize else float("inf")
        ),
        reverse=maximize,
    )

    logger.info(
        f"Collected {len(representatives)} branch representatives "
        f"from {len(agent.branch_successful_nodes)} successful solutions"
    )
    return representatives


def run(
    agent,
    mode: str = "node",
    parent_node: Optional[SearchNode] = None,
) -> Optional[SearchNode]:

    if parent_node and not agent.is_root(parent_node):
        logger.error(
            f"_aggregation() should only be called from root node! Got parent_node: {parent_node.id}"
        )
        return None

    if agent.fusion_draft_count >= agent.max_fusion_drafts:
        logger.info(
            f"Max fusion drafts ({agent.max_fusion_drafts}) reached, skipping aggregation"
        )
        return None

    branch_representatives = _collect_branch_representatives(agent)
    if len(branch_representatives) < 2:
        logger.info("Not enough successful branches for aggregation")
        return None

    introduction = (
        "You are a Kaggle grandmaster attending a competition. "
        "You are provided with multiple successful solutions from different independent branches below. "
        "Your task is to synthesize these diverse approaches and create a completely NEW solution "
        "that draws inspiration from their strengths. "
        "This is a fresh start to spark new ideas by combining insights from different successful directions."
    )

    reference_summaries = []
    if mode == "node":
        for i, node in enumerate(branch_representatives):
            trajectory = node.generate_node_trajectory(need_code=False)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Best Solution** (Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)
    elif mode == "trajectory":
        for i, node in enumerate(branch_representatives):
            trajectory = node.get_root_to_current_trajectory(max_steps=6)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Evolution Path** (Best Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)
    else:
        logger.warning(f"Unknown aggregation mode: {mode}, using node mode as default")
        for i, node in enumerate(branch_representatives):
            trajectory = node.generate_node_trajectory(need_code=False)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Best Solution** (Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)

    reference_experiences = "\n" + "-" * 80 + "\n".join(reference_summaries)

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Branch Experiences": reference_experiences,
        "Instructions": {},
    }

    prompt["Instructions"] |= prompt_resp_fmt()

    if mode == "node":
        prompt["Instructions"] |= {
            "Multi-branch aggregation guideline (Node Mode)": [
                "- You are provided with the BEST solutions from different independent branches.",
                "- Analyze what makes each branch's final solution successful - their key techniques and approaches.",
                "- This is NOT about improving a current solution - this is about creating a FRESH NEW approach.",
                "- Think creatively: how can you synthesize the strengths of different final solutions into an innovative approach?",
                "- Write a brief natural language description of your NEW synthesized approach.",
                "- The solution should be distinct and innovative, combining the best ideas in a novel way.",
                "- Focus on discovering new synergies between successful techniques from different branches.",
                "- The final code should be a single, runnable Python script.",
                "- Do not suggest to do EDA.",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "Multi-branch aggregation guideline (Trajectory Mode)": [
                "- You are provided with the EVOLUTION PATHS of different independent branches.",
                "- Analyze how each branch evolved from initial ideas to their best solutions - what worked and what didn't.",
                "- Learn from the successful improvement patterns and evolution strategies across branches.",
                "- This is NOT about improving a current solution - this is about creating a FRESH NEW approach.",
                "- Think creatively: what new directions emerge from understanding these different evolution paths?",
                "- Write a brief natural language description of your NEW synthesized approach.",
                "- The solution should be distinct and innovative, inspired by successful evolution patterns.",
                "- Focus on discovering unexplored directions suggested by the evolution insights from multiple branches.",
                "- The final code should be a single, runnable Python script.",
                "- Do not suggest to do EDA.",
            ],
        }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)

    if getattr(agent.cfg, "connectome_mode", False):
        from connectome.mlevolve_history import research_plan_guideline
        introduction = "You are MLEvolve. Use only search-only branch evidence to select the next causal Connectome experiment."
        prompt["Instructions"] = {
            "Response format": "Give a concise hypothesis and one Python code block containing exactly one complete literal CANDIDATE or EXPERIMENT assignment.",
            **get_impl_guideline_from_agent(agent),
            **research_plan_guideline(),
        }
    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    data_preview = getattr(agent, "data_preview", "") or ""
    assistant_prefix = (
        "Let me approach this systematically.\n"
        f"First, I'll examine the dataset:\n{data_preview}\n"
        "I have access to multiple successful approaches from different independent branches. "
        "I'll synthesize these diverse insights and create a completely new solution "
        "that combines the best ideas in an innovative way."
    )
    if getattr(agent.cfg, "connectome_mode", False):
        assistant_prefix = (
            "I will compare the search-only branch evidence and fixed error atlas, "
            "identify a specific remaining error, and choose a testable hypothesis. "
            "I may refine a known approach or branch to a different one."
        )

    user_prompt = (
        f"\n# Task description\n{prompt['Task description']}\n\n"
        f"# Branch Experiences\n{prompt['Branch Experiences']}\n\n{instructions}"
    )
    if getattr(agent.cfg, "connectome_mode", False):
        from connectome.mlevolve_history import planner_history_section
        user_prompt += planner_history_section(agent)
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    research_trace = None
    if getattr(agent.cfg, "connectome_mode", False):
        from connectome.research_planning import plan_with_research
        plan, code, prompt_complete, research_trace = plan_with_research(
            agent, prompt_complete, agent.virtual_root.id, "aggregation", plan_and_code_query)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    aggregation_node = SearchNode(
        plan=plan,
        code=code,
        parent=agent.virtual_root,
        stage="fusion_draft",
        local_best_node=agent.virtual_root,
        connectome_research_trace=research_trace,
    )
    register_node(agent, aggregation_node, prompt_complete, new_branch=True)
    agent.fusion_draft_count += 1

    logger.info(f"[aggregation] → node {aggregation_node.id} (branch={aggregation_node.branch_id})")
    return aggregation_node
