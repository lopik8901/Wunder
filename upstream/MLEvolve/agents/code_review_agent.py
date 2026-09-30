"""Code Review Agent: LLM-based code review and fix for node code."""

import ast
import logging
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import get_internet_clarification
from utils.response import wrap_code

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                "type": "string",
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


def _valid_connectome_revision(original: str, revised: str) -> bool:
    """A review may repair source, but must retain the literal interface."""
    try:
        def assignment_name(source: str) -> str:
            body = ast.parse(source).body
            if len(body) != 1 or not isinstance(body[0], ast.Assign):
                raise ValueError("expected one assignment")
            targets = body[0].targets
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                raise ValueError("expected one assignment")
            return targets[0].id

        name = assignment_name(original)
        if assignment_name(revised) != name:
            return False
        if name == "CANDIDATE":
            from connectome.mlevolve_generated import parse_candidate
            parse_candidate(revised)
        elif name == "EXPERIMENT":
            from connectome.mlevolve_bounded import parse_spec
            parse_spec(revised)
        else:
            return False
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def apply_review_revision(original: str, revision: str, *, connectome_mode: bool,
                          use_diff_mode: bool) -> str:
    """Never pass SEARCH/REPLACE markers to the literal candidate parser."""
    if "<<<<<<< SEARCH" in revision or "< SEARCH" in revision:
        try:
            patched, count = SearchReplacePatcher().apply_patch(revision, original, strict=False)
            if count and patched and patched != original and (
                not connectome_mode or _valid_connectome_revision(original, patched)
            ):
                return patched.strip()
        except Exception as error:
            logger.warning("Code review patch could not be applied: %s", error)
        return original
    if use_diff_mode:
        return original
    if connectome_mode and not _valid_connectome_revision(original, revision):
        logger.warning("Code review returned an invalid literal candidate; retaining original")
        return original
    return revision.strip()


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")

    prompt = get_code_review_prompt(
        task_desc=agent.task_desc,
        code=node.code,
    )
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    if "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification

    connectome_mode = bool(getattr(agent.cfg, "connectome_mode", False))
    if connectome_mode:
        prompt = {
            "Introduction": "Review the bounded Connectome candidate for critical interface, causality, and leakage defects. Preserve MLEvolve's model choice.",
            "Task description": agent.task_desc,
            "Code to review": wrap_code(node.code),
            "Instructions": {"Review scope": [
                "The code must be one literal CANDIDATE or EXPERIMENT assignment.",
                "Return exact SEARCH/REPLACE blocks for corrections, not standalone patch text as candidate code.",
                "Check the documented DataPoint fields, sequence reset, warm-up, source restrictions, and CPU callback contract.",
                "The supervisor alone computes search WP. Candidate code cannot access protected evaluation or network resources.",
            ]},
        }
    use_diff_for_review = agent.acfg.use_diff_mode
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Code review retry attempt {attempt + 1}/{max_retries} for node {node.id}")
                time.sleep(5)

            review_response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=CODE_REVIEW_SPEC,
                    model=agent.acfg.code.model,
                    temperature=agent.acfg.code.temp,
                    cfg=agent.cfg
                ),
            )

            needs_revision = review_response.get("needs_revision", False)
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            logger.info(f"Code review for node {node.id}: needs_revision={needs_revision}")
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})

            if needs_revision:
                if revised_code and revised_code.strip():
                    return apply_review_revision(node.code, revised_code,
                                                 connectome_mode=connectome_mode,
                                                 use_diff_mode=use_diff_for_review)

                if attempt < max_retries - 1:
                    logger.warning(f"Code review violation: needs_revision=True but revised_code is empty/None - Will retry ({attempt + 1}/{max_retries})")
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error(f"Code review violation: needs_revision=True but revised_code is empty/None - Max retries reached, returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                return node.code

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            return node.code

        except Exception as e:
            error_msg = f"Code review failed with exception: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"{error_msg} - Will retry (attempt {attempt + 1}/{max_retries})")
                continue
            logger.error(f"{error_msg} - Max retries reached, returning original code")
            return node.code

    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    return node.code
