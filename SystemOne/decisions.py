"""Centralized System One questions and confidence gates.

Open-ended text generation remains with the LLMs; this module only owns bounded
judgments (Choice / Score / Noul).
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from .client import get_system_one_client
from .config import env_value


QUERY_SEARCH_CRITERIA = {
    "basic_search_news": "General-purpose news/web research with no stronger special requirement.",
    "deep_search_news": "Broad background, multiple perspectives, or deeper context is needed.",
    "search_news_last_24_hours": "Breaking/today/current developments where the last 24 hours matter.",
    "search_news_last_week": "Recent developments over roughly the last seven days.",
    "search_images_for_news": "Images or other visual evidence are materially required.",
}
MEDIA_SEARCH_CRITERIA = {
    "comprehensive_search": "Broad multimodal/web research is needed.",
    "web_search_only": "Raw web pages/snippets are enough; provider-side synthesis is unnecessary.",
    "search_for_structured_data": "Structured facts/cards such as weather, stocks, or exchange rates are the target.",
    "search_last_24_hours": "Breaking/current developments in the last 24 hours are the target.",
    "search_last_week": "Recent developments over roughly the last seven days are the target.",
}
INSIGHT_SEARCH_CRITERIA = {
    "search_hot_content": "Find the hottest/highest-engagement recent content.",
    "search_topic_globally": "Search the whole social database for a topic across platforms.",
    "search_topic_by_date": "Search a topic inside an explicit historical date range.",
    "get_comments_for_topic": "Audience/user comments and reactions are the main evidence needed.",
    "search_topic_on_platform": "One specific social platform is explicitly required.",
}
PLATFORM_CRITERIA = {
    "none": "No single platform is required.",
    "bilibili": "Bilibili is explicitly requested or uniquely relevant.",
    "weibo": "Weibo is explicitly requested or uniquely relevant.",
    "douyin": "Douyin is explicitly requested or uniquely relevant.",
    "kuaishou": "Kuaishou is explicitly requested or uniquely relevant.",
    "xhs": "Xiaohongshu/XHS is explicitly requested or uniquely relevant.",
    "zhihu": "Zhihu is explicitly requested or uniquely relevant.",
    "tieba": "Baidu Tieba is explicitly requested or uniquely relevant.",
}
TIME_PERIOD_CRITERIA = {
    "24h": "The most recent 24 hours are the relevant window.",
    "week": "The last seven days are the relevant window.",
    "year": "A broad recent-history window up to roughly a year is needed.",
}

EVIDENCE_RELEVANCE_LEVELS = [
    "Unrelated to the section/question.",
    "Tangentially related but not useful.",
    "Directly relevant and useful supporting evidence.",
    "Core evidence that directly addresses a central claim.",
]
EVIDENCE_VALUE_LEVELS = [
    "No factual support; mostly noise, promotion, or unsupported opinion.",
    "Weak factual value or poorly attributable evidence.",
    "Useful secondary evidence with concrete facts or attributable claims.",
    "High-value evidence: direct, specific, primary/authoritative, or strongly corroborative.",
]
EVIDENCE_NOVELTY_LEVELS = [
    "Duplicates information already represented by other results.",
    "Mostly repetitive with only minor extra detail.",
    "Adds a useful distinct detail, source, or perspective.",
    "Adds materially new evidence, data, contradiction, or an important missing angle.",
]
CHAPTER_IMPORTANCE_LEVELS = [
    "Peripheral; can be very brief without harming the report.",
    "Useful supporting chapter but not central.",
    "Important to the report's main argument.",
    "Critical chapter; one of the report's main analytical pillars.",
]
CHAPTER_EVIDENCE_LEVELS = [
    "Very little usable evidence is available.",
    "Some evidence exists but coverage is thin.",
    "Good evidence density with multiple useful facts/sources.",
    "Very high evidence density with substantial material worth synthesizing.",
]
CHAPTER_COMPLEXITY_LEVELS = [
    "Simple and narrow; little explanation required.",
    "Moderate complexity with a few dimensions.",
    "Complex, multi-factor analysis requiring careful explanation.",
    "Highly complex with many dimensions, caveats, or comparisons.",
]

_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def _float_env(name: str, default: float) -> float:
    try:
        return float(env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def _answers(result: Optional[Dict[str, Any]]) -> Any:
    if not result:
        return None
    answers = result.get("answers")
    if answers is None and isinstance(result.get("data"), dict):
        answers = result["data"].get("answers")
    return answers


def _answer(result: Optional[Dict[str, Any]], question_id: str) -> Optional[Dict[str, Any]]:
    answers = _answers(result)
    if isinstance(answers, dict):
        value = answers.get(question_id)
        if isinstance(value, dict):
            return value
    return None


def _choice_value(answer: Optional[Dict[str, Any]]) -> Optional[str]:
    if not answer:
        return None
    value = answer.get("choice")
    return value if isinstance(value, str) and value else None


def _confidence(answer: Optional[Dict[str, Any]], choice: Optional[str] = None) -> float:
    if not answer:
        return 0.0
    raw = answer.get("confidence")
    if isinstance(raw, (int, float)):
        return float(raw)
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        if choice and isinstance(probabilities.get(choice), (int, float)):
            return float(probabilities[choice])
        numeric = [float(v) for v in probabilities.values() if isinstance(v, (int, float))]
        if numeric:
            return max(numeric)
    return 0.0


def _noul_value(answer: Optional[Dict[str, Any]]) -> Optional[float]:
    if not answer:
        return None
    value = answer.get("noul")
    return float(value) if isinstance(value, (int, float)) else None


def _score_value(answer: Optional[Dict[str, Any]]) -> Optional[float]:
    if not answer:
        return None
    value = answer.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def extract_explicit_date_range(text: str) -> Optional[Tuple[str, str]]:
    found: List[str] = []
    for raw in _DATE_RE.findall(text or ""):
        try:
            datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            continue
        found.append(raw)
        if len(found) == 2:
            break
    if len(found) != 2:
        return None
    return tuple(sorted(found))


def choose_query_search_tool(*, input_data: Any, generated_query: str, phase: str) -> Optional[Dict[str, Any]]:
    input_dict = input_data if isinstance(input_data, dict) else {"input": str(input_data)}
    combined = " ".join(str(input_dict.get(k, "")) for k in ("title", "content", "paragraph_latest_state"))
    date_range = extract_explicit_date_range(combined)
    criteria = dict(QUERY_SEARCH_CRITERIA)
    if date_range:
        criteria["search_news_by_date"] = f"Evidence is explicitly bounded to {date_range[0]} through {date_range[1]}."

    result = get_system_one_client().evaluate(
        state={"phase": phase, "section": input_dict, "generated_query": generated_query, "available_tools": list(criteria)},
        questions={"search_tool": {
            "type": "choice",
            "instructions": "Which single search tool is most likely to retrieve the evidence needed for this section?",
            "criteria": criteria,
        }},
        decision_id=f"query.search_tool.{phase}",
    )
    answer = _answer(result, "search_tool")
    choice = _choice_value(answer)
    confidence = _confidence(answer, choice)
    threshold = _float_env("SYSTEM_ONE_CHOICE_CONFIDENCE", 0.45)
    if not choice or choice not in criteria or confidence < threshold:
        return None
    routed: Dict[str, Any] = {"search_tool": choice, "decision_source": "system_one", "system_one_confidence": confidence}
    if choice == "search_news_by_date" and date_range:
        routed["start_date"], routed["end_date"] = date_range
    return routed


def choose_media_search_tool(*, input_data: Any, generated_query: str, phase: str, provider: str) -> Optional[Dict[str, Any]]:
    input_dict = input_data if isinstance(input_data, dict) else {"input": str(input_data)}
    criteria = dict(MEDIA_SEARCH_CRITERIA)
    if provider == "AnspireAPI":
        criteria.pop("web_search_only", None)
        criteria.pop("search_for_structured_data", None)
    result = get_system_one_client().evaluate(
        state={"phase": phase, "provider": provider, "section": input_dict, "generated_query": generated_query},
        questions={"search_tool": {
            "type": "choice",
            "instructions": "Which single available media-search tool best matches the evidence need?",
            "criteria": criteria,
        }},
        decision_id=f"media.search_tool.{phase}",
    )
    choice = _choice_value(_answer(result, "search_tool"))
    if not choice or choice not in criteria:
        return None
    return {"search_tool": choice, "decision_source": "system_one"}


def choose_insight_search_plan(*, input_data: Any, generated_query: str, phase: str) -> Optional[Dict[str, Any]]:
    input_dict = input_data if isinstance(input_data, dict) else {"input": str(input_data)}
    combined = " ".join(str(input_dict.get(k, "")) for k in ("title", "content", "paragraph_latest_state"))
    date_range = extract_explicit_date_range(combined)
    tool_criteria = dict(INSIGHT_SEARCH_CRITERIA)
    if not date_range:
        tool_criteria.pop("search_topic_by_date", None)
    result = get_system_one_client().evaluate(
        state={"phase": phase, "section": input_dict, "generated_query": generated_query, "explicit_date_range": date_range},
        questions={
            "search_tool": {"type": "choice", "instructions": "Which database tool best matches the evidence need?", "criteria": tool_criteria},
            "platform": {"type": "choice", "instructions": "Which single social platform is explicitly required? Choose none otherwise.", "criteria": PLATFORM_CRITERIA},
            "time_period": {"type": "choice", "instructions": "Which recency window best matches the section?", "criteria": TIME_PERIOD_CRITERIA},
            "enable_sentiment": {"type": "noul", "instructions": "Would sentiment analysis materially help answer this section?", "criteria": {"true": "Sentiment is analytically useful.", "false": "Sentiment adds little value."}},
        },
        decision_id=f"insight.search_plan.{phase}",
    )
    choice = _choice_value(_answer(result, "search_tool"))
    if not choice or choice not in tool_criteria:
        return None
    plan: Dict[str, Any] = {"search_tool": choice, "decision_source": "system_one"}
    if choice == "search_topic_by_date" and date_range:
        plan["start_date"], plan["end_date"] = date_range
    platform = _choice_value(_answer(result, "platform"))
    if choice == "search_topic_on_platform":
        if platform and platform != "none":
            plan["platform"] = platform
        else:
            plan["search_tool"] = "search_topic_globally"
    time_period = _choice_value(_answer(result, "time_period"))
    if choice == "search_hot_content" and time_period in TIME_PERIOD_CRITERIA:
        plan["time_period"] = time_period
    sentiment = _noul_value(_answer(result, "enable_sentiment"))
    if sentiment is not None:
        plan["enable_sentiment"] = sentiment >= _float_env("SYSTEM_ONE_SENTIMENT_THRESHOLD", 0.5)
    return plan


def should_continue_research(state: Dict[str, Any], *, decision_id: str = "query.reflection.continue") -> Optional[bool]:
    result = get_system_one_client().evaluate(
        state=state,
        questions={"continue_research": {
            "type": "noul",
            "instructions": "Would one more targeted search materially improve factual coverage of this section?",
            "criteria": {"true": "There is an important evidence gap.", "false": "Coverage is already sufficient."},
        }},
        decision_id=decision_id,
    )
    value = _noul_value(_answer(result, "continue_research"))
    if value is None:
        return None
    return value >= _float_env("SYSTEM_ONE_STOP_THRESHOLD", 0.30)


def triage_evidence(*, query: str, section: Dict[str, Any], results: Sequence[Dict[str, Any]], decision_id: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
    if not results:
        return []
    max_results = max_results or _int_env("SYSTEM_ONE_EVIDENCE_MAX_RESULTS", 7)
    candidate_limit = _int_env("SYSTEM_ONE_EVIDENCE_CANDIDATES", 10)
    candidates = [dict(item) for item in list(results)[:candidate_limit]]
    state_results = []
    questions: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(candidates):
        state_results.append({
            "id": idx,
            "title": str(item.get("title", ""))[:300],
            "content": str(item.get("content") or item.get("raw_content") or "")[:900],
            "url": str(item.get("url", ""))[:500],
            "published_date": item.get("published_date"),
            "platform": item.get("platform"),
        })
        questions[f"r{idx}_relevance"] = {"type": "score", "instructions": f"How relevant is results[{idx}] to the section/question?", "criteria": EVIDENCE_RELEVANCE_LEVELS}
        questions[f"r{idx}_evidence"] = {"type": "score", "instructions": f"How strong is the factual evidence value of results[{idx}]?", "criteria": EVIDENCE_VALUE_LEVELS}
        questions[f"r{idx}_novelty"] = {"type": "score", "instructions": f"How much materially new information does results[{idx}] add?", "criteria": EVIDENCE_NOVELTY_LEVELS}
    result = get_system_one_client().evaluate(
        state={"query": query, "section": section, "results": state_results},
        questions=questions,
        decision_id=decision_id,
    )
    if not result:
        return candidates[:max_results]
    ranked = []
    for idx, item in enumerate(candidates):
        relevance = _score_value(_answer(result, f"r{idx}_relevance"))
        evidence = _score_value(_answer(result, f"r{idx}_evidence"))
        novelty = _score_value(_answer(result, f"r{idx}_novelty"))
        if None in (relevance, evidence, novelty):
            logger.warning(
                "System One evidence triage returned an incomplete answer set; "
                "preserving original result order"
            )
            return candidates[:max_results]
        utility = 0.50 * relevance + 0.35 * evidence + 0.15 * novelty
        item["_system_one_evidence"] = {
            "relevance": relevance,
            "evidence_value": evidence,
            "novelty": novelty,
            "utility": round(utility, 4),
        }
        ranked.append((utility, idx, item))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in ranked[:max_results]]


def decide_forum_host(forum_logs: Sequence[str]) -> Optional[Dict[str, Any]]:
    speeches = [str(x) for x in forum_logs if str(x).strip()]
    if not speeches:
        return {"needed": False, "probability": 0.0, "reason": "no_speeches"}
    result = get_system_one_client().evaluate(
        state={"agent_speeches": speeches[-5:]},
        questions={
            "host_needed": {"type": "noul", "instructions": "Would a moderator intervention materially improve the discussion now?", "criteria": {"true": "Intervention would improve the next research step.", "false": "Agents are progressing normally."}},
            "host_reason": {"type": "choice", "instructions": "What is the strongest reason for a moderator turn, if any?", "criteria": {
                "disagreement": "Agents materially disagree.",
                "contradiction": "A factual/logical contradiction needs resolution.",
                "missing_evidence": "A key evidence gap should be called out.",
                "stalled": "Discussion is repetitive or stalled.",
                "new_direction": "A valuable new analytical direction is needed.",
                "none": "No intervention is needed.",
            }},
        },
        decision_id="forum.host_gate",
    )
    probability = _noul_value(_answer(result, "host_needed"))
    if probability is None:
        return None
    return {
        "needed": probability >= _float_env("SYSTEM_ONE_HOST_THRESHOLD", 0.55),
        "probability": probability,
        "reason": _choice_value(_answer(result, "host_reason")) or "unknown",
    }



def should_expand_insight_keywords(original_query: str, context: str = "") -> Optional[bool]:
    """Gate the expensive free-form keyword-expansion LLM.

    True means the query is too abstract/compound and benefits from social-language
    expansion. False means the query is already a concrete searchable entity/event
    and deterministic token extraction is sufficient. None preserves legacy behavior.
    """
    result = get_system_one_client().evaluate(
        state={
            "original_query": original_query,
            "context": context,
        },
        questions={
            "expand_keywords": {
                "type": "noul",
                "instructions": (
                    "Does this database query materially benefit from generating multiple "
                    "alternative social-media search keywords, rather than searching the "
                    "concrete query terms directly?"
                ),
                "criteria": {
                    "true": (
                        "The query is abstract, analytical, uses formal language, contains "
                        "multiple concepts, or likely needs colloquial aliases/synonyms."
                    ),
                    "false": (
                        "The query already names a concrete event, person, organization, "
                        "product, hashtag, or short phrase that can be searched directly."
                    ),
                },
            }
        },
        decision_id="insight.keyword_expansion_gate",
    )
    probability = _noul_value(_answer(result, "expand_keywords"))
    if probability is None:
        return None
    return probability >= _float_env("SYSTEM_ONE_KEYWORD_EXPANSION_THRESHOLD", 0.55)

def _report_excerpt(report: Any, limit: int = 1200) -> str:
    if isinstance(report, dict):
        content = report.get("content", str(report))
    elif hasattr(report, "content"):
        content = getattr(report, "content")
    else:
        content = str(report)
    return str(content)[:limit]


def choose_report_template(*, query: str, reports: Sequence[Any], forum_logs: str, available_templates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not available_templates:
        return None
    criteria: Dict[str, str] = {}
    option_to_template: Dict[str, Dict[str, Any]] = {}
    template_state: List[Dict[str, str]] = []
    for index, template in enumerate(available_templates):
        option = f"template_{index}"
        option_to_template[option] = template
        description = str(template.get("description", "通用报告模板"))
        criteria[option] = f"Use {template.get('name', option)} when this description best fits: {description}"
        template_state.append({"option": option, "name": str(template.get("name", option)), "description": description})
    result = get_system_one_client().evaluate(
        state={"query": query, "templates": template_state, "report_excerpts": [_report_excerpt(r) for r in reports[:4]], "forum_excerpt": (forum_logs or "")[:1200]},
        questions={"report_template": {"type": "choice", "instructions": "Which single report template best matches the request and evidence?", "criteria": criteria}},
        decision_id="report.template",
    )
    answer = _answer(result, "report_template")
    choice = _choice_value(answer)
    confidence = _confidence(answer, choice)
    if not choice or choice not in option_to_template or confidence < _float_env("SYSTEM_ONE_CHOICE_CONFIDENCE", 0.45):
        return None
    template = option_to_template[choice]
    return {"template_name": template["name"], "template_content": template["content"], "selection_reason": f"System One choice; confidence={confidence:.3f}", "decision_source": "system_one", "system_one_confidence": confidence}


def plan_word_budget(*, sections: Sequence[Dict[str, Any]], query: str, reports: Dict[str, str], forum_logs: str, total_words: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if not sections:
        return None
    total_words = total_words or _int_env("SYSTEM_ONE_REPORT_TOTAL_WORDS", 10000)
    state_sections = []
    questions: Dict[str, Dict[str, Any]] = {}
    for idx, section in enumerate(sections):
        state_sections.append({"chapterId": section.get("chapterId") or f"S{idx + 1}", "title": section.get("title", ""), "outline": section.get("outline", [])})
        questions[f"s{idx}_importance"] = {"type": "score", "instructions": f"How important is sections[{idx}] to fully answering the query?", "criteria": CHAPTER_IMPORTANCE_LEVELS}
        questions[f"s{idx}_evidence"] = {"type": "score", "instructions": f"How much useful evidence is available for sections[{idx}]?", "criteria": CHAPTER_EVIDENCE_LEVELS}
        questions[f"s{idx}_complexity"] = {"type": "score", "instructions": f"How analytically complex is sections[{idx}]?", "criteria": CHAPTER_COMPLEXITY_LEVELS}
    result = get_system_one_client().evaluate(
        state={"query": query, "sections": state_sections, "report_excerpts": {k: str(v)[:1800] for k, v in reports.items()}, "forum_excerpt": (forum_logs or "")[:1200]},
        questions=questions,
        decision_id="report.word_budget",
    )
    if not result:
        return None
    rows = []
    for idx, section in enumerate(state_sections):
        importance = _score_value(_answer(result, f"s{idx}_importance"))
        evidence = _score_value(_answer(result, f"s{idx}_evidence"))
        complexity = _score_value(_answer(result, f"s{idx}_complexity"))
        if None in (importance, evidence, complexity):
            return None
        weight = max(0.1, 0.50 * importance + 0.30 * evidence + 0.20 * complexity)
        rows.append((section, importance, evidence, complexity, weight))
    total_weight = sum(row[4] for row in rows) or 1.0
    chapter_count = len(rows)
    # Keep the historical 250-word floor when the total budget can support it.
    # For small budgets, lower the floor so allocation remains mathematically valid.
    base_words = min(250, total_words // chapter_count) if chapter_count else 0
    distributable = max(0, total_words - base_words * chapter_count)
    raw_extras = [distributable * row[4] / total_weight for row in rows]
    integer_extras = [int(value) for value in raw_extras]
    remainder = distributable - sum(integer_extras)
    remainder_order = sorted(
        range(chapter_count),
        key=lambda idx: raw_extras[idx] - integer_extras[idx],
        reverse=True,
    )
    for idx in remainder_order[:remainder]:
        integer_extras[idx] += 1
    targets = [
        base_words + integer_extras[idx]
        for idx in range(chapter_count)
    ]
    chapters = []
    for idx, (section, importance, evidence, complexity, _) in enumerate(rows):
        target = targets[idx]
        chapters.append({
            "chapterId": section["chapterId"],
            "title": section["title"],
            "targetWords": target,
            "minWords": max(200, int(round(target * 0.80))),
            "maxWords": max(300, int(round(target * 1.20))),
            "emphasis": list(section.get("outline") or [])[:4],
            "rationale": f"System One scores: importance={importance:.2f}, evidence={evidence:.2f}, complexity={complexity:.2f}",
        })
    return {
        "totalWords": total_words,
        "tolerance": 0.20,
        "globalGuidelines": [
            "Prioritize factual density and source-grounded claims.",
            "Allocate more space to chapters with higher importance/evidence/complexity scores.",
            "Avoid padding low-evidence chapters solely to meet length targets.",
        ],
        "chapters": chapters,
        "decision_source": "system_one",
    }



def apply_layout_controls(
    *,
    design: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    query: str,
) -> Dict[str, Any]:
    """Let Jev own bounded SWOT/PEST applicability flags.

    The report schema allows at most one SWOT chapter and at most one PEST
    chapter. We ask all chapter judgments in parallel, then enforce that global
    constraint deterministically by taking the highest Noul above threshold.
    """
    toc_plan = design.get("tocPlan")
    if not isinstance(toc_plan, list) or not toc_plan:
        return design

    state_sections = []
    questions: Dict[str, Dict[str, Any]] = {}
    for idx, section in enumerate(sections):
        state_sections.append(
            {
                "chapterId": section.get("chapterId") or f"S{idx + 1}",
                "title": section.get("title", ""),
                "outline": section.get("outline", []),
            }
        )
        questions[f"s{idx}_swot"] = {
            "type": "noul",
            "instructions": (
                f"Would a SWOT framework materially improve analysis in sections[{idx}] "
                "for the user's query?"
            ),
            "criteria": {
                "true": "SWOT is genuinely useful for this chapter's analytical goal.",
                "false": "SWOT would be forced, redundant, or decorative.",
            },
        }
        questions[f"s{idx}_pest"] = {
            "type": "noul",
            "instructions": (
                f"Would a PEST framework materially improve analysis in sections[{idx}] "
                "for the user's query?"
            ),
            "criteria": {
                "true": "Political/economic/social/technological decomposition is genuinely useful.",
                "false": "PEST would be forced, redundant, or decorative.",
            },
        }

    result = get_system_one_client().evaluate(
        state={"query": query, "sections": state_sections},
        questions=questions,
        decision_id="report.layout_controls",
    )
    if not result:
        return design

    threshold = _float_env("SYSTEM_ONE_LAYOUT_FRAMEWORK_THRESHOLD", 0.60)
    swot_scores = []
    pest_scores = []
    for idx in range(len(state_sections)):
        swot = _noul_value(_answer(result, f"s{idx}_swot"))
        pest = _noul_value(_answer(result, f"s{idx}_pest"))
        if swot is not None:
            swot_scores.append((swot, idx))
        if pest is not None:
            pest_scores.append((pest, idx))

    swot_idx = max(swot_scores, default=(0.0, -1))[1]
    swot_prob = max(swot_scores, default=(0.0, -1))[0]
    pest_idx = max(pest_scores, default=(0.0, -1))[1]
    pest_prob = max(pest_scores, default=(0.0, -1))[0]

    updated = copy.deepcopy(design)
    toc = updated.get("tocPlan") or []
    section_index = {
        str(section.get("chapterId") or f"S{idx + 1}"): idx
        for idx, section in enumerate(state_sections)
    }

    for entry_idx, entry in enumerate(toc):
        if not isinstance(entry, dict):
            continue
        chapter_id = str(entry.get("chapterId") or "")
        idx = section_index.get(chapter_id, entry_idx if entry_idx < len(state_sections) else -1)
        entry["allowSwot"] = bool(idx == swot_idx and swot_prob >= threshold)
        entry["allowPest"] = bool(idx == pest_idx and pest_prob >= threshold)

    updated["layoutDecisionSource"] = "system_one"
    return updated
