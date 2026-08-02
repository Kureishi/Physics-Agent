"""
Bulk Problem Set Generator.

Not a pipeline stage -- a convenience tool for generating many new
practice problems to expand what problem_set_cli.py runs against. Reuses
Stage 8's ProblemGenerator directly, but doesn't require any accumulated
weak-area data first: each domain tag gets a synthetic "general practice"
signal instead of a real one from meta_learning.curriculum_signals.
weak_areas, so this works even against a completely fresh, empty memory/.

This is deliberately a separate tool from curriculum_cli.py: that one
targets a *specific, measured* weakness and solves what it generates
immediately (with a before/after measurement). This one just wants volume
-- broad, well-posed coverage across domains -- and doesn't solve anything
itself; feed its output into problem_set_cli.py separately.

Usage:
    python -m physics_agent.generate_problem_set_cli --n-per-domain 5 --out data/problem_sets/expanded_set.json
    python -m physics_agent.generate_problem_set_cli --domains energy dynamics --n-per-domain 10 --dry-run
    python -m physics_agent.generate_problem_set_cli --n-per-domain 3 --append data/problem_sets/intro_physics_set.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .curriculum.problem_generator import ProblemGenerator
from .llm_client import LLMClient, MockLLMClient
from .planner import DOMAIN_TAXONOMY
from .tools.literature import LiteratureSearchTool

# Higher than cli.py's solving temperature on purpose: diversity matters
# more here than for actually solving a problem, and a low-temperature
# local model asked the same "generate a problem" prompt repeatedly tends
# to converge on the same handful of scenarios (block-on-incline, etc.).
DEFAULT_GENERATION_TEMPERATURE = 0.9


def _build_generator(dry_run: bool, config: Config) -> ProblemGenerator:
    if dry_run:
        llm = MockLLMClient()
    else:
        llm = LLMClient(
            base_url=config.lm_studio_base_url,
            api_key=config.lm_studio_api_key,
            model=config.lm_studio_model,
            temperature=DEFAULT_GENERATION_TEMPERATURE,
            timeout=config.lm_studio_timeout_seconds,
            max_tokens=config.lm_studio_max_tokens,
        )
    literature_tool = None if dry_run else LiteratureSearchTool()
    return ProblemGenerator(llm, literature_tool=literature_tool)


def generate_for_domains(
    domains: List[str],
    n_per_domain: int,
    dry_run: bool = False,
    config: Optional[Config] = None,
    id_prefix: str = "gen",
) -> List[Dict[str, Any]]:
    config = config or Config()
    generator = _build_generator(dry_run, config)

    problems: List[Dict[str, Any]] = []
    for domain in domains:
        generated_texts: List[str] = []
        signal = {
            "source": "manual_expansion",
            "domain_tags": [domain],
            "reason": f"general practice problem for the '{domain}' domain",
            "weight": 0,
        }
        for i in range(n_per_domain):
            try:
                result = generator.generate(signal, avoid=generated_texts)
            except Exception as e:
                # generator.generate() already converts both bad-JSON and
                # raw LLM-call failures into a plain ValueError internally,
                # so this should always be a ValueError in practice -- this
                # broader catch is defense-in-depth, consistent with the
                # rest of this codebase's stance that one bad sub-step
                # (here: one signal's generation) should never crash an
                # entire batch run, whatever specific exception it happens
                # to raise.
                print(f"  [{domain} #{i + 1}] generation failed, skipping: {e}")
                continue
            problems.append(
                {
                    "id": f"{id_prefix}-{domain}-{i + 1:02d}",
                    "domain_hint": domain,
                    "problem_text": result["problem_text"],
                }
            )
            generated_texts.append(result["problem_text"])
            print(f"  [{domain} #{i + 1}/{n_per_domain}] {result['problem_text'][:70]}...")

    return problems


def load_existing(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate new physics problems to expand a problem set")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help=f"Domain tags to generate for (default: all {len(DOMAIN_TAXONOMY)} known tags)",
    )
    parser.add_argument("--n-per-domain", type=int, default=3, help="How many problems to generate per domain")
    parser.add_argument("--out", default="data/problem_sets/expanded_set.json", help="Output path (overwritten)")
    parser.add_argument(
        "--append",
        default=None,
        help="Instead of --out, append newly generated problems to this existing problem-set JSON file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use a mock LLM instead of calling LM Studio")
    args = parser.parse_args()

    domains = args.domains or DOMAIN_TAXONOMY
    unknown = set(domains) - set(DOMAIN_TAXONOMY)
    if unknown:
        raise SystemExit(f"Unknown domain tag(s): {sorted(unknown)}. Valid tags: {DOMAIN_TAXONOMY}")

    print(f"Generating {args.n_per_domain} problem(s) each for {len(domains)} domain(s)...\n")
    new_problems = generate_for_domains(domains, args.n_per_domain, dry_run=args.dry_run)

    if args.append:
        existing = load_existing(args.append)
        existing_ids = {p["id"] for p in existing}
        # Guard against id collisions if this is run more than once against
        # the same file -- re-prefix on collision rather than silently
        # overwriting an existing problem under the same id.
        for p in new_problems:
            base_id = p["id"]
            suffix = 1
            while p["id"] in existing_ids:
                p["id"] = f"{base_id}-{suffix}"
                suffix += 1
            existing_ids.add(p["id"])
        combined = existing + new_problems
        out_path = args.append
    else:
        combined = new_problems
        out_path = args.out

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nWrote {len(new_problems)} new problem(s); {len(combined)} total now in {out_path}")


if __name__ == "__main__":
    main()
