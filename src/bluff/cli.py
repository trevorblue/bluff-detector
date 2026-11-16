"""Command line: `bluff <stage> --config configs/variants/<variant>.yaml`.

Stages run in order with `bluff all`. Each one can also be rerun alone.

Entry point of the pipeline: pyproject.toml maps the `bluff` command to main() here, and
each stage lives in bluff/pipeline/<stage>.py. CRISP-DM: runs the phases in order, from
data understanding (validate) to deployment (register).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

from bluff.config import load_config

STAGES = ["validate", "ingest", "audit", "generate", "label", "featurize", "train",
          "evaluate", "register", "export"]


def _stage(name: str) -> Callable[[dict[str, Any]], Any]:
    from bluff import pipeline  # noqa: F401  (keeps imports lazy per stage)

    module = __import__(f"bluff.pipeline.{name}", fromlist=["run"])
    return module.run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bluff", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in [*STAGES, "all", "semantic_entropy", "finetune_qlora"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
    cp = sub.add_parser("compare", help="paired comparison of two variants (H3)")
    cp.add_argument("--config", required=True, help="variant A, e.g. the base model")
    cp.add_argument("--other", required=True, help="variant B, e.g. the fine-tuned model")
    ip = sub.add_parser("import-sheet", help="students' .xlsx -> data/kenya/questions.jsonl")
    ip.add_argument("--config", required=True)
    ip.add_argument("--sheet", required=True, help="path to the downloaded Kenya Question Sheet")
    mp = sub.add_parser("monitor", help="drift report: live requests vs the training data")
    mp.add_argument("--config", required=True)
    mp.add_argument("--log", default="logs/requests.jsonl", help="the API's request log")
    pp = sub.add_parser("promote", help="move a registry alias (also used for rollback)")
    pp.add_argument("--config", required=True)
    pp.add_argument("--version", type=int, required=True)
    pp.add_argument("--alias", default="champion")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)

    if args.cmd == "all":
        results = {}
        for name in STAGES:
            logging.getLogger("bluff").info("== stage: %s ==", name)
            results[name] = _short(_stage(name)(cfg))
        print(json.dumps(results, indent=2, default=str))
        return 0
    if args.cmd == "compare":
        from bluff.pipeline import compare

        result = compare.run(cfg, load_config(args.other))
    elif args.cmd == "import-sheet":
        from bluff import sheet_import

        result = sheet_import.run(cfg, args.sheet)
    elif args.cmd == "monitor":
        from bluff import monitoring

        result = monitoring.run(cfg, args.log)
    elif args.cmd == "promote":
        from bluff.pipeline import register

        register.promote(cfg, args.version, args.alias)
        result = {"promoted": args.version, "alias": args.alias}
    else:
        result = _stage(args.cmd)(cfg)
    print(json.dumps(_short(result), indent=2, default=str))
    return 0


def _short(result: Any) -> Any:
    """Keep console output readable: drop bulky nested blocks."""
    if isinstance(result, dict) and "detectors" in result and "abstention" in result:
        return {k: result[k] for k in ("variant", "served_detector", "tau_flag", "t_abstain")}
    return result


if __name__ == "__main__":
    sys.exit(main())
