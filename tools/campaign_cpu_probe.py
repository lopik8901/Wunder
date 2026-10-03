"""Production-isolated callback timing on fixed search inputs, no scoring."""
import argparse
import json
import shutil
import time
from pathlib import Path

from competition_engineering.isolated_campaign import DEPLOY, callback_source
from competition_engineering.generated_runner import GeneratedSandbox, WORKER, replay
from competition_engineering.manual_search_core import ROOT, SEARCH
from competition_engineering.pipeline import cached, write_json
from competition_engineering.autonomous_campaign import CAMPAIGN, event


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    out = CAMPAIGN / args.tag
    out.mkdir(exist_ok=False)
    source_work = ROOT / 'competition_engineering/mlevolve_runs/autonomous_20261001/e01b_additive_isolated'
    _, z = next(cached(SEARCH))
    inputs = {key: z[key] for key in ['seq', 'x', 'need']}
    results = {}
    for mode in ['baseline', 'linear']:
        work = out / mode
        shutil.copytree(source_work / 'deploy', work / 'deploy')
        source = callback_source('linear', False)
        if mode == 'baseline':
            source = source[:source.index('\nclass PredictionModel:')] + '\nclass PredictionModel(Combo):\n    pass\n'
        (work / 'deploy/solution.py').write_text(source)
        sandbox = GeneratedSandbox()
        sandbox.preflight(WORKER, work)
        timings = []
        for repeat in range(3):
            _, timing = replay(sandbox, work, [inputs], time.monotonic()+120)
            timings.append(timing['callback_seconds'] / timing['rows'] * 1e6)
        results[mode] = timings
        print(json.dumps({mode: timings}), flush=True)
    write_json(out / 'report.json', results)
    event('cpu_integration_probe', id=args.tag, timings=results)


if __name__ == '__main__':
    main()
