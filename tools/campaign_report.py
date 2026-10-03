"""Recoverable status/report from an explicit search-only evidence allowlist."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from competition_engineering.autonomous_campaign import CAMPAIGN, ROOT, sha
from competition_engineering.pipeline import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--final', action='store_true')
    p.add_argument('--reason', default='Campaign running; resume incomplete isolated replays with --resume.')
    args=p.parse_args()
    config=json.loads((CAMPAIGN/'campaign.json').read_text())
    prior_path=ROOT/'competition_engineering/reports/manual_research_search_only_20261001.json'
    prior=json.loads(prior_path.read_text())
    events=[json.loads(line) for line in (CAMPAIGN/'ledger.jsonl').read_text().splitlines()]
    results=[]
    for path in sorted(CAMPAIGN.glob('e*/report.json')):
        record=json.loads(path.read_text())
        if 'search_result' in record:
            results.append({'id':path.parent.name,'report_path':str(path.relative_to(ROOT)),
                            'report_sha256':sha(path),**record})
    scientific=[r for r in results if 'isolation' in r['config']]
    best=max(scientific,key=lambda r:r['search_result']['candidate']['weighted_pearson']) if scientific else None
    statuses={}
    for e in events:
        if 'id' in e:
            statuses[e['id']]=e['kind']
    awaiting=[r['id'] for r in scientific if r['status']=='AWAITING PROMOTION']
    hashes={str(q.relative_to(ROOT)):sha(q) for q in [CAMPAIGN/'campaign.json',CAMPAIGN/'library_manifest.json',CAMPAIGN/'library_manifest_v3.json',CAMPAIGN/'input_hashes.json',CAMPAIGN/'ledger.jsonl'] if q.exists()}
    artifact={'utc':datetime.now(timezone.utc).isoformat(),'status':'ended' if args.final else 'running',
              'stop_reason':args.reason,'starting_incumbent':config['starting_incumbent'],
              'prior_search_trajectory':prior['experiments'],'campaign_experiments':results,
              'event_trajectory':events,'experiment_statuses':statuses,'awaiting':awaiting,
              'best_isolated_search_candidate':best['id'] if best else None,
              'reproduction_hashes':hashes,
              'evidence':'reused_search_evidence_not_independent_validation',
              'evaluation_boundary':'Only training and designated search caches used. No protected evaluation or external result accessed.'}
    stem='FINAL_REPORT' if args.final else 'CHECKPOINT'
    write_json(CAMPAIGN/(stem+'.json'),artifact)
    lines=[f'# Autonomous Connectome research {"report" if args.final else "checkpoint"}', '',
           f'Updated: {artifact["utc"]}', '', f'Status: {args.reason}', '',
           f'Starting frozen search benchmark: **{config["starting_incumbent"]:.12f}**.',
           'All intervals are descriptive paired sequence bootstraps on reused search evidence; adaptive selection is not corrected.',
           'Training and inference use the existing production bubblewrap sandbox. No protected evaluation or submission was performed.', '',
           '## Current campaign trajectory', '',
           '| Experiment | Combined WP | Delta | Paired 95% interval | CPU µs/row | Status |',
           '|---|---:|---:|---|---:|---|']
    for r in results:
        s=r['search_result']; v=r['validation']; ci=s['paired_95ci_combined']
        cpu=v.get('callback_us_per_row',v.get('incremental_cpu_us_per_row'))
        suffix='full isolated callback' if 'callback_us_per_row' in v else 'residual only; engineering rehearsal'
        lines.append(f'| {r["id"]} | {s["candidate"]["weighted_pearson"]:.12f} | {s["delta_combined"]:+.9f} | [{ci[0]:+.9f}, {ci[1]:+.9f}] | {cpu:.2f} | {r["status"]}; {suffix} |')
    lines += ['', '## Interrupted attempts and repairs', '']
    for e in events:
        if e['kind'] in ['experiment_interrupted','engineering_repair','cpu_integration_probe']:
            lines.append('- '+json.dumps(e,ensure_ascii=False))
    lines += ['', '## Best distinct isolated search candidate', '']
    if best:
        s=best['search_result']
        lines += [f'**{best["id"]}**: {s["candidate"]["weighted_pearson"]:.12f}, delta {s["delta_combined"]:+.9f}.',
                  f'Paired 99% interval: {best["paired_99ci_combined"]}.',
                  'The incumbent remains unchanged unless the fixed rule is met.']
    else:
        lines.append('No isolated experiment has completed yet.')
    lines += ['', '## Prior search trajectory', '', '| Mechanism | Search WP | Interpretation |', '|---|---:|---|']
    for e in prior['experiments']:
        lines.append(f'| {e["name"]} | {e["search_wp"]:.12f} | {e.get("conclusion", "See original search-only report.")} |')
    lines += ['', '## Literature', '',
              'Library v2 preserves all 23 v1 cards and adds `nonlinear_vector_autoregression`.',
              '[Gauthier et al., Next generation reservoir computing (2021)](https://www.nature.com/articles/s41467-021-25801-2) motivates polynomial functions of causal history with a regularized readout. The featurewise EMA adaptation is curator inference, not a paper reproduction or evidence of task improvement.',
              'Library v3 preserves v2 and adds `functional_gradient_fitting`: [Friedman (2001)](https://doi.org/10.1214/aos/1013203451). It motivates objective-aligned function fitting; the proposed clipped-correlation derivative is curator inference and has not been implemented or tested in this campaign.',
              '', '## What was learned and next action', '',
              'Neither additive multiscale memory nor featurewise current-by-history products improved the frozen combo. The nonlinear candidate also worsened late t0; this weakens the tested temporal mechanism rather than supporting more timescale tuning. Other nonlinear temporal architectures remain untested.',
              'The best practical search incumbent remains 0.6588353223316641. The earlier tiny-tree point score 0.6588825374213778 remains uncertain (paired 95% delta interval [-0.0002228343, 0.0002908522]) and its measured residual-only CPU cost was 174.47 microseconds/row.',
              'Recommended next experiment: a train-only, objective-aligned correction, compared with the matched additive control. Training t0 correlation-tangent slope was 0.8453; clipped prediction weight mass was 0.838% in sampled training versus 4.161% in search. This is a diagnostic lead, not evidence that the proposed correction works. Causal input normalization and learned cross-feature temporal interactions remain alternative hypotheses.',
              'Engineering repairs: fused ONNX export and alternating preallocated buffers brought full callbacks below the unchanged 90-microsecond gate; worker EOF now preserves error diagnostics; numerical supervisor threads are bounded and sandbox jobs serialized. The final serialized Linux preflight had 32 passing tests.',
              'Only two distinct scientific experiments completed. Control repetitions and failed callback representations were engineering verification, not extra scientific trials. No experiments were running during the later interaction gap; elapsed campaign-window time is not active compute time.',
              '', '## Frozen candidates awaiting user review', '', ', '.join(awaiting) if awaiting else 'None from this campaign.',
              '', '## Reproduction', '',
              'Each experiment has a pre-scoring config, source snapshots, retrieval query/card hashes, immutable isolated fit, model hashes, callback validation, and checkpointed search predictions. The JSON report contains the complete trajectory and exact deployment-file hashes.', '',
              '| Artifact | SHA256 |','|---|---|']
    lines += [f'| {name} | `{value}` |' for name,value in hashes.items()]
    lines += ['', 'Use `python -m competition_engineering.isolated_campaign --id <incomplete-id> --mode <mode> --resume` to resume an interrupted scoring phase without refitting. Completed experiments reject reruns in their original directories.', '',
              'The source, input-cache and library manifests identify the exact reproducibility inputs. Earlier failed workspaces are retained.']
    (CAMPAIGN/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(CAMPAIGN/(stem+'.md')),'completed':len(scientific),'best':artifact['best_isolated_search_candidate']}))


if __name__=='__main__':
    main()
