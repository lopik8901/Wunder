"""Search evidence and callback engineering checks, never a protected gate."""
import argparse
import json
import shutil
import time
from competition_engineering import objective_alignment as oa, campaign_onnx
import numpy as np
from competition_engineering import autonomous_campaign as lab
from competition_engineering.manual_search_core import ROOT, TRAIN_1024, load_combo, predict_combo, assess
from competition_engineering.pipeline import cached, write_json
from competition_engineering.generated_runner import GeneratedSandbox, WORKER, _check_artifacts, _package, validate_callback, replay, digest


def main(tier, target=0, group=False, full=False):
    from competition_engineering.full_moment_objective import output_directory as full_directory
    directory=(ROOT/('competition_engineering/runs/group_objective_t'+str(target)+'_20261002') if group else
               full_directory(tier) if full else oa.output_directory(tier,target))
    oa.OUT=directory
    oa.verify_fit_identity(directory/'isolated_training')
    q=json.loads((directory/'diagnosis.json').read_text())
    deadline=time.monotonic()+max(1,json.loads((directory/'campaign.json').read_text())['started_unix']+7200-time.time())
    combo=load_combo(); all_results={}
    with np.load(directory/'isolated_training/artifacts/models.npz') as archive:
        models={k:archive[k].copy() for k in archive.files}
    selections={k:v for k,v in q['selections'].items() if k.endswith('_select_wp')}
    # Include any grid candidate above the practical threshold, with its adaptive
    # selection disclosed. Tiny positive diagnostic fluctuations are not winners.
    for name,score in q['search'].items():
        if '@' in name and score['wp']-q['search']['incumbent']['wp'] >= .0002:
            model,strength=name.split('@')
            selections[name]={'model':model,'strength':float(strength),'selection':'adaptive_search_diagnostic_grid'}
    for key,selected in selections.items():
        # A separate repair directory preserves the original timing failure.
        path=directory/('current_checks_'+key)
        if (path/'report.json').exists():
            all_results[key]=json.loads((path/'report.json').read_text()); continue
        coef=models[selected['model']]; strength=selected['strength']
        assert np.count_nonzero(coef[115:])==0, 'Temporal branch must stay closed'
        def predictor(z,base):
            f=np.column_stack((np.ones(len(base)),np.clip(base,-2,2),
                               np.clip((z['x']-combo['mean'])/combo['scale'],-8,8))).astype(np.float32)
            p=base.copy(); p[:,target]+=(strength*(f@coef[:115])).astype(np.float32)
            return p
        result=assess(predictor,diagnostics=False)
        ci=lab.paired99(predictor)
        path.mkdir(); (path/'artifacts').mkdir()
        np.savez(path/'artifacts/model.npz',coef=coef,mean=combo['mean'],scale=combo['scale'])
        campaign_onnx.export(path/'artifacts/model.npz',path/'artifacts/fused.onnx','current',strength=strength,target=target)
        (path/'callback.py').write_text(campaign_onnx.callback_source('current'),encoding='utf-8')
        original={'callback.py':digest(path/'callback.py')}
        artifacts=_check_artifacts(path,original); archive,deploy_hashes=_package(path)
        sandbox=GeneratedSandbox(); sandbox.preflight(WORKER,path)
        validation=validate_callback(sandbox,path,TRAIN_1024,deadline)
        _,z=next(cached(TRAIN_1024)); predictions,_=replay(sandbox,path,[z],deadline)
        expected=predictor(z,predict_combo(z,combo))
        np.testing.assert_allclose(predictions[0][z['need']],expected[z['need']],atol=3e-5,rtol=3e-5)
        validation.update(stream_batch_parity=True,label_mask_independence=True,finite_outputs=True,
                          max_abs_parity=float(np.max(np.abs(predictions[0][z['need']]-expected[z['need']]))))
        eligible=(result['delta_combined']>=.0002 and ci[0]>0 and min(result['delta_per_target'])>=-.0002
                  and validation['callback_us_per_row']<=76.92753780833335)
        report={'selection':selected,'search':result,'paired_99ci':ci,'callback_checks':validation,
                'status':'AWAITING PROMOTION' if eligible else 'scientific_negative_or_insufficient_search_evidence',
                'identity':{'source':original,'artifacts':artifacts,'deployment':deploy_hashes,'package':digest(archive)},
                'scope':'Training-row callback engineering checks and fixed search scores only; no protected evaluation'}
        write_json(path/'report.json',report); all_results[key]=report
        if eligible:
            preserved=directory/('awaiting_promotion_'+key)
            shutil.copytree(path/'deploy',preserved)
            write_json(preserved/'status.json',{'status':'AWAITING PROMOTION','evidence':str(path.relative_to(ROOT)/'report.json'),
                       'reference_sha256':oa.COMBO_SHA})
        oa.event('candidate_checked',id=key,search=result['candidate'],delta=result['delta_combined'],paired_99ci=ci,
                 callback_checks=validation,status=report['status'],report_sha256=digest(path/'report.json'))
        print(json.dumps({'id':key,'delta':result['delta_combined'],'ci99':ci,'callback_us':validation['callback_us_per_row'],'status':report['status']}),flush=True)
    write_json(directory/'candidate_checks.json',all_results)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--tier',type=int,choices=[1024,4096],default=1024)
    p.add_argument('--target',type=int,choices=[0,1],default=0)
    p.add_argument('--group',action='store_true')
    p.add_argument('--full',action='store_true')
    args=p.parse_args(); main(args.tier,args.target,args.group,args.full)
