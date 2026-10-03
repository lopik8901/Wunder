"""Aggregate training/search saturation diagnostics; no new model or fit."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from competition_engineering.autonomous_campaign import CAMPAIGN, event
from competition_engineering.manual_search_core import TRAIN_1024, SEARCH, load_combo, predict_combo
from competition_engineering.pipeline import cached, write_json


def main():
    out=CAMPAIGN/'objective_diagnostic.json'
    if out.exists():
        raise ValueError('Diagnostic is already frozen')
    combo=load_combo()
    result={}
    with threadpool_limits(limits=1):
        for name,directory in [('training',TRAIN_1024),('search',SEARCH)]:
            sums=np.zeros((7,2),np.float64)
            n=0
            saturation=np.zeros(2,np.float64)
            saturation_weight=np.zeros(2,np.float64)
            for _,z in cached(directory):
                indices=np.flatnonzero(z['need'])[::40] if name=='training' else np.flatnonzero(z['mask'])
                base=predict_combo(z,combo)[indices].astype(np.float64)
                y=np.clip(z['y'][indices],-2,2).astype(np.float64)
                prediction=np.clip(base,-2,2)
                w=np.abs(y)
                sums+=np.stack([w.sum(0),(w*y).sum(0),(w*prediction).sum(0),
                                (w*y*y).sum(0),(w*prediction*prediction).sum(0),
                                (w*y*prediction).sum(0),(w*base).sum(0)])
                saturated=np.abs(base)>=2
                saturation+=saturated.sum(0)
                saturation_weight+=(w*saturated).sum(0)
                n+=len(indices)
            mass,sy,sp,syy,spp,syp,sraw=sums
            mean_y=sy/mass;mean_p=sp/mass
            covariance=syp-sy*sp/mass
            variance=spp-sp*sp/mass
            result[name]={'rows':n,'saturated_prediction_fraction':(saturation/n).tolist(),
                          'saturated_weight_fraction':(saturation_weight/mass).tolist(),
                          'weighted_target_mean':mean_y.tolist(),'weighted_clipped_prediction_mean':mean_p.tolist(),
                          'weighted_raw_prediction_mean':(sraw/mass).tolist(),
                          'correlation_tangent_slope':(covariance/variance).tolist()}
    result['interpretation']='Descriptive objective geometry only: weighted raw residual regression differs from the centered, variance-adjusted, unsaturated direction of clipped weighted correlation. A new objective requires a train-only fit and isolated search test.'
    write_json(out,result)
    event('objective_geometry_diagnostic',**result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
