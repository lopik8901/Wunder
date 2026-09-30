import json
import numpy as np
import pytest

from competition_engineering.run_experiment import validate_inputs


@pytest.fixture
def caches(tmp_path):
    config={}
    for name,start,count in [("train",0,256),("tune",300,2)]:
        directory=tmp_path/name; directory.mkdir()
        for group in range(count): np.savez(directory/f"{group:05d}.npz",seq=start+group)
        (directory/"identity.json").write_text(json.dumps({"data":{"path":"same-source"},
                                                         "split":name,"groups":list(range(count))}))
        config[name]=str(directory)
    return config


def test_runner_rejects_old_eight_sequence_training(caches):
    from pathlib import Path
    assert len(validate_inputs(caches)["train"]["groups"])==256
    path=Path(caches["train"])/"identity.json"
    identity=json.loads(path.read_text()); identity["groups"]=identity["groups"][:8]
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match="at least 256"): validate_inputs(caches)


def test_runner_rejects_actual_sequence_leakage_and_final_gate(caches):
    from pathlib import Path
    np.savez(Path(caches["tune"])/"00000.npz",seq=0)
    with pytest.raises(ValueError,match="leakage"): validate_inputs(caches)
    path=Path(caches["tune"])/"identity.json"
    identity=json.loads(path.read_text()); identity["split"]="final_gate"
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match="final-gate"): validate_inputs(caches)


def test_runner_rejects_protected_holdout_as_tune(caches):
    from pathlib import Path
    path=Path(caches["tune"])/"identity.json"
    identity=json.loads(path.read_text()); identity["split"]="train_holdout"
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match="designated tuning split"):
        validate_inputs(caches)


def test_timeout_record_is_emitted_and_mlevolve_preserves_its_reason(capsys):
    from types import SimpleNamespace
    from competition_engineering.run_experiment import emit_outcome
    from connectome.mlevolve_adapter import adapt_execution_result, FEEDBACK_PREFIX
    emit_outcome({"status":"timeout","error":"600-second training budget exhausted"})
    output=capsys.readouterr().out
    execution=SimpleNamespace(term_out=[output],exc_type="TimeoutExpired",exc_info=None)
    adapt_execution_result(execution)
    feedback=json.loads(execution.term_out[-1][len(FEEDBACK_PREFIX):])
    assert feedback["is_bug"] and feedback["metric"] is None
    assert "600-second training budget exhausted" in feedback["summary"]
