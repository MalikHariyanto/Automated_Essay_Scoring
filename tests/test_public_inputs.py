"""Bukti clone publik tidak membutuhkan input penelitian untuk unit test."""
import json
from pathlib import Path
import pytest
from aes.cli import main


def test_missing_private_inputs_have_actionable_error(tmp_path,capsys):
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/default.json').read_text())
    config.update(csv=str(tmp_path/'private.csv'),rubric=str(tmp_path/'private.json'))
    path=tmp_path/'configs'/'default.json';path.parent.mkdir();path.write_text(json.dumps(config))
    with pytest.raises(SystemExit) as error:main(['--config',str(path),'inspect'])
    assert error.value.code==2
    message=capsys.readouterr().err
    assert 'Data tidak dibagikan di repo publik' in message
    assert '--csv/--rubric' in message


def test_prediction_uses_checkpoint_rubric_when_local_input_absent(tmp_path,monkeypatch,capsys):
    import aes.cli as cli
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/default.json').read_text())
    config.update(csv=str(tmp_path/'absent.csv'),rubric=str(tmp_path/'absent.json'))
    path=tmp_path/'configs'/'default.json';path.parent.mkdir();path.write_text(json.dumps(config))
    # Tiruan routing saja. Test ini tidak memuat model atau memvalidasi skor prediksi.
    calls=[]
    def fake_predict(checkpoint,texts,expected_rubric):
        calls.append((checkpoint,texts,expected_rubric));return []
    monkeypatch.setattr(cli,'predict',fake_predict)
    assert main(['--config',str(path),'predict','--checkpoint','technical.pt','--text','fixture teknis'])==0
    assert calls==[('technical.pt',['fixture teknis'],None)]
