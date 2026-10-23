"""Pemenang per jawaban dihitung sekali, termasuk hasil seri numerik."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_mlp_context_results import winner_counts


@pytest.mark.parametrize('predictions,expected',[
    ((82,81,83),{'S1':0,'S2':60,'S3':0,'Seri':0}),
    ((81,81,82),{'S1':0,'S2':0,'S3':0,'Seri':60}),
    ((81,81.000005,82),{'S1':0,'S2':0,'S3':0,'Seri':60}),
])
def test_one_count_per_answer_including_numerical_ties(predictions,expected):
    rows=[{'Nilai Guru':80,**{'Prediksi '+s:p for s,p in zip(('S1','S2','S3'),predictions)}} for _ in range(60)]
    counts=winner_counts(rows)
    assert counts==expected and sum(counts.values())==60


def test_partial_test_set_cannot_be_reported_as_sixty_answers():
    with pytest.raises(ValueError,match='60 jawaban'):
        winner_counts([{'Nilai Guru':80,'Prediksi S1':80,'Prediksi S2':81,'Prediksi S3':82}]*59)
