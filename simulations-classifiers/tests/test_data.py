import numpy as np
import pytest

from classifiers.data.data import (
    Data,
    MsaCompositionData,
    SequencesData,
    SiteCompositionData,
)
from classifiers.data.sources import DictSource, FastaSource
from classifiers.data.tokenizers import DNA_TOKENIZER
from classifiers.utils import LABEL_REAL, LABEL_SIMULATED


@pytest.fixture
def real_source():
    return DictSource({"seq1": ["ATGC", "ATG-"]})


@pytest.fixture
def real_source_with_ambigs():
    return DictSource({"seq1": ["ATGC", "ATXCATG"], "seq1b": ["ATG-", "*CGC"]})


@pytest.fixture
def sim_source():
    return DictSource({"seq2": ["AGCT", "-GCG"]})


@pytest.fixture
def wrong_source():
    return DictSource({"seq2": ["I", "-GCG"]})


@pytest.fixture
def real_source_width():
    return DictSource({"seq1": ["ATGC", "ATG-"], "seq1b": ["ATGCGC", "ATG-AT"]})


@pytest.fixture
def sim_source_height():
    return DictSource({"seq2": ["AGCT", "-GCG"], "seq2b": ["-A-T", "CGTA", "CGGG"]})


def test_data_from_fasta(tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    real_path = data_path / "class_0"
    sim_path = data_path / "class_1"
    real_path.mkdir()
    sim_path.mkdir()
    (real_path / "seq1.fasta").write_text(">seq11\nATGC\n>seq12\nATG-")
    (sim_path / "seq2.fasta").write_text(">seq21\nAGCT\n>seq22\n-GCG")

    real_source = FastaSource(real_path)
    sim_source = FastaSource(sim_path)
    data = Data(source_real=real_source, source_simulated=sim_source, tokenizer=DNA_TOKENIZER)
    assert data.n_tokens == DNA_TOKENIZER.n_tokens + 1
    assert np.array_equal(data.aligns["seq1"], np.array([[2, 3, 4, 5], [2, 3, 4, 1]]))
    assert np.array_equal(data.aligns["seq2"], np.array([[2, 4, 5, 3], [1, 4, 5, 4]]))
    assert data.labels == {"seq1": LABEL_REAL, "seq2": LABEL_SIMULATED}


def test_data_from_dict(real_source, sim_source):
    data = Data(real_source, sim_source, tokenizer=DNA_TOKENIZER)
    assert data.n_tokens == DNA_TOKENIZER.n_tokens + 1
    assert np.array_equal(data.aligns["seq1"], np.array([[2, 3, 4, 5], [2, 3, 4, 1]]))
    assert np.array_equal(data.aligns["seq2"], np.array([[2, 4, 5, 3], [1, 4, 5, 4]]))
    assert data.labels == {"seq1": LABEL_REAL, "seq2": LABEL_SIMULATED}


def test_data_with_wrong_char(wrong_source, sim_source):
    with pytest.raises(KeyError):
        Data(wrong_source, sim_source, tokenizer=DNA_TOKENIZER)


def test_data_with_duplicated_key(sim_source):
    data = Data(sim_source, sim_source, tokenizer=DNA_TOKENIZER)
    assert set(data.aligns.keys()) == {"seq2", "seq2_sim"}


def test_data_with_ambig(real_source_with_ambigs, sim_source):
    data = Data(real_source_with_ambigs, sim_source, tokenizer=DNA_TOKENIZER)
    assert np.array_equal(data.aligns["seq1"], np.array([[2, 3, 4, 5, 0, 0, 0]]))
    assert np.array_equal(data.aligns["seq1b"], np.array([[2, 3, 4, 1]]))
    assert np.array_equal(data.aligns["seq2"], np.array([[2, 4, 5, 3], [1, 4, 5, 4]]))
    assert data.labels == {"seq1": LABEL_REAL, "seq1b": LABEL_REAL, "seq2": LABEL_SIMULATED}


def test_msa_composition_data(real_source, sim_source):
    data = Data(real_source, sim_source, tokenizer=DNA_TOKENIZER)
    msa_data = MsaCompositionData(data)
    assert msa_data.labels == {"seq1": LABEL_REAL, "seq2": LABEL_SIMULATED}
    assert np.array_equal(msa_data.aligns["seq1"], np.array([0, 1, 2, 2, 2, 1]) / 8)
    assert np.array_equal(msa_data.aligns["seq2"], np.array([0, 1, 1, 1, 3, 2]) / 8)


def test_site_composition_data(real_source, sim_source):
    data = Data(real_source, sim_source, tokenizer=DNA_TOKENIZER)
    site_data = SiteCompositionData(data)
    assert site_data.labels == {"seq1": LABEL_REAL, "seq2": LABEL_SIMULATED}
    assert np.array_equal(
        site_data.aligns["seq1"],
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            ]
        ),
    )
    assert np.array_equal(
        site_data.aligns["seq2"],
        np.array(
            [
                [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            ]
        ),
    )


def test_sequences_data(real_source, sim_source):
    data = Data(real_source, sim_source, tokenizer=DNA_TOKENIZER)
    seq_data = SequencesData(data)
    assert np.array_equal(seq_data.aligns["seq1"], np.array([[2, 3, 4, 5], [2, 3, 4, 1]]))
    assert np.array_equal(seq_data.aligns["seq2"], np.array([[2, 4, 5, 3], [1, 4, 5, 4]]))
    assert seq_data.labels == {"seq1": LABEL_REAL, "seq2": LABEL_SIMULATED}


def test_filter_width_height(real_source_width, sim_source_height):
    data = Data(real_source_width, sim_source_height, tokenizer=DNA_TOKENIZER)
    data.filter_aligns_width(4)
    lim_data = SequencesData(data)
    assert set(lim_data.aligns.keys()) == {"seq1", "seq2", "seq2b"}
    assert set(lim_data.labels.keys()) == {"seq1", "seq2", "seq2b"}

    data = Data(real_source_width, sim_source_height, tokenizer=DNA_TOKENIZER)
    data.filter_aligns_height(2)
    lim_data = SequencesData(data)
    assert set(lim_data.aligns.keys()) == {"seq1", "seq2", "seq1b"}
    assert set(lim_data.labels.keys()) == {"seq1", "seq2", "seq1b"}
