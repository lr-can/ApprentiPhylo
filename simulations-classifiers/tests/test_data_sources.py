import pytest

from classifiers.data.sources import DictSource, FastaSource


@pytest.fixture
def real_align():
    return {"seq1": ["ATGC", "ATG-"]}


@pytest.fixture
def sim_align():
    return {"seq2": ["AGCT", "-GCG"], "seq3": ["G", "AA---"]}


@pytest.fixture
def sample_labels():
    return {"seq1": 0, "seq2": 1}


def test_fasta_source(tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    real_path = data_path / "class_0"
    sim_path = data_path / "class_1"
    real_path.mkdir()
    sim_path.mkdir()
    (real_path / "seq1.fasta").write_text(">seq11\nATGC\n>seq12\nATG-\n>seq13\nC\n---")
    (sim_path / "seq2.fasta").write_text(">seq21\nAG\nCT\n>seq22\n-GC\nG")
    (sim_path / "seq3.fa").write_text(">seq31\nCTGA\n>seq32\nAA\n---")

    real_aligns = FastaSource(real_path).aligns
    sim_aligns = FastaSource(sim_path).aligns
    assert list(real_aligns.keys()) == ["seq1"]
    assert set(sim_aligns.keys()) == {"seq2", "seq3"}
    assert list(real_aligns.values()) == [["ATGC", "ATG-", "C---"]]
    assert list(sim_aligns.values()) == [["AGCT", "-GCG"], ["CTGA", "AA---"]]


def test_dict_source(real_align, sim_align):
    real_aligns = DictSource(real_align).aligns
    sim_aligns = DictSource(sim_align).aligns
    assert list(real_aligns.keys()) == ["seq1"]
    assert set(sim_aligns.keys()) == {"seq2", "seq3"}
    assert list(real_aligns.values()) == [["ATGC", "ATG-"]]
    assert list(sim_aligns.values()) == [["AGCT", "-GCG"], ["G", "AA---"]]


def test_fasta_invalid_path():
    wrong_path = "/foo/bar"
    with pytest.raises(FileNotFoundError):
        FastaSource(wrong_path)
