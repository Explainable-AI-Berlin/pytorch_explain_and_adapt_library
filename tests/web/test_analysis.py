from peal.web.analysis import direction_atoms, group_accuracies


def test_direction_atoms():
    assert direction_atoms("squirting (SAE #5717)") == [5717]
    assert direction_atoms("OFF a (SAE #3303)  ->  ON b (SAE #853)") == [3303, 853]
    assert direction_atoms("Class") == []


def test_group_accuracies():
    labels = [0, 0, 0, 1, 1, 1]
    correct = [1, 0, 1, 1, 1, 0]
    present = [True, True, False, True, False, False]
    out = group_accuracies(labels, correct, present)
    accs = {(g["class"], g["concept"]): (g["n"], g["accuracy"]) for g in out["groups"]}
    assert accs == {
        (0, True): (2, 0.5),
        (0, False): (1, 1.0),
        (1, True): (1, 1.0),
        (1, False): (2, 0.5),
    }
    assert out["average"] == 0.75 and out["worst"] == 0.5
    empty = group_accuracies([0], [1], [True])
    assert empty["groups"][1]["accuracy"] is None and empty["worst"] == 1.0
