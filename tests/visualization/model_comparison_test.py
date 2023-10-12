import unittest
import numpy as np
import torch
import os

from peal.visualization.model_comparison import change_all, create_checkbox_dict


class TestModelComparison(unittest.TestCase):
    def test_change_all(self):
        x = torch.zeros([2, 2, 2, 2])
        print(change_all(torch.zeros_like(x), 1, 0).flatten())
        self.assertEqual(
            torch.sum(
                torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
                == change_all(torch.zeros_like(x), 0, 0).flatten()
            ),
            16,
        )
        self.assertEqual(
            torch.sum(
                torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
                == change_all(torch.zeros_like(x), 1, 0).flatten()
            ),
            16,
        )
        self.assertEqual(
            torch.sum(
                torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
                == change_all(torch.zeros_like(x), 2, 0).flatten()
            ),
            16,
        )
        self.assertEqual(
            torch.sum(
                torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
                == change_all(torch.zeros_like(x), 3, 0).flatten()
            ),
            16,
        )

    def test_create_checkbox_dict(self):
        checkbox_dict = {
            "is_blond": torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long
            ),
            "has_confounder": torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long
            ),
            "student_correct": torch.tensor(
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long
            ),
            "pclarc_correct": torch.tensor(
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long
            ),
        }
        checkbox_dict_created = create_checkbox_dict(checkbox_dict.keys())
        for key in checkbox_dict.keys():
            self.assertEqual(
                torch.sum(checkbox_dict[key] == checkbox_dict_created[key]), 16
            )
