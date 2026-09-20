import torch

from reference import alibi_reference


def test_known_values():
    scores = torch.zeros(1, 2, 4, 4)
    slopes = torch.tensor([0.5, 0.25])

    actual = alibi_reference(scores, slopes)

    expected_head_0 = torch.tensor([
        [ 0.0,           -torch.inf, -torch.inf, -torch.inf],
        [-0.5,            0.0,       -torch.inf, -torch.inf],
        [-1.0,           -0.5,        0.0,       -torch.inf],
        [-1.5,           -1.0,       -0.5,        0.0],
    ])

    torch.testing.assert_close(actual[0, 0], expected_head_0)


def test_future_positions_are_masked():
    torch.manual_seed(42)

    scores = torch.randn(2, 3, 8, 8)
    slopes = torch.tensor([0.5, 0.25, 0.125])

    masked_scores = alibi_reference(scores, slopes)

    future_mask = torch.triu(
        torch.ones(8, 8, dtype=torch.bool),
        diagonal=1,
    )

    assert torch.isneginf(masked_scores[:, :, future_mask]).all()


def test_attention_rows_sum_to_one():
    torch.manual_seed(42)

    scores = torch.randn(2, 3, 8, 8)
    slopes = torch.tensor([0.5, 0.25, 0.125])

    masked_scores = alibi_reference(scores, slopes)
    weights = torch.softmax(masked_scores, dim=-1)

    row_sums = weights.sum(dim=-1)

    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
    )