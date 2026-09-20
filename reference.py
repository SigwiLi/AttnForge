import torch

def alibi_reference(scores, slopes):
    """
    
    scores: [B, H, H, D], floating point attention scores
    slopes: [H], on the same device and dtype as scores

    Returns scores with ALiBi and a causal mask, before softmax
    """

    seq_len = scores.shape[-1]

    positions = torch.arange(seq_len)

    i = positions[:, None] # Query -> Columns
    j = positions[None, :] # Keys -> Rows

    distance = i - j
    head_slopes = slopes[None, :, None, None]
    bias = -head_slopes * distance

    modified_scores = scores + bias
    return modified_scores.masked_fill(j > i, float('-inf'))


if __name__ == "__main__":
    batch_size = 1
    num_heads = 2
    seq_len = 4

    torch.manual_seed(42)
    scores = torch.randn(
        batch_size, num_heads, seq_len, seq_len
    )

    slopes = torch.tensor([0.5, 0.25])

    masked_scores = alibi_reference(scores, slopes)
    weights = torch.softmax(masked_scores, dim=-1)

    print("Masked scores:\n", masked_scores[0, 0])
    print("Attention weights:\n", weights[0, 0])
    print(1.6487 * -0.5)