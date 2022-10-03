import torch
import torch.nn.functional as F


def distributed_sinkhorn(
    out, sinkhorn_iterations=3, epsilon=0.05
):  # (n_pixels, k)  (n, p)
    Q = torch.exp(out / epsilon).t()  # (B, K) -> (K x B)
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)  #
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows  # (K x B)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)  # (K x B) / (1 x B), make sum to 1
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    # Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()
    Q = F.gumbel_softmax(Q, tau=0.5, hard=True)

    return Q, indexs
    # Q: (n, 10) is pixel belongs to which prototype, it's a one-hot label
    # index: (n, ) is prototype index
