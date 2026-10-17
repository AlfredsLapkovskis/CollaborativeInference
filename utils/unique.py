import torch


def unique_per_row(x: torch.Tensor, pad_value=-1):
    R, C = x.shape
    if R * C == 0:
        return (torch.empty((R, 0), dtype=x.dtype, device=x.device),
                torch.empty((R, 0), dtype=torch.int32, device=x.device),
                torch.empty((R, C), dtype=torch.int32, device=x.device),
                torch.zeros((R, C), dtype=torch.bool, device=x.device))

    x_sorted, sort_idx = torch.sort(x, dim=1)
    sort_idx = sort_idx.to(torch.int32)

    change = torch.ones_like(x_sorted, dtype=torch.bool)
    change[:, 1:] = x_sorted[:, 1:] != x_sorted[:, :-1]

    group_id = change.cumsum(dim=1, dtype=torch.int32) - 1
    num_unique = change.sum(dim=1)
    max_unique = num_unique.max()

    counts = torch.zeros((R, max_unique), dtype=torch.int32, )
    counts.scatter_add_(1, group_id, torch.ones_like(group_id))

    uniques = torch.full((R, max_unique), pad_value, dtype=x.dtype)
    uniques.scatter_(1, group_id, x_sorted)

    inverse = torch.empty_like(x, dtype=torch.int32)
    inverse.scatter_(1, sort_idx, group_id)

    first_idx = torch.full((R, max_unique),
                           C,          # sentinel larger than any valid index
                           dtype=torch.int32,
                           device=x.device)

    first_idx.scatter_reduce_(
        1,
        group_id,
        sort_idx,
        reduce="amin",
        include_self=True,
    )

    first_occurrence_mask = torch.zeros_like(x, dtype=torch.bool)
    row_indices = torch.arange(R, device=x.device).unsqueeze(
        1).expand(R, max_unique)
    valid_mask = first_idx < C
    first_occurrence_mask[row_indices[valid_mask],
                          first_idx[valid_mask]] = True

    return uniques, counts, inverse, first_occurrence_mask


def fill_all_occurrences(input_matrix: torch.Tensor, unique_matrix: torch.Tensor, inverse_index: torch.Tensor, first_idx: torch.Tensor):
    """
    Fills all occurrences in input_matrix based on unique_matrix and inverse_index.
    unique_matrix and inverse_index are obtained from unique_per_row function.
    """

    R, C = input_matrix.shape
    if R * C == 0:
        return torch.empty((R, C), dtype=input_matrix.dtype)

    Kmax = unique_matrix.shape[1]

    input_unique = torch.zeros((R, Kmax), dtype=input_matrix.dtype)

    row_ids = torch.arange(input_matrix.shape[0])[
        :, None].expand(R, C)[first_idx]
    unique_ids = inverse_index[first_idx]
    values = input_matrix[first_idx]

    input_unique[row_ids, unique_ids] = values

    input_filled = input_unique.gather(1, inverse_index)

    return input_filled
