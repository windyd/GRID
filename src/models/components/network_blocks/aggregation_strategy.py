from abc import ABC, abstractmethod
from typing import Optional

import torch

# from src.utils.masking_utils import create_last_k_mask


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pass


class MeanAggregation(AggregationStrategy):
    """
    Aggregates the embeddings by computing their mean. If last_k is specified, only the last K embeddings are considered.
    """

    def __init__(self, last_k: Optional[int] = None):
        """
        Initializes the MeanAggregation class with the specified number of last embeddings to consider.

        Args:
            last_k Optional[int] = None
                The number of last K embeddings to consider for aggregation. If None, all embeddings are considered.
        """
        self.last_k = last_k

    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregates the last K embeddings for each row by computing their mean.

        Args:
            embeddings (torch.Tensor): Shape (batch_size, sequence_length, embedding_dim).
                The tensor containing embeddings for each row.
            row_ids (torch.Tensor): Shape (return_size,).
                The tensor containing row ids for which the aggregated embedding has to be returned.
            attention_mask (torch.Tensor): Shape (batch_size, sequence_length).
                The attention mask with 1 for valid tokens and 0 for padding.

        Returns:
            torch.Tensor: The aggregated embeddings of shape (return_size, embedding_dim).
        """
        # Select the embeddings and mask for the specified row ids
        embeddings = embeddings[
            row_ids
        ]  # Shape (return_size, sequence_length, embedding_dim)
        mask = attention_mask[row_ids]  # Shape (return_size, sequence_length)
        mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)

        if self.last_k is not None:
            # If last_k is specified, we need to mask out everything except the last k valid tokens.
            # We can do this by counting valid tokens from the right (end of sequence).
            # 1. Compute cumulative sum of valid tokens from right to left.
            #    We flip the mask, cumsum, and flip back.
            #    mask: [0, 1, 1, 1] (Left padding example)
            #    flip: [1, 1, 1, 0]
            #    cumsum: [1, 2, 3, 3]
            #    flip back: [3, 3, 2, 1] -> effectively "valid index from end"
            # 2. Only Keep the valid tokens
            #    apply mask: [0, 3, 2, 1]

            # Using flip is safe for both left and right padding if we only care about 'valid' tokens (1s).
            # Actually, simpler: cumsum(mask) gives index from start. sum(mask) gives total.
            # distance_from_end = total_valid - current_valid_index + 1

            # Let's stick to the flip-cumsum logic as it's robust.
            valid_count_from_right = torch.flip(
                torch.cumsum(torch.flip(mask, dims=[1]), dim=1), dims=[1]
            )

            # We only keep positions where mask is 1 AND valid_count_from_right <= last_k
            mask = mask * (valid_count_from_right <= self.last_k).to(dtype=mask.dtype)

        # Apply the mask to the embeddings
        masked_embeddings = embeddings * mask.unsqueeze(
            2
        )  # Shape (return_size, sequence_length, embedding_dim)

        # Sum the masked embeddings and divide by the count of non-zero elements in the mask
        sum_embeddings = torch.sum(
            masked_embeddings, dim=1
        )  # Shape (return_size, embedding_dim)
        count = (
            torch.sum(mask, dim=1).clamp(min=1).unsqueeze(1)
        )  # Shape (return_size, embedding_dim)
        return sum_embeddings / count  # Shape (return_size, embedding_dim)


class LastAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # We need to find the index of the last valid token (where mask == 1) for each row.
        # Using the same logic as before: last valid index is where valid_count_from_right == 1
        # Or simpler: argmax of (mask * index_range)

        mask = attention_mask[row_ids]
        seq_len = mask.size(1)

        # Create indices [0, 1, ..., seq_len-1]
        indices = torch.arange(seq_len, device=embeddings.device).unsqueeze(
            0
        )  # (1, seq_len)

        # Multiply indices by mask. Invalid positions become 0.
        # Note: This assumes there's at least one valid token.
        # For [0, 0, 1, 1, 0] (valid at 2, 3), indices are [0, 1, 2, 3, 4].
        # masked_indices = [0, 0, 2, 3, 0]. Argmax is 3. Correct.
        # Edge case: [1, 0, 0] -> masked [0, 0, 0] -> Argmax 0. Correct.
        # Edge case: [0, 0, 1] -> masked [0, 0, 2] -> Argmax 2. Correct.

        # However, if we have left padding and the token at 0 is padding (0),
        # and we have valid tokens later, argmax works.
        # But if we have [0, 0, 0] (empty), argmax is 0.

        # For robustness with left padding, we need to ensure we don't pick 0 if 0 is padding
        # unless it's the only option.
        # Actually, (mask * indices) works for finding the LAST valid index because
        # indices increase to the right. The largest index with mask=1 will be the argmax.
        # UNLESS the only valid token is at 0. Then 0*1 = 0. Argmax is 0. Correct.

        last_item_indices = (mask * indices).argmax(dim=1)

        # Using gather to select the embedding at the calculated index for each row
        # embeddings: (return_size, seq_len, dim)
        # last_item_indices: (return_size,) -> (return_size, 1, 1)

        return embeddings[row_ids, last_item_indices]


class FirstAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Return the first valid item in each sequence.
        # For right padding: [1, 1, 0] -> index 0.
        # For left padding: [0, 1, 1] -> index 1.

        mask = attention_mask[row_ids]

        # argmax on the boolean mask returns the index of the first True (1) value.
        first_item_indices = mask.argmax(dim=1)

        return embeddings[row_ids, first_item_indices]
