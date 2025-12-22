import sys, os

sys.path.insert(0, os.getcwd())

import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from utils import log_paths

class MLPContrastive(nn.Module):
    def __init__(self, clip_feat_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * clip_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single similarity score per (i,k) pair
        )

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor):
        """
        Args:
            image_feat: (N, D) image features
            text_feat: (K, D) text features (for K materials)

        Returns:
            sim_table: (N, K) similarity score for each image i and material k
        """
        N, D = image_feat.shape
        K = text_feat.shape[0]

        # Expand and concat
        image_feat_exp = image_feat.unsqueeze(1).expand(N, K, D)   # (N, K, D)
        text_feat_exp = text_feat.unsqueeze(0).expand(N, K, D)     # (N, K, D)
        concat_feat = torch.cat([image_feat_exp, text_feat_exp], dim=-1)  # (N, K, 2D)

        # Compute similarity logits via MLP
        sim = self.mlp(concat_feat).squeeze(-1)  # (N, K)
        return sim

class MaskSelfAttentionAggregator(nn.Module):
    def __init__(self, input_dim, attn_dim=64):
        super().__init__() 
        self.query_proj = nn.Linear(input_dim, attn_dim)
        self.key_proj = nn.Linear(input_dim, attn_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, features, mask_ids):
        """
        Vectorized forward pass for mask-based self-attention.
        Args:
            features (torch.Tensor): A tensor of shape (N, D) containing point features.
            mask_ids (List[Set[int]]): A list of length N, where each element is a set of
                                       mask indices corresponding to that point.
        Returns:
            torch.Tensor: A tensor of shape (N, D) with aggregated features.
        """
        N, D = features.shape
        device = features.device

        # Step 1: Project features into Query, Key, and Value spaces
        Q = self.query_proj(features)  # (N, attn_dim)
        K = self.key_proj(features)  # (N, attn_dim)
        V = self.value_proj(features)  # (N, D)

        # Step 2: Build the adjacency matrix for attention
        # This matrix indicates which points share a mask.
        adjacency = torch.zeros(N, N, device=device, dtype=torch.bool)
        
        # Create a reverse mapping from a mask ID to the points it contains.
        mask_to_indices = defaultdict(list)
        for i, mask_set in enumerate(mask_ids):
            for m in mask_set:
                mask_to_indices[m].append(i)

        # For each mask, mark all points within that mask as neighbors.
        for indices in mask_to_indices.values():
            # Create a grid of indices to efficiently mark pairs.
            idx_grid = torch.tensor(indices, device=device)
            rows, cols = torch.meshgrid(idx_grid, idx_grid, indexing='ij')
            adjacency[rows, cols] = True
        
        # Ensure points don't attend to themselves if they have no other neighbors.
        # This is implicitly handled by the softmax later, but good practice.
        adjacency.fill_diagonal_(True)

        # Step 3: Compute attention scores in a batched manner
        # The attention score between query i and key j is Q[i] @ K[j].T
        # This is a large matrix multiplication: (N, attn_dim) @ (attn_dim, N) -> (N, N)
        scores = torch.matmul(Q, K.T) / (K.shape[-1] ** 0.5)

        # Step 4: Apply the adjacency mask
        # We use a large negative value for non-adjacent points to make their
        # softmax scores effectively zero.
        attention_mask = torch.full_like(scores, -1e9)
        masked_scores = torch.where(adjacency, scores, attention_mask)

        # Step 5: Compute softmax and aggregate values
        # The softmax is computed row-wise. Each row i represents the attention
        # weights from point i to all other points j.
        weights = F.softmax(masked_scores, dim=-1) # (N, N)

        # Check for NaNs which can occur if a point has no neighbors.
        # In such cases, the attention weights for that row will be all NaNs.
        # We can replace these with zeros, so the point's feature remains unchanged.
        weights = torch.nan_to_num(weights, nan=0.0)

        # Aggregate the values using the computed weights.
        # (N, N) @ (N, D) -> (N, D)
        agg_features = torch.matmul(weights, V)

        # Step 6: Handle points with no neighbors
        # For points that had no neighbors, their aggregated feature will be zero.
        # We should use their original feature instead.
        no_neighbors_mask = (adjacency.sum(dim=1) <= 1)
        agg_features[no_neighbors_mask] = features[no_neighbors_mask]

        return agg_features

class MaskAwareMLPContrastive(nn.Module):
    def __init__(self, clip_feat_dim, hidden_dim=512, attn_dim=64):
        super().__init__()
        self.mask_aggregator = MaskSelfAttentionAggregator(clip_feat_dim, attn_dim)

        input_dim = 2 * clip_feat_dim
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, image_feat, text_feat, mask_ids):
        """
        image_feat: (N, D)
        text_feat:  (K, D)
        mask_ids:   List[Set[int]] of length N
        Returns:
            sim_table: (N, K)
        """
        N, D = image_feat.shape
        K = text_feat.shape[0]


        image_feat = F.normalize(image_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=1)

        #print(f"[feat means] image: {image_feat.mean().item():.4f}, text: {text_feat.mean().item():.4f}")


        # Attention-based aggregation within masks
        agg_feat = self.mask_aggregator(image_feat, mask_ids)  # (N, D)

        # Compute similarity with each material
        agg_feat_exp = agg_feat.unsqueeze(1).expand(N, K, D)
        text_feat_exp = text_feat.unsqueeze(0).expand(N, K, D)

        # Before MLP
        pairwise_input = torch.cat([agg_feat_exp, text_feat_exp], dim=-1)  # (N, K, 2D)

        if not torch.isfinite(pairwise_input).all():
            print("[‼️] NaN or Inf in pairwise_input")
            print("    stats:", pairwise_input.min().item(), pairwise_input.max().item())

        # Pass through MLP
        sim = self.mlp(pairwise_input).squeeze(-1)  # (N, K)

        # After MLP
        if not torch.isfinite(sim).all():
            print("[‼️] NaN in sim output")
            print("    stats:", sim.min().item(), sim.max().item())

        return sim

    def get_mask_aggregator_output(self, image_feat, mask_ids):
        """
        Returns the aggregated features from the mask-aware aggregator.
        This is useful for debugging or further processing.
        """
        
        image_feat = F.normalize(image_feat, dim=1)
        return self.mask_aggregator(image_feat, mask_ids)

def supervised_contrastive_loss(z, mask_ids, tau=0.1, args=None):
    """
    Fully vectorized supervised contrastive loss.
    Args:
        z: Tensor of shape (N, D) — feature vectors.
        mask_ids: List[Set[int]] of length N — each set contains non-negative
                  mask indices for that point.
        tau: temperature scalar.
    Returns:
        A scalar contrastive loss (torch.Tensor) connected to z's computation graph.
    """
    
    device = z.device
    N = z.shape[0]

    # Handle N=0 or N=1 cases where contrastive loss is not well-defined or is 0.
    # Initialize with a graph-connected zero.
    if N == 0:
        # If z is an empty tensor (e.g., torch.empty((0,D), requires_grad=True)),
        # z.sum()*0.0 will correctly produce a tensor(0., grad_fn=<MulBackward0>)
        # or tensor(0.) if requires_grad is False.
        return z.sum() * 0.0 
    if N == 1: # No other samples to contrast with.
        return z.sum() * 0.0

    # Initialize a graph-connected zero for early exits if no valid loss terms are computed.
    graph_connected_zero = z.sum() * 0.0

    # 1. Build bin_masks: shape (N, M)
    # M is the number of unique mask categories (max_mask_id + 1).
    # This part is from your original code and is efficient.
    # It correctly handles cases where mask_ids might be empty or contain empty sets.
    # It also assumes mask indices are non-negative.
    all_mask_values = [m for s in mask_ids if s for m in s]
    if not all_mask_values: # No masks defined at all
        M = 0
    else:
        if any(m < 0 for m in all_mask_values):
            raise ValueError("Mask indices must be non-negative.")
        M = max(all_mask_values) + 1

    if M == 0:
        # No actual mask categories exist to form positives.
        return graph_connected_zero

    bin_masks = torch.zeros((N, M), dtype=torch.bool, device=device)
    for i, s_per_sample in enumerate(mask_ids):
        if s_per_sample: # If the set for this sample is not empty
            # Filter for valid indices, although M should ensure this.
            valid_indices_in_set = [m_val for m_val in s_per_sample if 0 <= m_val < M]
            if valid_indices_in_set:
                bin_masks[i, valid_indices_in_set] = True
    
    # If no sample has any mask after building bin_masks (e.g., all mask_ids were empty sets)
    if not bin_masks.any():
        return graph_connected_zero

    # 2. Create positive pair mask (pos_mask[i,j] is True if j is a positive for anchor i)
    #   pos_mask[i,j] means sample j shares at least one mask with sample i, AND i != j.
    #   bin_masks.unsqueeze(1): (N, 1, M)
    #   bin_masks.unsqueeze(0): (1, N, M)
    #   shared_specific_masks[i,j,k] is true if sample i AND sample j both have mask k.
    shared_specific_masks = bin_masks.unsqueeze(1) & bin_masks.unsqueeze(0) # Shape: (N, N, M)
    
    # has_shared_mask[i,j] is true if sample i and sample j share AT LEAST ONE mask.
    has_shared_mask = shared_specific_masks.any(dim=2) # Shape: (N, N)
    
    # identity_mask is True where i != j (used to exclude self-similarity).
    identity_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = has_shared_mask & identity_mask # Shape: (N, N)

    num_positives = pos_mask.sum().item()
    pos_ratio = num_positives / (N * (N - 1))
    
    #log_paths(args.log_file, [f"[Pos ratio] {pos_ratio:.4f}, total pairs: {N * (N - 1)}"])

    # If no positive pairs exist anywhere in the batch, loss is 0.
    if not pos_mask.any():
        return graph_connected_zero

    # 3. Calculate all-pairs similarities (dot product)
    # z is (N, D). z.T is (D, N).
    # sim_matrix[i,j] = dot(z_i, z_j) / tau
    sim_matrix = torch.matmul(z, z.T) / tau # Shape: (N, N)

    sim_matrix = torch.clamp(sim_matrix, min=-50.0, max=50.0)

    # 4. Calculate denominators for each anchor
    # Denominator for anchor i: sum_{k!=i} exp(sim(z_i, z_k)/tau)
    exp_sim_matrix = torch.exp(sim_matrix)
    
    if not torch.isfinite(exp_sim_matrix).all():
        print("[‼️] exp(sim_matrix) has inf or nan!")
        print("  sim_matrix stats:", sim_matrix.min().item(), sim_matrix.max().item())


    # Zero out diagonal (self-similarity) for denominator calculation using identity_mask.
    exp_sim_for_denom = exp_sim_matrix * identity_mask
    denominators = exp_sim_for_denom.sum(dim=1) # Shape: (N,). Sum over k for each anchor i.
    
    # Clamp denominators to avoid log(0) or division by very small numbers.
    denominators = torch.clamp(denominators, min=1e-9) 

    # 5. Calculate log-probabilities for all pairs using the numerically stable form:
    # log_probs_matrix[i,j] = sim_matrix[i,j] - log(denominators[i])
    # Need to broadcast denominators (N,) to (N,N) for subtraction.
    log_probs_matrix = sim_matrix - torch.log(denominators.unsqueeze(1)) # Shape: (N, N)

    # 6. Calculate loss per anchor
    # Loss_i = -1/|P_i| * sum_{p in P_i} log_probs_matrix[i,p]
    # where P_i is the set of positives for anchor i.
    
    nll_matrix = -log_probs_matrix # Negative log-likelihood for each pair (i,j). Shape: (N, N)
    
    # Sum NLL only for positive pairs for each anchor.
    # (nll_matrix * pos_mask) will zero out terms where (i,j) is not a positive pair.
    sum_nll_for_positives_per_anchor = (nll_matrix * pos_mask).sum(dim=1) # Shape: (N,)
    
    # Number of positive examples for each anchor.
    num_positives_per_anchor = pos_mask.sum(dim=1).float() # Shape: (N,)
    
    skipped = (num_positives_per_anchor == 0).sum().item()

    #log_paths(args.log_file, [f"[Skipped anchors] {skipped}/{N}"])

    # Average NLL over positives for each anchor.
    # For anchors with no positives (num_positives_per_anchor[i] == 0),
    # sum_nll_for_positives_per_anchor[i] will also be 0 (due to pos_mask).
    # So, 0 / clamp(0, min=1.0) = 0 / 1.0 = 0 for these anchors.
    loss_per_anchor = sum_nll_for_positives_per_anchor / torch.clamp(num_positives_per_anchor, min=1.0)

    # 7. Total loss: sum of mean losses for anchors that contributed (had positives).
    # Since loss_per_anchor is 0 for anchors without positives or those that had no masks,
    # we can sum all elements of loss_per_anchor. This matches the structure of your
    # original code: `loss += loss_i.mean()`.
    total_loss = loss_per_anchor.sum()
    
    return total_loss / N  # Normalize by N to get average loss per sample

def calculate_combined_loss(sim_mlp, sim_dot, mask_ids, l2_lambda=0.1, contrastive_tau=0.1, args=None):
    """
    Calculates the combined contrastive and L2 regularization loss.

    Args:
        sim_mlp (torch.Tensor): The (N, K) similarity matrix from the MLP.
        sim_dot (torch.Tensor): The (N, K) similarity matrix from the dot product.
        mask_ids (list): The list of mask sets for the contrastive loss.
        l2_lambda (float): The weight for the L2 regularization term.
        contrastive_tau (float): The temperature for the contrastive loss.

    Returns:
        A tuple containing: (total_loss, contrastive_loss, l2_loss)
    """
    # 1. Supervised Contrastive Loss
    # This loss operates on the rows of the MLP similarity matrix. It encourages
    # two image points that share a mask to have similar relationships (similarity profiles)
    # to all text queries.
    contrastive_loss = supervised_contrastive_loss(
        z=sim_mlp,  # Using the (N,K) MLP similarity matrix as input features
        mask_ids=mask_ids,
        tau=contrastive_tau,
        args=args
    )

    # 2. L2 Regularization Loss (Mean Squared Error)
    # This penalizes the MLP for deviating from the dot product similarity.
    # We use `.detach()` on `sim_dot` so gradients only flow to `sim_mlp`,
    # effectively regularizing the MLP without trying to change the source features.
    l2_regularization_loss = F.mse_loss(sim_mlp, sim_dot.detach())

    # print('contrastive_loss', contrastive_loss.item())
    # print('l2_regularization_loss', l2_regularization_loss.item())
    # 3. Combine the losses
    total_loss = contrastive_loss + l2_lambda * l2_regularization_loss

    #print('total_loss', total_loss.item())
    return total_loss, contrastive_loss, l2_lambda*l2_regularization_loss

import torch
import torch.nn.functional as F

def calculate_combined_loss_entropy(
        sim_mlp: torch.Tensor,
        sim_dot: torch.Tensor,
        mask_ids,
        ce_lambda: float = 0.1,
        contrastive_tau: float = 0.1,
        args=None,
):
    """
    Combined loss = supervised-contrastive + λ * hard-cross-entropy (teacher labels).

    Args:
        sim_mlp (Tensor): (N, K) similarity matrix predicted by the MLP.
        sim_dot (Tensor): (N, K) similarity matrix from the CLIP dot-product (teacher).
        mask_ids (List[Set[int]]): mask grouping for the contrastive loss.
        ce_lambda (float): weight for the cross-entropy term.
        contrastive_tau (float): temperature for the contrastive loss.
        args: extra namespace forwarded to `supervised_contrastive_loss`.

    Returns:
        Tuple(total_loss, contrastive_loss, weighted_ce_loss)
    """
    # 1. Supervised contrastive loss on sim_mlp rows
    contrastive_loss = supervised_contrastive_loss(
        z=sim_mlp,
        mask_ids=mask_ids,
        tau=contrastive_tau,
        args=args,
    )

    # 2. Hard cross-entropy to keep the teacher’s arg-max material
    with torch.no_grad():
        teacher_labels = sim_dot.argmax(dim=1).long()        # shape (N,)

    ce_loss = F.cross_entropy(sim_mlp, teacher_labels)

    # 3. Combine losses
    total_loss = contrastive_loss + ce_lambda * ce_loss

    return total_loss, contrastive_loss, ce_lambda * ce_loss