# stephanie/models/epistemic/trace_encoder.py

import torch
import torch.nn as nn
import numpy as np
# Assuming PlanTrace and ExecutionStep dataclasses exist
# from stephanie.data_structures.plan_trace import PlanTrace, ExecutionStep 
# Assuming access to embedding store or memory
# from stephanie.memory.embedding_store import EmbeddingStore 

# If the dataclasses are not in those exact paths, adjust the imports accordingly.
# For self-containment, let's assume PlanTrace/ExecutionStep are dict-based or defined elsewhere for now.

class EpistemicTraceEncoder(nn.Module):
    """
    Encodes a PlanTrace into a fixed-size embedding for the Epistemic Plan HRM.
    This version focuses on aggregating goal, final output, and statistical 
    features derived from scorer internal states across the trace steps.
    """
    def __init__(self, embedding_dim: int, stats_input_dim: int = 12, hidden_dim: int = 256, output_dim: int = 128):
        """
        Initializes the Epistemic Trace Encoder.

        Args:
            embedding_dim (int): The dimension of the input embeddings (e.g., from H-Net).
            stats_input_dim (int): The dimension of the concatenated statistics vector 
                                   derived from scorer outputs. Default 12 assumes 
                                   [mean, std, final] for 4 metrics (e.g., sicql_q, sicql_v, ebt_energy, ebt_uncertainty).
            hidden_dim (int): The hidden dimension for internal processing layers.
            output_dim (int): The dimension of the final encoded trace embedding (z_trace).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.stats_input_dim = stats_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layers to process statistics aggregated from scores across steps
        # Input: stats_vector (stats_input_dim,)
        self.stats_projector = nn.Sequential(
            nn.Linear(stats_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Layer to combine goal embedding, final output embedding, and processed stats
        # Input: cat(goal_emb, final_out_emb, processed_stats) -> (embedding_dim + embedding_dim + hidden_dim,)
        self.combiner = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim + hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, trace, embedding_store_or_memory):
        """
        Encodes the trace into a single vector.
        
        Args:
            trace (dict or PlanTrace object): The plan trace data structure. 
                                              Expected keys/attributes if dict:
                                              - 'goal_embedding': List/np.array
                                              - 'final_output_embedding': List/np.array (optional)
                                              - 'final_output_text': str (used if final_output_embedding not present)
                                              - 'execution_steps': List of dicts/steps, each with a 'scores' key (ScoreBundle)
                                              - 'final_scores': ScoreBundle for the final output
            embedding_store_or_memory (object): An object with a method like 
                                                `get_or_create(text)` to obtain embeddings.
                                                This could be `self.memory` from an agent.
        Returns:
            torch.Tensor: Encoded trace embedding of shape (1, output_dim).
        """
        # --- 1. Get Goal and Final Output Embeddings ---
        # Handle potential differences in data structure (dict vs object)
        if hasattr(trace, 'goal_embedding'):
            goal_emb_np = np.array(trace.goal_embedding)
        else: # Assume dict
            goal_emb_np = np.array(trace['goal_embedding'])
            
        goal_emb = torch.tensor(goal_emb_np, dtype=torch.float32) # Shape: (embedding_dim,)

        # Get final output embedding
        final_out_emb = None
        if hasattr(trace, 'final_output_embedding') and getattr(trace, 'final_output_embedding', None) is not None:
            final_out_emb_np = np.array(trace.final_output_embedding)
        elif isinstance(trace, dict) and trace.get('final_output_embedding') is not None:
             final_out_emb_np = np.array(trace['final_output_embedding'])
        else:
            # Fallback to computing embedding from text
            final_out_text = ""
            if hasattr(trace, 'final_output_text'):
                final_out_text = trace.final_output_text
            elif isinstance(trace, dict):
                final_out_text = trace.get('final_output_text', "")
            # Use embedding store/memory to get embedding
            # Assuming embedding_store_or_memory has a .embedding attribute with .get_or_create
            # Adjust based on your actual memory/embedding store API
            try:
                final_out_emb_np = np.array(embedding_store_or_memory.embedding.get_or_create(final_out_text))
            except AttributeError:
                 # Fallback if structure is different
                 final_out_emb_np = np.array(embedding_store_or_memory.get_or_create(final_out_text))

        final_out_emb = torch.tensor(final_out_emb_np, dtype=torch.float32) # Shape: (embedding_dim,)

        # --- 2. Extract and Process Score Statistics ---
        # Collect values from all score bundles (intermediate steps + final)
        sicql_q_values = []
        sicql_v_values = []
        ebt_energies = []
        ebt_uncertainties = []

        # Get all score bundles
        all_bundles = []
        # Handle intermediate steps
        if hasattr(trace, 'execution_steps'):
            steps = trace.execution_steps
        elif isinstance(trace, dict):
            steps = trace.get('execution_steps', [])
        else:
            steps = []
            
        for step in steps:
            if hasattr(step, 'scores'):
                all_bundles.append(step.scores)
            elif isinstance(step, dict) and 'scores' in step:
                all_bundles.append(step['scores'])

        # Add final scores
        if hasattr(trace, 'final_scores'):
            all_bundles.append(trace.final_scores)
        elif isinstance(trace, dict) and 'final_scores' in trace:
            all_bundles.append(trace['final_scores'])

        # Extract relevant metrics from each bundle
        # Assuming ScoreBundle.results is a dict like {'dimension_name': ScoreResult}
        # and ScoreResult has attributes like q_value, state_value, energy, uncertainty
        for bundle in all_bundles:
             # Assume we are interested in a primary dimension, e.g., 'alignment'
             # This could be made configurable
             primary_dimension = 'alignment' # Or passed as an argument
             
             # --- SICQL Metrics ---
             sicql_result = bundle.results.get(primary_dimension) if hasattr(bundle, 'results') else bundle.get('results', {}).get(primary_dimension)
             if sicql_result:
                 # Use getattr for objects, dict.get for dicts
                 q_val = getattr(sicql_result, 'q_value', None) if not isinstance(sicql_result, dict) else sicql_result.get('q_value')
                 v_val = getattr(sicql_result, 'state_value', None) if not isinstance(sicql_result, dict) else sicql_result.get('state_value')
                 if q_val is not None: sicql_q_values.append(float(q_val))
                 if v_val is not None: sicql_v_values.append(float(v_val))

             # --- EBT Metrics ---
             ebt_result = bundle.results.get(primary_dimension) if hasattr(bundle, 'results') else bundle.get('results', {}).get(primary_dimension) # Could be different dim
             if ebt_result:
                 energy_val = getattr(ebt_result, 'energy', None) if not isinstance(ebt_result, dict) else ebt_result.get('energy')
                 uncertainty_val = getattr(ebt_result, 'uncertainty', None) if not isinstance(ebt_result, dict) else ebt_result.get('uncertainty')
                 if energy_val is not None: ebt_energies.append(float(energy_val))
                 if uncertainty_val is not None: ebt_uncertainties.append(float(uncertainty_val))

        # --- Calculate Statistics ---
        def calc_stats(values):
            """Calculate mean, std, and final value for a list."""
            if not values: 
                # Return zeros or a default if no values
                # The size of the stats vector component must be consistent
                # For [mean, std, final], return [0.0, 0.0, 0.0]
                return [0.0, 0.0, 0.0] 
            return [float(np.mean(values)), float(np.std(values)), values[-1]] # mean, std, final

        sicql_q_stats = calc_stats(sicql_q_values)
        sicql_v_stats = calc_stats(sicql_v_values)
        ebt_energy_stats = calc_stats(ebt_energies)
        ebt_uncertainty_stats = calc_stats(ebt_uncertainties)

        # Concatenate all stats into a single vector
        # Ensure the size matches stats_input_dim (12 in this default setup)
        stats_vector_np = np.array(sicql_q_stats + sicql_v_stats + ebt_energy_stats + ebt_uncertainty_stats)
        # Debug print to verify size (optional)
        # print(f"Stats vector shape: {stats_vector_np.shape}") 
        stats_vector = torch.tensor(stats_vector_np, dtype=torch.float32) # Shape: (stats_input_dim,) e.g., (12,)

        # --- 3. Process stats through the projector ---
        # Add batch dimension for the linear layer: (stats_input_dim,) -> (1, stats_input_dim)
        processed_stats = self.stats_projector(stats_vector.unsqueeze(0)) # Output: (1, hidden_dim)

        # --- 4. Combine all parts ---
        # Add batch dimension to goal and final output embeddings: (dim,) -> (1, dim)
        # torch.cat expects tensors with the same number of dimensions
        combined_input = torch.cat([
            goal_emb.unsqueeze(0),       # (1, embedding_dim)
            final_out_emb.unsqueeze(0),  # (1, embedding_dim)
            processed_stats              # (1, hidden_dim)
        ], dim=-1)  # Result: (1, embedding_dim + embedding_dim + hidden_dim)
        
        # --- 5. Final combination to produce z_trace ---
        z_trace = self.combiner(combined_input) # Input: (1, ...), Output: (1, output_dim)

        return z_trace # Shape: (1, output_dim)

# Note: This encoder is designed to be simple and compatible with the HRM input size.
# It focuses on easily derivable statistics. More complex encoders could process
# the sequence of step embeddings or scorer states using RNNs or Transformers.
