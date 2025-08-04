# stephanie/data/loaders/plan_trace_loader.py

import json
import os
from typing import List, Optional, Union

# Assuming PlanTrace and ExecutionStep dataclasses exist
# Adjust the import path as needed based on your project structure.
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.data.score_bundle import \
    ScoreBundle  # Needed for reconstruction


def load_plan_traces(source: Union[str, List[Union[dict, PlanTrace]]], 
                     target_quality_required: bool = True) -> List[PlanTrace]:
    """
    Loads PlanTrace objects from various sources.

    Args:
        source: The source of the data. Can be:
                - A list of PlanTrace objects or dictionaries representing them.
                - A path to a single JSON file containing a list of serialized PlanTraces.
                - A path to a directory containing individual JSON files, each 
                  representing one PlanTrace.
        target_quality_required: If True, only PlanTraces with a non-None
                                 target_epistemic_quality will be included in the output.

    Returns:
        A list of loaded PlanTrace objects.

    Raises:
        FileNotFoundError: If the specified file or directory does not exist.
        ValueError: If the source format is not recognized or data is malformed.
        Exception: For other unexpected errors during loading/parsing.
    """
    loaded_traces = []

    if isinstance(source, list):
        # Source is already a list of objects/dicts
        print("Loading PlanTraces from provided list...")
        raw_data_list = source

    elif isinstance(source, str):
        # Source is a file path or directory path
        if not os.path.exists(source):
            raise FileNotFoundError(f"The specified path does not exist: {source}")

        if os.path.isfile(source):
            # Single file (e.g., JSON array)
            print(f"Loading PlanTraces from file: {source}")
            try:
                with open(source, 'r') as f:
                    raw_data_list = json.load(f)
                if not isinstance(raw_data_list, list):
                    raise ValueError(f"JSON file {source} must contain a list of trace objects.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from file {source}: {e}")

        elif os.path.isdir(source):
            # Directory of files
            print(f"Loading PlanTraces from directory: {source}")
            raw_data_list = []
            for filename in os.listdir(source):
                file_path = os.path.join(source, filename)
                if os.path.isfile(file_path) and filename.lower().endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            trace_data = json.load(f)
                            # Add filename as potential metadata
                            # trace_data['_source_file'] = filename # Optional
                            raw_data_list.append(trace_data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping file {filename} due to JSON decode error: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error loading file {filename}: {e}")
                        continue
        else:
            raise ValueError(f"Path {source} is neither a file nor a directory.")

    else:
        raise ValueError(f"Unsupported source type: {type(source)}. Expected list or string path.")

    # --- Convert raw data (dicts) to PlanTrace objects ---
    print(f"Processing {len(raw_data_list)} raw trace data items...")
    for i, raw_data in enumerate(raw_data_list):
        try:
            if isinstance(raw_data, PlanTrace): # Already an object
                trace_obj = raw_data
            elif isinstance(raw_data, dict): # Need to convert from dict
                trace_obj = _dict_to_plan_trace(raw_data)
            else:
                print(f"Warning: Skipping item {i} as it's not a dict or PlanTrace object.")
                continue

            # Filter based on target quality requirement
            if target_quality_required and trace_obj.target_epistemic_quality is None:
                # print(f"Skipping trace {trace_obj.trace_id} due to missing target_epistemic_quality.")
                continue # Skip traces without the required target

            loaded_traces.append(trace_obj)

        except Exception as e:
            print(f"Warning: Error processing trace data item {i}: {e}")
            # Depending on robustness needs, you could log, skip, or re-raise
            continue 

    print(f"Successfully loaded {len(loaded_traces)} PlanTrace objects.")
    return loaded_traces


def _dict_to_plan_trace(data: dict) -> PlanTrace:
    """
    Converts a dictionary representation into a PlanTrace object.
    Handles reconstruction of nested ExecutionStep and ScoreBundle objects.
    """
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")

    # Extract core fields, providing defaults for missing optional ones
    trace_kwargs = {
        "trace_id": data.get("trace_id", f"unknown_trace_{id(data)}"),
        "goal_text": data["goal_text"], # Required
        "goal_embedding": data["goal_embedding"], # Required
        "input_data": data.get("input_data", {}),
        "plan_signature": data.get("plan_signature", "unknown"),
        "execution_steps": [],
        "final_output_text": data["final_output_text"], # Required
        "final_scores": data.get("final_scores"), # Will be processed below
        "final_output_embedding": data.get("final_output_embedding"),
        "target_epistemic_quality": data.get("target_epistemic_quality"),
        "target_epistemic_quality_source": data.get("target_epistemic_quality_source"),
        "created_at": data.get("created_at", ""),
        "meta": data.get("meta", {}),
    }

    # --- Reconstruct Execution Steps ---
    raw_steps_data = data.get("execution_steps", [])
    reconstructed_steps = []
    for i, step_data in enumerate(raw_steps_data):
        if isinstance(step_data, ExecutionStep):
            reconstructed_steps.append(step_data)
        elif isinstance(step_data, dict):
            try:
                # Reconstruct ExecutionStep fields
                step_kwargs = {
                    "step_id": step_data.get("step_id", f"step_{i}"),
                    "description": step_data.get("description", ""),
                    "output_text": step_data.get("output_text", ""),
                    "scores": step_data.get("scores"), # Will be processed below
                    "output_embedding": step_data.get("output_embedding"),
                    "meta": step_data.get("meta", {}),
                }

                # --- Reconstruct ScoreBundle for the step ---
                raw_step_scores = step_kwargs["scores"]
                if raw_step_scores is not None:
                    if isinstance(raw_step_scores, ScoreBundle):
                        step_kwargs["scores"] = raw_step_scores
                    elif isinstance(raw_step_scores, dict):
                        # If ScoreBundle is just a dict or has a from_dict method
                        # step_kwargs["scores"] = reconstruct_score_bundle(raw_step_scores)
                        # For now, assuming it can be passed as dict if ScoreBundle is dict-like
                        # Or if it's a dataclass, it might need manual reconstruction
                        # Placeholder: Assume it's compatible or needs simple dict passing
                        # This part is highly dependent on the real ScoreBundle structure
                        # Let's assume for now it's handled correctly by PlanTrace constructor
                        # if SCORE_BUNDLE_AVAILABLE and hasattr(ScoreBundle, 'from_dict'):
                        #     step_kwargs["scores"] = ScoreBundle.from_dict(raw_step_scores)
                        # else:
                        step_kwargs["scores"] = raw_step_scores # Pass dict as-is for now
                    else:
                        # Unexpected type for scores
                        print(f"Warning: Unexpected type for scores in step {step_kwargs['step_id']}: {type(raw_step_scores)}")
                        step_kwargs["scores"] = {} # Default to empty dict

                reconstructed_step = ExecutionStep(**step_kwargs)
                reconstructed_steps.append(reconstructed_step)
            except Exception as e:
                print(f"Error reconstructing ExecutionStep {i} for trace {trace_kwargs['trace_id']}: {e}")
                # Depending on policy, raise or create a placeholder step
                # raise e # Re-raise if strict
                # Or create a placeholder
                # placeholder_step = ExecutionStep(step_id=f"error_step_{i}", description="Reconstruction Error", output_text="", scores={}, meta={"error": str(e)})
                # reconstructed_steps.append(placeholder_step)
                continue # Skip this step
        else:
            print(f"Warning: Skipping execution step {i} for trace {trace_kwargs['trace_id']} due to unexpected type: {type(step_data)}")
            continue

    trace_kwargs["execution_steps"] = reconstructed_steps

    # --- Reconstruct Final ScoreBundle ---
    raw_final_scores = trace_kwargs["final_scores"]
    if raw_final_scores is not None:
        if isinstance(raw_final_scores, ScoreBundle):
            trace_kwargs["final_scores"] = raw_final_scores
        elif isinstance(raw_final_scores, dict):
            # Similar logic as for step scores
            # trace_kwargs["final_scores"] = reconstruct_score_bundle(raw_final_scores)
            # Placeholder assumption
            # if SCORE_BUNDLE_AVAILABLE and hasattr(ScoreBundle, 'from_dict'):
            #     trace_kwargs["final_scores"] = ScoreBundle.from_dict(raw_final_scores)
            # else:
            trace_kwargs["final_scores"] = raw_final_scores # Pass dict as-is
        else:
            print(f"Warning: Unexpected type for final_scores in trace {trace_kwargs['trace_id']}: {type(raw_final_scores)}")
            trace_kwargs["final_scores"] = {} # Default to empty dict

    # --- Create the PlanTrace object ---
    # This assumes PlanTrace is a dataclass or has a constructor accepting these kwargs
    plan_trace_obj = PlanTrace(**trace_kwargs)
    return plan_trace_obj

# Example Usage (Conceptual):
# 1. From a list in memory:
# traces_from_memory = [PlanTrace(...), PlanTrace(...)]
# loaded_traces = load_plan_traces(traces_from_memory, target_quality_required=True)

# 2. From a single JSON file:
# loaded_traces = load_plan_traces("path/to/traces.json", target_quality_required=True)

# 3. From a directory of JSON files:
# loaded_traces = load_plan_traces("path/to/trace_directory/", target_quality_required=True)
