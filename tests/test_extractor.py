# test_extractor.py
from stephanie.memory.memory_tool import Session
import torch
import numpy as np
from stephanie.zero.casebook_residual_extractor import CaseBookResidualExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_skill_extraction():
    session = Session()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    
    # Create dummy models (in practice, these would be real checkpoints)
    model_before = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    model_after = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    
    # Mock some parameter changes
    with torch.no_grad():
        # Modify some layers to simulate training effect
        model_after.transformer.h[0].attn.c_attn.weight.add_(
            0.1 * torch.randn_like(model_after.transformer.h[0].attn.c_attn.weight)
        )
    
    # Extract skill
    extractor = CaseBookResidualExtractor(
        session=session,
        output_dir="test_skill_filters"
    )
    
    skill_id = extractor.extract_skill(
        casebook_name="test_math_cases",
        model_before=model_before,
        model_after=model_after,
        tokenizer=tokenizer,
        domain="math",
        description="Test math skill extraction"
    )
    
    print(f"\n✅ Skill extraction test completed. Skill ID: {skill_id}")
    print("Check test_skill_filters directory for outputs")

if __name__ == "__main__":
    test_skill_extraction()