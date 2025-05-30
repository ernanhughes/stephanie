INSERT INTO symbolic_rules (
    target, 
    target_name, 
    filter, 
    attributes, 
    context_hash, 
    rule_text, 
    created_at
) VALUES (
    'prompt',
    'generate_cot.txt',
    '{"goal_type": "theoretical", "goal_category": "ai_research"}',
    '{"prompt_file": "generate_cot_theoretical.txt", "model.name": "ollama/phi3"}',
    'e3b0c44298fc1c149afbf4c8996fb924', -- placeholder: you should compute this properly
    'If goal is theoretical and category is ai_research, use cot_theoretical prompt',
    CURRENT_TIMESTAMP
);
