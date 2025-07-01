class PromptBuilder:
    """Helper class for systematically building prompts for In-Context Learning"""
    
    def __init__(self, initial_system_message: str = "You are a helpful assistant."):
        self.sections = []
        self.system_message = initial_system_message
    
    def add_section(self, title: str, content: str, new_line: bool = True):
        """Add a labeled section to the prompt"""
        separator = "\n" if new_line else " "
        self.sections.append(f"{title}:{separator}{content}")
    
    def add_instruction(self, instruction: str):
        """Add task instructions"""
        self.add_section("Instructions", instruction)
    
    def add_example(self, input_text: str, output_text: str, example_type: str = "simple_io"):
        """Add a basic input-output example"""
        if example_type == "simple_io":
            self.add_section("Example", f"Input: {input_text}\nOutput: {output_text}")
        elif example_type == "cot":  # Chain-of-Thought
            self.add_section("Example", f"Input: {input_text}\nThought: {output_text.split('Output: ')[0]}\nOutput: {output_text.split('Output: ')[1]}")
    
    def add_cot_example(self, input_text: str, thought_process: str, output_text: str):
        """Add a Chain-of-Thought example"""
        self.add_section("Example", f"Input: {input_text}\nThought: {thought_process}\nOutput: {output_text}")
    
    def add_task_description(self, description: str, current_input: str):
        """Add task-specific description and input"""
        self.add_section("Task Description", description)
        self.add_section("Current Input", current_input)
    
    def add_constraints(self, constraints: list):
        """Add constraints to the task"""
        self.add_section("Constraints", "\n".join([f"- {c}" for c in constraints]))
    
    def build(self) -> str:
        """Build the final prompt string"""
        return f"{self.system_message}\n\n" + "\n\n".join(self.sections)
    
    def reset(self):
        """Reset the prompt builder"""
        self.sections = []