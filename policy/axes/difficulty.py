class DifficultyAxis:

    name = "difficulty"

    def __init__(self, difficulty_computer):
        self.difficulty_computer = difficulty_computer

    def compute(self, context):
        return self.difficulty_computer.compute(context)
