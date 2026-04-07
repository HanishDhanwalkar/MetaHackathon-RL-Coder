from openenv.core.env_server import create_web_interface_app
from .models import CodeAction, CodeObservation

class CodeAutocompleteEnv:
    def __init__(self):
        self.code = ""
        
    def reset(self, code_context: str = "") -> CodeObservation:
        self.code = code_context
        return CodeObservation(
            code_context=self.code,
            kg_context=["Sovereign Project Context"],
            cursor_position=len(self.code)
        )

    def step(self, action: CodeAction):
        # Apply completion
        new_text = action.completion
        self.code += new_text
        
        # RL Reward Calculation
        reward = 0.0
        try:
            # Check if the code is syntactically valid
            compile(self.code, '<string>', 'exec')
            reward += 1.0
            # Additional reward for project relevance
            if "import" in new_text: reward += 0.5
        except SyntaxError:
            reward -= 1.0 # Penalty for broken code
            
        return self.reset(self.code), reward, True, {}

# Compliant with 'openenv validate'
app = create_web_interface_app(CodeAutocompleteEnv, CodeAction, CodeObservation)