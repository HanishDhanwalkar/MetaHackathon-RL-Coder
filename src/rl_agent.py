from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class RLCompletionAgent:
    """
    Policy Wrapper that tracks returns from code env (RL feedback)
    and adapts prompts / generation options so later suggestions improve online
    """
    
    _ema_reward: float = 0.45
    alpha: float = 0.22
    reward_hist: List[float] = field(default_factory=list)
    max_hist: int = 48
    
    def observe_reward(self, reward: float | None) -> None:
        r = float(reward) if reward is not None else 0.0
        self.reward_hist.append(r)
        if len(self.reward_hist) > self.max_hist:
            self.reward_hist = self.reward_hist[-self.max_hist:]
            
        self._ema_reward = self.alpha * r + (1 - self.alpha) * self._ema_reward

    # Keep old misspelled name as alias for backward compatibility
    def obserce_reward(self, reward: float | None) -> None:
        return self.observe_reward(reward)

    def ema_reward(self) -> float:
        return self._ema_reward
        
    def trend(self) -> float:
        if len(self.reward_hist) < 6:
            return 0.0
        recent = sum(self.reward_hist[-3:]) / 3.0
        older = sum(self.reward_hist[-6:-3]) / 3.0
        return recent - older
    
    def sys_msg(self) -> str:
        base = (
            "You are the policy network for a RL-Based Python IDE. "
            "Emit ONLY raw code to insert at cursor-no markdown,explanations,etc."
        )
        
        if self._ema_reward > 0.70 and self.trend() > 0.0:
            return f"{base}  High return: keep ghost suggestions short (1 line when possible)."
        if self._ema_reward < 0.42 and self.trend() < -0.08:
            return f"{base}  Low return: favor minimal edits that fix syntax; avoid hallucinations of new APIs."
        return base + " Balance brevity with correctness; match local indentations"
    
    # def ollama_options(self) -> Dict[str, Any]:  # Commented out: renamed to openai_options
    def openai_options(self) -> Dict[str, Any]:
        """As EMA rises, cool sampling -> more deterministic (exploit good policy region)."""
        temp = max(0.12, min(0.92, 0.88 - 0.55 * self._ema_reward))
        top_p = max(0.35, min(0.95, 0.55 + 0.45 * self._ema_reward))
        return {"temperature": temp, "top_p": top_p}