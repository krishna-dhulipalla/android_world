import os
import re
from typing import List, Tuple
from android_world.env import json_action
from openai import OpenAI


VALID_ACTION_TYPES = {
    "click", "double_tap", "scroll", "swipe", "input_text",
    "navigate_home", "navigate_back", "keyboard_enter", "open_app",
    "status", "wait", "long_press", "answer", "unknown"
}


class LLMAgent:
    def __init__(self, env, goal: str, model: str = "gpt-4o-2024-08-06"):
        self.env = env
        self.goal = goal
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.history = []
        self.state = env.reset(go_home=True)

    def extract_visible_elements(self, ui_elements) -> List[str]:
        visible = []
        for el in ui_elements:
            label = el.text or el.content_description
            if label and el.is_visible:
                visible.append(label)
        return visible

    def build_prompt(self, app: str, ui_elements: List[str]) -> str:
        few_shot = """--- Example ---

Goal: Open the Gmail app  
Observation:  
- App: Home  
- UI Elements: ["Clock", "Gmail", "YouTube", "Camera"]

Step 1:  
Action: {"action_type": "open_app", "app_name": "Gmail"}

Observation:  
- App: Gmail  
- UI Elements: ["Inbox", "Compose", "Search"]

Step 2:  
Action: {"action_type": "status", "goal_status": "complete"}

--- End Example ---
"""
        history_str = "\n".join(self.history[-5:])
        return f"""{few_shot}

Now complete the task below:

Goal: {self.goal}  
Observation:
- App: {app}
- UI Elements: {ui_elements}
- History: {history_str}

What is the next best action?  
Respond in JSON format like:
{{"action_type": "click", "index": 3}} or {{"action_type": "open_app", "app_name": "Clock"}}  
or  
{{"action_type": "status", "goal_status": "complete"}}
"""

    def call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    def parse_llm_response(self, raw_text: str) -> dict:
        cleaned = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        try:
            return eval(cleaned)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def is_valid_action(self, action: dict) -> bool:
        return action.get("action_type") in VALID_ACTION_TYPES

    def step(self, step_num: int) -> Tuple[bool, str]:
        ui_elements = self.state.ui_elements
        visible_ui = self.extract_visible_elements(ui_elements)
        prompt = self.build_prompt(app="Unknown", ui_elements=visible_ui)
        print(f"\n--- Step {step_num} ---")
        print("Observation:", visible_ui)

        llm_raw = self.call_llm(prompt)
        llm_action_dict = self.parse_llm_response(llm_raw)

        if not self.is_valid_action(llm_action_dict):
            print("❌ Invalid LLM Action:", llm_action_dict)
            return False, "invalid_action"

        try:
            action_obj = json_action.JSONAction(**llm_action_dict)
        except Exception as e:
            print("❌ Failed to parse into JSONAction:", e)
            return False, "json_action_error"

        print("🤖 LLM-Predicted Action:", llm_action_dict)

        try:
            self.env.execute_action(action_obj)
            history_line = f"Step {step_num}: UI={visible_ui}, Action={llm_action_dict}"
            self.history.append(history_line)
            self.state = self.env.get_state()

            if self.state is None:
                return False, "state_error"

            return True, "success"
        except Exception as e:
            print("❌ Failed to execute action in env:", e)
            return False, "execution_error"

    def check_success(self, task) -> bool:
        return task.is_successful(self.env) == 1