# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a single task using a minimal setup."""

import csv
import json
import os
import random
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Type

from absl import app, flags, logging
from android_world import registry
from android_world.agents import infer, t3a
from android_world.env import env_launcher, json_action
from android_world.task_evals import task_eval
from openai import OpenAI

# Set logging and environment verbosity
logging.set_verbosity(logging.WARNING)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'none'


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('/opt/android/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found in common Android SDK paths. Please install Android '
        'SDK and ensure adb is in one of the expected directories. If already '
        'installed, set the path explicitly.'
    )


# Define command-line flags
_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.'
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Perform emulator setup once before using Android World.'
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'Console port of the Android device. Usually 5554, 5556, etc.'
)
_TASK = flags.DEFINE_string(
    'task',
    None,
    'Specific task to run.'
)

# Constants
VALID_ACTION_TYPES = {
    "click", "double_tap", "scroll", "swipe", "input_text",
    "navigate_home", "navigate_back", "keyboard_enter", "open_app",
    "status", "wait", "long_press", "answer", "unknown"
}

example = """--- Example ---

Goal: Send a message to Alice on WhatsApp  
Observation:  
- App: Home  
- UI Elements: ["WhatsApp", "Gmail", "Camera", "Chrome"]

Step 1:  
Action: {"action_type": "open_app", "app_name": "WhatsApp"}

Observation:  
- App: WhatsApp  
- UI Elements: ["Chats", "Alice", "Bob", "New Chat"]

Step 2:  
Action: {"action_type": "click", "index": 1}

Observation:  
- App: WhatsApp Chat (Alice)  
- UI Elements: ["Type a message", "Attachment", "Send"]

Step 3:  
Action: {"action_type": "input_text", "text": "Hi Alice, I'm running late!"}

Observation:  
- App: WhatsApp Chat (Alice)  
- UI Elements: ["Send", "Back", "Mic"]

Step 4:  
Action: {"action_type": "click", "index": 0}

Step 5:  
Action: {"action_type": "status", "goal_status": "complete"}

--- End Example ---

Valid action types:
["click", "double_tap", "scroll", "swipe", "input_text",
 "navigate_home", "navigate_back", "keyboard_enter", "open_app",
 "status", "wait", "long_press", "answer", "unknown"]

Formatting Rules:
- Always return **both** reasoning and action.
- You may use "open_app" for well-known apps (e.g., "Clock", "Gmail") even if not shown in the current UI.
- Do not default to searching unless necessary.
- Output exactly two lines:
    Reason: <why you chose the action>  
    Action: <valid JSON object>
- For "click", use: {"action_type": "click", "index": <int>}
- For "open_app", use: {"action_type": "open_app", "app_name": "<label>"}
- For "input_text", use: {"action_type": "input_text", "text": "<your text>"} 
- and so on
- For "status", use: {"action_type": "status", "goal_status": "complete"}
- Do NOT invent new fields or actions."""

def build_prompt(goal: str, app: str, ui_elements: list[str], history) -> str:
    return f"""{example}

Now complete the task below:

Goal: {goal}  
Observation:

- UI Elements: {ui_elements}
- History: {chr(10).join(history[-5:])}

Respond with:
Reason: <your reasoning>
Action: <valid JSON action>
"""


def extract_visible_elements(ui_elements):
    visible = []
    for el in ui_elements:
        label = el.text or el.content_description
        if label and el.is_visible:
            visible.append(label)
    return visible

def call_llm(prompt: str) -> tuple[str, dict]:
    """Returns (reason, action_dict)"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content

    # Extract Reason and Action
    reason_match = re.search(r"Reason:\s*(.*?)\s*Action:", content, re.DOTALL)
    action_match = re.search(r"Action:\s*(\{.*\})", content, re.DOTALL)

    reason = reason_match.group(1).strip() if reason_match else "N/A"
    action_raw = action_match.group(1).strip() if action_match else "{}"

    try:
        action_dict = json.loads(action_raw)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse LLM action:", action_raw)
        raise e

    return reason, action_dict


import json

def parse_llm_response(raw_text: str) -> dict:
    try:
        if "Action:" in raw_text:
            raw_text = raw_text.split("Action:", 1)[1].strip()

        cleaned = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed. Raw response:", raw_text)
        raise e

def is_valid_action(action: dict) -> bool:
    return action.get("action_type") in VALID_ACTION_TYPES

def extract_action_dict(text: str) -> dict:
    """
    Extract JSON dict from a line like: Action: {...}
    """
    try:
        match = re.search(r'Action:\s*(\{.*\})', text)
        if not match:
            raise ValueError("No valid Action JSON found in T3A response.")
        json_str = match.group(1)
        return json.loads(json_str)
    except Exception as e:
        print("❌ Failed to extract action from text:", text)
        raise e

def _main() -> None:
    Path("results").mkdir(exist_ok=True)
    log_path = "results/task2_episode_logs.jsonl"
    summary_path = "results/task2_summary.csv"

    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    env.reset(go_home=True)

    # Load task
    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if _TASK.value:
        if _TASK.value not in aw_registry:
            raise ValueError(f"Task {_TASK.value} not found in registry.")
        task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
    else:
        task_type = random.choice(list(aw_registry.values()))

    task = task_type(task_type.generate_random_params())
    task.initialize_task(env)

    # Initialize agents
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4o-2024-08-06'))
    state = env.reset(go_home=True)

    # Metrics
    history = []
    step_matches = 0
    total_steps = 0
    success = False
    step_matches = 0
    total_steps = 0
    step_accuracy = step_matches / total_steps if total_steps > 0 else 0

    # === Clear & initialize log files ===
    # Only write header if file doesn't exist
    write_header = not os.path.exists(summary_path)

    # Append to JSONL log file
    log_file = open(log_path, "a")

    # Append to CSV summary file
    summary_file = open(summary_path, "a", newline="")
    writer = csv.writer(summary_file)

    if write_header:
        writer.writerow(["timestamp", "goal", "llm_steps", "match_count", "step_accuracy", "llm_success", "agent_success"])
    with open(log_path, "a") as writer:
      llm_steps = 0
      agent_success = False
      llm_success = False

      state = env.reset(go_home=True)
      history = []
      print(f"🎯 Goal: {task.goal}")
      for step_num in range(10):
          print(f"\n--- Step {step_num + 1} ---")
          ui_elements = extract_visible_elements(state.ui_elements)
          print("📱 Observation (visible UI elements):", ui_elements)
          prompt = build_prompt(task.goal, app="Unknown", ui_elements=ui_elements, history=history)

          try:
              # LLM
              llm_reason, llm_action_dict = call_llm(prompt)
              print("🤖 LLM Reason:", llm_reason)
              print("🤖 LLM Action:", llm_action_dict)

              if not is_valid_action(llm_action_dict):
                  print("❌ Invalid LLM action:", llm_action_dict)
                  continue

              llm_steps += 1

              # Apply LLM Action
              env.execute_action(json_action.JSONAction(**llm_action_dict))
              history.append(f"Step {step_num + 1}: UI={ui_elements}, Action={llm_action_dict}")
              state = env.get_state()

              # Log LLM info + agent step
              t3a_response = agent.step(task.goal)
              t3a_reason = t3a_response.data.get("thought", "N/A")
              t3a_action = t3a_response.data.get("action_output", {})
              print("🧠 T3A Reasoning:", t3a_reason)
              print("🧠 T3A Action:", t3a_action)
              
              match = llm_action_dict == t3a_action
              step_matches += int(match)
              total_steps += 1

              log_record = {
                    "step": step_num + 1,
                    "goal": task.goal,
                    "observation": {
                        "app": "Unknown",
                        "ui_elements": ui_elements
                    },
                    "llm_action": llm_action_dict,
                    "llm_reasoning": llm_reason,
                    "agent_action": t3a_action,
                    "agent_reasoning": t3a_reason,
                    "match": match
                }

              writer.write(json.dumps(log_record, indent=2) + "\n")

              # Check LLM task success
              if task.is_successful(env) == 1:
                  print("🎯 LLM completed the task successfully!")
                  llm_success = True
                  break

              # Check if agent finishes it *after* each step (as it’s not synced with LLM)
              if t3a_response.done and task.is_successful(env) == 1:
                  agent_success = True

          except Exception as e:
              print("❌ Error during step:", e)
              break
    env.close()
    # Write summary
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            task.goal,
            llm_steps,
            f"{step_matches}/{total_steps}",
            f"{step_matches / total_steps:.2f}" if total_steps > 0 else "0.00",
            "Yes" if llm_success else "No",
            "Yes" if agent_success else "No"
        ])

def main(argv: Sequence[str]) -> None:
    del argv
    _main()

if __name__ == "__main__":
    app.run(main)