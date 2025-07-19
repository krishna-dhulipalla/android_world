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

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""

from collections.abc import Sequence
import os
import random
from typing import Type

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.task_evals import task_eval
from android_world.env import json_action

import openai
from openai import OpenAI

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


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
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run.',
)


def _main() -> None:
  """Runs a single task."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  env.reset(go_home=True)
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  if _TASK.value:
    if _TASK.value not in aw_registry:
      raise ValueError('Task {} not found in registry.'.format(_TASK.value))
    task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
  else:
    task_type: Type[task_eval.TaskEval] = random.choice(
        list(aw_registry.values())
    )
  params = task_type.generate_random_params()
  task = task_type(params)
  task.initialize_task(env)
  agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))

  #print('Goal: ' + str(task.goal))
  is_done = False
  # for _ in range(int(task.complexity * 10)):
  #   response = agent.step(task.goal)
  #   if response.done:
  #     is_done = True
  #     break
  def extract_visible_elements(ui_elements):
    visible = []
    for el in ui_elements:
        label = el.text or el.content_description
        if label and el.is_visible:
            visible.append(label)
    return visible
  
  def build_prompt(goal: str, app: str, ui_elements: list[str], history) -> str:
    example = """--- Example ---

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

    prompt = f"""{example}

    Now complete the task below:

    Goal: {goal}  
    Observation:
    - App: {app}
    - UI Elements: {ui_elements}
    - History: {chr(10).join(history[-5:])}  

    What is the next best action?  
    Respond in JSON format like:
    {{"action_type": "click", "index": 3}} or {{"action_type": "open_app", "app_name": "Clock"}}  
    or  
    {{"action_type": "status", "goal_status": "complete"}}
    """
    return prompt
      
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

  def call_llm(prompt: str) -> dict:
      response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
      )
      return response.choices[0].message.content
    
  import re
  def parse_llm_response(raw_text: str) -> dict:
    cleaned = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return eval(cleaned)
  
  VALID_ACTION_TYPES = {
      "click", "double_tap", "scroll", "swipe", "input_text",
      "navigate_home", "navigate_back", "keyboard_enter", "open_app",
      "status", "wait", "long_press", "answer", "unknown"
  }

  def is_valid_action(action: dict) -> bool:
      if action.get("action_type") not in VALID_ACTION_TYPES:
          return False
      return True

  # Reset environment to start the episode
  state = env.reset(go_home=True)
  history = []

  for step_num in range(10):
      print(f"\n--- Step {step_num + 1} ---")

      # 1. Extract current observation
      ui_elements = extract_visible_elements(state.ui_elements)
      prompt = build_prompt(task.goal, app="Unknown", ui_elements=ui_elements, history=history)

      # 2. Send to LLM
      print("observation:\n", ui_elements)
      llm_raw = call_llm(prompt)
      llm_action = parse_llm_response(llm_raw)
      if not is_valid_action(llm_action):
          print("❌ Invalid LLM action:", llm_action)
          continue

      try:
          action = json_action.JSONAction(**llm_action)
      except TypeError as e:
          print("❌ Invalid keys for JSONAction:", e)
          continue
      print("🤖 LLM-Predicted Action:", llm_action)
      
      # if llm_action.get("action_type") == "click":
      #   index = llm_action.get("index")
      #   if index is None or index >= len(state.ui_elements):
      #       print(f"❌ Invalid index: {index}, max allowed: {len(state.ui_elements) - 1}")
      #       continue

      # # 3. Call T3A agent and get its decision
      # t3a_response = agent.step(task.goal)
      # t3a_action = t3a_response.data.get("action_output")
      # print("📌 T3A Agent Action:", t3a_action)

      # 4. Step the environment using GPT action (not T3A's)
      try:
          # Execute in the environment
          env.execute_action(action)
          history_summary = f"Step {step_num + 1}: UI={ui_elements}, Action={llm_action}"
          history.append(history_summary)

          # Get updated state
          state = env.get_state()
          if task.is_successful(env) == 1:
            print("✅ Task complete according to task evaluator.")
            break
      except Exception as e:
          print("❌ Failed to send LLM action to env:", e)
          break

  # agent_successful = is_done and task.is_successful(env) == 1
  # print(
  #     f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
  #     f' {task.goal}'
  # )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)