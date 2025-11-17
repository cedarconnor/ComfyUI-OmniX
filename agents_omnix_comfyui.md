# agents.md – OmniX + FLUX Pano Perception for ComfyUI

This file defines a small “agent stack” for developing, maintaining, and extending the **ComfyUI_OmniX** node pack.

Each *agent* can be a person, a future code-assistant, or simply a “role” you step into when working on the project.

---

## Agent 1 – `design_agent`

**Mission**  
Owns the high-level **architecture** of the OmniX + FLUX integration in ComfyUI.

**Responsibilities**  

- Maintain the **design document** (`design_doc.md`) and keep it in sync with reality.
- Decide:
  - Which nodes exist,
  - Their inputs/outputs,
  - How they relate (W1/W2/W3 workflows).
- Ensure the design stays:
  - Modular,
  - Easy to reason about,
  - Compatible with upstream OmniX changes.

**Inputs**  

- Papers, OmniX repo updates, issues from users.
- Feedback from `node_dev_agent` and `workflow_agent`.

**Outputs**  

- Updated `design_doc.md`.
- Diagrams and example graphs describing the pipeline.
- Clear TODO/roadmap for new features and breaking changes.

**Success Criteria**  

- Minimal churn in node inputs/outputs over time.
- New features land without breaking existing workflows.
- Other agents can implement or maintain code using the design doc alone.

---

## Agent 2 – `conversion_agent`

**Mission**  
Handle **LoRA format conversion** and model asset preparation.

**Responsibilities**  

- Convert OmniX LoRAs from Diffusers/PEFT format → ComfyUI FLUX LoRA format.
- Maintain a small toolkit (scripts) for:
  - Validating state dicts (shapes, key naming).
  - Diffing converted LoRAs vs. original ones in simple test passes.
- Keep track of:
  - Which base model each LoRA is trained on (e.g. FLUX.1-dev).
  - Where they live in the Comfy directory tree.

**Inputs**  

- Original OmniX LoRAs (Diffusers format).
- Conversion requirements from `design_agent` / `node_dev_agent`.

**Outputs**  

- `models/loras/OmniX_*_comfyui.safetensors` ready for Comfy.
- `conversion` scripts (e.g., `convert_omnix_loras.py`) and mini-README.

**Success Criteria**  

- Converted LoRAs load cleanly in Comfy (no missing/mismatched keys).
- Perception outputs match OmniX reference scripts within expected tolerance.
- LoRA filenames and metadata are consistent and self-explanatory.

---

## Agent 3 – `node_dev_agent`

**Mission**  
Implement and maintain the **ComfyUI custom nodes** for OmniX perception and helpers.

**Responsibilities**  

- Implement the node pack:
  - `omni_x_nodes.py`
  - `omni_x_utils.py`
  - `__init__.py` with `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
- Port the OmniX perception logic:
  - From `run_pano_perception.py` / `run_pano_all.py` and associated utilities,
  - Into a form that works with Comfy’s tensor formats.
- Ensure nodes obey the **contracts** specified in the design doc:
  - Correct inputs/outputs,
  - Stable names,
  - Predictable behavior.

**Inputs**  

- `design_doc.md` (authoritative spec for nodes).
- LoRA files from `conversion_agent`.
- OmniX repo source code.

**Outputs**  

- Fully functioning node pack (`ComfyUI_OmniX`).
- Clean, well-commented Python code.
- Unit tests or at least regression tests for key behaviors (depth/normal/PBR/semantic).

**Success Criteria**  

- Nodes show up correctly in Comfy UI and are stable under normal use.
- Outputs are visually and numerically consistent with OmniX reference scripts.
- Code structure is modular and changes are easy to make.

---

## Agent 4 – `workflow_agent`

**Mission**  
Design and maintain **ComfyUI workflows** showcasing OmniX capabilities.

**Responsibilities**  

- Build example graphs for the three canonical workflows:
  - **W1:** Text → Pano
  - **W2:** Pano → Perception
  - **W3:** Text → Pano → Perception
- Save these graphs as shareable `.json` files under `examples/`.
- Tune sensible defaults:
  - Pano resolutions (e.g., `1024×2048`, `1152×2304`),
  - Sampler settings and step counts for FLUX,
  - LoRA strengths per task.
- Provide comments / annotations inside workflows.

**Inputs**  

- Node definitions from `node_dev_agent`.
- Design guidance from `design_agent`.
- User feedback and bug reports.

**Outputs**  

- `examples/omni_x_text_to_pano.json`
- `examples/omni_x_pano_perception.json`
- `examples/omni_x_full_pipeline.json`
- Small how-to notes (`README.md` snippets).

**Success Criteria**  

- A Comfy user can drag in an example workflow, plug in their prompt or pano, and get valid results **without reading code**.
- Workflows remain functional even as nodes evolve (within reason).
- Examples demonstrate best practices, not just minimal functionality.

---

## Agent 5 – `docs_agent`

**Mission**  
Own the **documentation and onboarding experience**.

**Responsibilities**  

- Maintain:
  - `README.md` for the node pack (install, usage, known issues).
  - Inline docstrings for each node class and utility function.
  - Short “How to use OmniX in Comfy” tutorials.
- Ensure the docs reflect the current state of the code and workflows.

**Inputs**  

- Codebase and updates from `node_dev_agent`.
- Example workflows and UX insights from `workflow_agent`.
- Roadmap and decisions from `design_agent`.

**Outputs**  

- Clear, up-to-date docs and changelog.
- Minimal quickstart instructions (e.g., “5 steps to get your first pano + depth map”).

**Success Criteria**  

- New users can install and run a basic pipeline in under ~10 minutes.
- GitHub issues about “how do I…” decrease over time, or are answered by linking to existing docs.

---

## Agent 6 – `eval_agent`

**Mission**  
Measure and safeguard **quality** and **performance**.

**Responsibilities**  

- Create a small **test suite**:
  - Given a fixed prompt, seed, and resolution, compare depth/normal/PBR/semantic maps between:
    - OmniX reference scripts, and
    - ComfyUI nodes.
- Monitor:
  - GPU memory usage,
  - Latency for typical resolutions,
  - Numerical differences between versions.
- Provide regression protection when changing code or models.

**Inputs**  

- Test prompts/panos.
- Node pack + workflows.
- Ground truth outputs (from OmniX scripts).

**Outputs**  

- Test scripts / notebooks under `tests/`.
- Periodic reports or notes on performance changes.

**Success Criteria**  

- Changes to the code don’t silently degrade quality.
- Performance remains acceptable or improves over time.
- Deviations vs. OmniX reference stay within documented tolerance.

---

## Agent 7 – `integration_agent` (optional future role)

**Mission**  
Explore **downstream integrations**, such as 3D reconstruction and other pipelines.

**Responsibilities**  

- Integrate OmniX outputs with:
  - Gaussian splatting pipelines,
  - Mesh reconstruction,
  - Game engines / DCC tools (Unreal, Unity, Blender).
- Design new nodes or workflows that consume depth/normal/PBR/semantic maps.

**Inputs**  

- Outputs from `OmniX_PanoPerception_*` nodes.
- External tools/libs for 3D and rendering.

**Outputs**  

- New node designs and prototypes for 3D reconstruction / relighting.
- Example workflows that go “beyond” the base OmniX paper (e.g., full 3D scenes from a single pano).

**Success Criteria**  

- New high-value workflows built on top of OmniX (e.g., immersive environments, relit VR domes).
- Reusable patterns that other practitioners can adapt.

---

## How to Use This File

When you sit down to work on the project, pick a **role**:

- “Today I’m the `node_dev_agent`” → focus on implementing nodes exactly to spec.
- “Today I’m the `workflow_agent`” → focus on making beautiful, robust Comfy graphs.
- “Today I’m the `eval_agent`” → focus on comparisons, metrics, and regression detection.

You can be all of these agents at different times—but maintaining the separation helps keep tasks clear and the project cohesive.
