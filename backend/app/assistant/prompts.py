"""System prompts for AETHER's AI assistant."""

TRUTHFULNESS_SYSTEM_PROMPT = """You are the AETHER Studio assistant — a physics expert.

CRITICAL RULES:
1. NEVER fake physics. If you don't know, say so.
2. EVERY time you mention a physical result (force, stress, stability, failure), you MUST include:
   - Confidence level: high / medium / low
   - Basis: "measured from video" | "classical simulator" | "learned model" | "heuristic estimate" | "LLM reasoning only"
   - Key assumptions made
3. You can explain, translate questions into simulation requests, and suggest parameters.
4. You CANNOT compute final physics results yourself — use the simulation tools.
5. Be honest about uncertainty from video analysis.

Your job is to:
- Explain what the user is seeing in their mechanism
- Translate "what if" questions into precise simulation parameters
- Explain simulation results in simple language
- Flag when results come from video measurement vs. simulation vs. estimation
"""


def build_scene_explanation_prompt(scene_graph) -> str:
    summary = scene_graph.summary()
    prompt = f"""Explain this mechanism to a maker or engineering student:

Scene: {summary['object_count']} objects, {summary['edge_count']} connections.
Confidence: {summary['confidence']:.0%}

Objects:
"""
    for obj in scene_graph.objects:
        prompt += f"  - {obj.label} ({obj.object_type})\n"
    prompt += "\nConnections:\n"
    for edge in scene_graph.edges:
        prompt += f"  - {edge.source_id} --[{edge.joint_type.value}]--> {edge.target_id}\n"
    prompt += "\nKeep explanation simple and focused on what matters for what-if questions."
    return prompt


def build_simulation_explanation_prompt(question: str, scene_graph, simulation_result: dict) -> str:
    change = simulation_result.get("change_from_baseline", {})
    prompt = f"""A user asked: "{question}"

Simulation results (classical simulator):
- Vibration frequency: {simulation_result.get('vibration_freq_Hz', 'N/A')} Hz
- Vibration amplitude: {simulation_result.get('vibration_amplitude_mm', 'N/A')} mm
- Trajectory error: {simulation_result.get('trajectory_error_mm', 'N/A')} mm
"""
    if change:
        pct_vib = change.get("vibration_amplitude_change_pct")
        tension_change = change.get("tension_change_N", 0)
        if pct_vib is not None:
            direction = "increased" if pct_vib > 0 else "decreased"
            prompt += f"\nCompared to baseline:\n- Vibration amplitude {direction} by {abs(pct_vib):.0f}%\n"
        if tension_change:
            direction = "increased" if tension_change > 0 else "decreased"
            prompt += f"- Belt tension {direction} by {abs(tension_change):.1f} N\n"
    prompt += f"\nAssumptions: {', '.join(simulation_result.get('assumptions', [])[:3])}\n"
    prompt += f"Confidence: {simulation_result.get('confidence', 0):.0%} ({simulation_result.get('confidence_basis', 'unknown')})\n"
    prompt += "\nExplain this result in simple, honest terms."
    return prompt
