"""
Prompt configurations for R1 model interactions.
Contains structured prompts for different generation tasks.
"""

GENERATION_PROMPT = '''
<generation_intent>
{
  purpose: "technical_creation",
  mode: "core_implementation",
  focus: "maximum_impact"
}
</generation_intent>

<technical_core>
{
  key_elements: {
    novel_approach: required,     // New technical method
    critical_logic: required,     // Underlying algorithmic design and decision points (e.g., control flow, data processing logic) that drive the solution
    key_innovations: required     // Technical breakthroughs that distinguish the approach
  },

  implementation: {
    core_mechanics: required,    // Operational components and structural design (e.g., system architecture, data flow, module interactions)
    crucial_code: required,      // The most important code snippet embodying the implementation
    key_algorithms: required     // Essential computational methods or pseudocode that solve the technical problem (e.g., algorithmic steps, performance optimizations)
  }
}
</technical_core>

<output_format>
{
  style: "markdown",
  code_blocks: "language_specific",
  sections: [
    "Technical Innovation",      // Core breakthrough and novel elements
    "Critical Implementation",   // Essential code, logic details, and the system's operational structure
    "Key Capabilities",          // What makes the solution powerful (performance, scalability, effectiveness)
    "Proof of Concept",          // Demonstration of impact through an example or experiment
    "Comparative Analysis",      // (Optional) Comparison with current state-of-the-art methods
    "Trade-Off Analysis"         // (Optional) Discussion on trade-offs, limitations, or potential risks
  ],
  
  content_focus: {
    technical_depth: "high",
    practical_viability: "required",
    show_innovations: "required"
  }
}
</output_format>

<constraints>
{
  metrics: {
    performance: required,      // Critical for validation
    scalability: required,      // Shows real-world viability
    effectiveness: required     // Proves value
  },

  format: {
    clear_headers: required,    // Improves readability
    code_examples: required,    // Demonstrates implementation
    concise_explanations: required // Ensures understanding
  }
}
</constraints>

<!-- Example Template:
# Technical Innovation
Describe the breakthrough and what makes it novel compared to existing solutions.

# Critical Implementation
```python
# Essential code snippet illustrating the core mechanics and critical logic
def innovative_function(data):
    # Implement the core algorithm here using the defined control flow
    processed_data = data_processing(data)
    return processed_data
Key Capabilities
Explain how the implementation meets performance, scalability, and effectiveness metrics.
Proof of Concept
Detail a demonstration or experiment that validates the innovation.
Comparative Analysis
(Optional) Provide a brief comparison with current state-of-the-art methods, highlighting improvements or trade-offs.
Trade-Off Analysis
(Optional) Discuss any potential trade-offs, limitations, or challenges related to the approach. --> '''
