"""Skills — instruction packages for Dendrux agents.

Adopts the Agent Skills open standard (agentskills.io). A skill is a name,
a description, Markdown instructions, and optional files (text, binary,
or scripts under ``scripts/``).

Usage:
    from dendrux.skills import Skill

    skill = Skill(name="triage", description="Triage tickets.", body="...")
    skill = Skill.from_markdown(text, files={"references/policy.md": "..."})
    skill = Skill.from_dir("./skills/pdf-processing")
    skills = Skill.scan_dir("./skills")
"""

from dendrux.skills._model import Skill

__all__ = ["Skill"]
