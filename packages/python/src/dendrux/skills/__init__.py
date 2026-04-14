"""Skills — instruction packages for Dendrux agents.

Adopts the Agent Skills open standard (agentskills.io). A skill is a
directory containing a SKILL.md file with YAML frontmatter and
Markdown instructions.

Usage:
    from dendrux.skills import Skill

    skill = Skill.from_dir("./skills/pdf-processing")
    skills = Skill.scan_dir("./skills")
"""

from dendrux.skills._loader import Skill

__all__ = ["Skill"]
