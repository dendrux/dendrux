---
name: summarize-file
description: Read a file and produce a concise summary. Use when the user asks to summarize, review, or explain the contents of a file.
metadata:
  author: dendrux-examples
  version: "2.0"
---

## Instructions

1. Use the `filesystem__read_text_file` tool to read the file contents
2. Read `references/style-guide.md` with `read_skill_resource` for the required format
3. Analyze the content and identify key themes, topics, or data points
4. Draft the summary following the style guide
5. Check the draft with `scripts/word_count.py`, passing the draft text as the only argument,
   using `run_skill_script`. If it reports more than 200 words, shorten it and check again

## Guidelines

- If the file is very short (under 50 words), note that and provide the full content instead of summarizing
- If the file contains structured data (CSV, JSON), describe the structure and key fields
- Always mention the filename in your response
