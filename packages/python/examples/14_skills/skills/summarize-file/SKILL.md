---
name: summarize-file
description: Read a file and produce a concise summary. Use when the user asks to summarize, review, or explain the contents of a file.
metadata:
  author: dendrux-examples
  version: "1.0"
---

## Instructions

1. Use the `filesystem__read_text_file` tool to read the file contents
2. Analyze the content and identify key themes, topics, or data points
3. Produce a concise summary with:
   - A one-sentence overview
   - 3-5 bullet points covering the main points
   - A brief conclusion or takeaway
4. Keep the summary under 200 words

## Guidelines

- If the file is very short (under 50 words), note that and provide the full content instead of summarizing
- If the file contains structured data (CSV, JSON), describe the structure and key fields
- Always mention the filename in your response
