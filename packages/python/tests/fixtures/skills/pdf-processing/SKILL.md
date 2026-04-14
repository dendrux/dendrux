---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when handling PDFs.
license: Apache-2.0
metadata:
  author: test-org
  version: "1.0"
---

## Instructions

1. Read the PDF using the read_file tool
2. Parse the content and extract structured data
3. Format the output as Markdown tables

## Edge Cases

- If the PDF is image-only, inform the user that OCR is needed
