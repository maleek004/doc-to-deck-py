"""Everything that talks to an LLM: the provider-neutral contract, the fake, and (from 2.2) Gemini and Claude.

Kept free of re-exports on purpose: import from the submodules (`doctodeck.llm.contracts`, ...), so the
fake can import the pipeline's prompts without a circular import through this package.
"""
