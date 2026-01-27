# cross-lingual-stability-judges
2026 EACL MME Workshop: Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from Finno-Ugric Languages

## Conversation Generator

Generate synthetic customer support conversations in Finno-Ugric languages (Estonian, Finnish, Hungarian) and English.

```bash
# Install
pip install -e .

# Set API key
cp .env.example .env  # then add your OPENAI_API_KEY

# Generate conversations
python -m conversation_generator -n 1 -l et           # 1 Estonian conversation
python -m conversation_generator -n 10 -o data.jsonl  # 10 random, save to file
python -m conversation_generator --help               # all options
```
