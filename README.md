# Interview Studio

A Streamlit mock-interview coach that routes interview practice, answer evaluation,
coaching, and reporting to separate LangChain agents. ChromaDB retrieves questions
and feedback for the selected company and role; evaluation scores correctness,
depth, and communication.

> **Data note:** `data/sample_interview_archive.csv` contains clearly labeled,
> illustrative practice examples. It is not a real institutional archive. Upload
> an authorised, de-identified archive to demonstrate authentic company patterns.

## Run locally

1. Use Python 3.11 or newer. Create and activate a virtual environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. If using the OpenAI API, copy the template and add your key to `.env` (not to
   `.env.example`):

   ```powershell
   Copy-Item .env.example .env
   ```

   Restart Streamlit after changing `.env`. The key is read from the environment
   and is never entered or displayed in the interface.

4. Start a local model with [Ollama](https://ollama.com/), then download the default:

   ```powershell
   ollama pull llama3.2
   ```

   Alternatively, set `OPENAI_API_KEY` in `.env`; the app selects the hosted
   service automatically when that key is present. To override this, set
   `AI_PROVIDER` to `OpenAI API` or `Ollama (local)` in `.env`. Model names and
   local service URL can also be configured there. An `.env.example` is included.

5. Run the app:

   ```powershell
   streamlit run app.py
   ```

ChromaDB uses a deterministic, local feature-hashing embedding for offline
similarity search; it does not download an embedding model. The vector database
is persisted under `data/chroma/` and excluded from version control.

## Upload an interview archive

Use the sidebar uploader to add a UTF-8 CSV with these required columns:

| Column | Description |
| --- | --- |
| `company` | Company name used for exact-match filtering |
| `role` | Role name used for exact-match filtering |
| `type` | `question` or `feedback` |
| `content` | Interview question or anonymised feedback text |
| `source` | Optional provenance label |
| `round` | Optional round label for round-specific question retrieval |

Rows are upserted by content and provenance, so uploading the same file does not
create duplicate records. The app does not extract data from uploaded documents;
convert authorised source material to this schema before upload. Remove candidate names and other personal data. When OpenAI is configured,
prompt context, answers, and retrieved snippets are sent to its API. Ollama
processes those requests locally.

## Assignment acceptance criteria

| Criterion | Where it is demonstrated |
| --- | --- |
| LLM integration | Ollama when no API key is configured, or OpenAI API when configured (`interview_coach/model.py`) |
| Interactive GUI | Streamlit interview room, scorecards, archive uploader, and report download (`app.py`) |
| Prompt engineering | System and user prompts are visible in **How it works**; zero-shot, persona, grounding, and schema constraints |
| LangChain core / LCEL | `ChatPromptTemplate` and `prompt \| model \| parser` chains (`interview_coach/agents.py`) |
| Agentic RAG | Intent/session-dependent retrieval from Chroma with exact-role then company-only matching (`interview_coach/archive.py`) |
| Intent detection / routing | Transparent request router (`interview_coach/routing.py`) |
| Vector database | Persistent ChromaDB with local embeddings and CSV ingestion (`interview_coach/archive.py`) |

The evaluator uses validated structured scores rather than displaying model output
as authoritative fact. This is practice feedback, not a hiring decision.
When an unused question matches the selected role and round, the interviewer
asks it verbatim; the LLM generates a role- and round-relevant question when no
matching archive question is available. Built-in role packs are illustrative,
not institutional history.

## Structured interview rounds

The mock interview has six rounds with two questions each: aptitude and analytical
reasoning, behavioral and ownership, role fundamentals, hands-on problem solving,
case study and design, and HR, communication, and culture. The evaluator scores
the two answers together once at the end of each round. The built-in role packs
cover Data Analyst, Software Engineer, AI/ML Engineer, DevOps Engineer,
Mechanical Engineer, and Product Manager. Company names remain editable; each
role pack is indexed under the selected company and role, while uploaded archive
records can provide authentic company-specific questions and feedback.

## Quick tests

```powershell
python -m unittest discover -s tests -v
```
