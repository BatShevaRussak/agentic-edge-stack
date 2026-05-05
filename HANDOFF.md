# Project Handoff - Agentic Edge Stack

## הקשר כללי

זהו **פרויקט למשימת ראיון עבודה** לתפקיד **Machine Learning Systems Engineer**. שם המשימה: "The Agentic Edge Stack". המשתמשת קיבלה את המשימה ויש לה שבוע להגיש דרך GitHub repository פומבי.

**העבודה התבצעה ב-Cursor IDE על Windows.**

---

## דרישות המטלה (5 חלקים)

### Part 1: Model Serving & Deployment
- מודל: `gemma-3-1b-it` / `Llama-3.2-3B-Instruct` / מודל קל מקביל.
- Inference Engine: Ollama / llama.cpp / vLLM.
- Deliverables:
  - Deployment script (Shell או Docker Compose).
  - Verification script (Python או cURL) להדגמת "Hello World".

### Part 2: In-Memory RAG
- Dataset של 2-10 עמודי טקסט טכני.
- Embeddings: `all-MiniLM-L6-v2` או `bge-small-en-v1.5`.
- Vector Store: FAISS או ChromaDB (in-memory).
- Flow: query → embed → retrieve top 3 chunks → inject ל-prompt.

### Part 3: Agentic Orchestrator
- עטיפת ה-RAG ככלי (Tool) ש-Agent יוכל לקרוא לו.
- שימוש ב-LangChain / LangGraph / loop משלנו.
- Trace שמדגים שימוש בכלי.

### Part 4: API Serving & Streaming
- FastAPI עם endpoint `/chat`.
- Streaming אמת (Server-Sent Events).

### Part 5: Bonuses
1. Structured Output Responses (Pydantic + JSON schema).
2. Model Quantization & Performance Profiling (TPS + RAM).
3. Production-Grade Vector DB on K8s (Qdrant/Weaviate על k3s/minikube).

---

## פרופיל המשתמשת

- **שפה:** עברית (כל ההסברים בעברית).
- **רמה:** מבינה אבל לא מנוסה - צריכה הסברים מעמיקים על כל החלטה.
- **גישה:** רוצה לראות **כל קטע קוד לפני יצירה** ולקבל אישור (interview prep mode).
- **דורשת:** הסבר של כל בחירה ארכיטקטונית בהקשר של דרישות המטלה.
- **הלך רוח:** "כל הסבר תראה לי בהתאמה לדרישות מהמסמך" - ביקשה לעבור קובץ-קובץ ולהראות איך כל אחד עונה לדרישה.

---

## חומרת המחשב (קריטי!)

```
CPU:        Intel Core i7-6500U (2 cores, 4 threads, Gen 6, year 2015)
RAM:        16 GB
GPU:        Intel HD Graphics 520 (NO NVIDIA GPU available!)
Disk:       159 GB free on C:
OS:         Windows
Shell:      PowerShell
Python:     3.13.3 (already installed)
Internet:   SLOW & UNSTABLE (~104 KB/s - 1 MB/s, drops mid-download)
```

**משמעות:**
- אי אפשר להריץ vLLM (דורש NVIDIA CUDA).
- אי אפשר להריץ מודלים גדולים (3B+ ירוץ ב-1-2 tokens/sec).
- חייבים לתעדף **system usability** מעל גודל המודל.

---

## החלטות ארכיטקטוניות שאושרו

### 1. מודל: `llama3.2:1b` (לא gemma:2b ולא 3B)
**רציונל:**
- המטלה הזכירה במפורש את `Llama-3.2` family.
- בחרנו `1b` במקום `3B` בגלל אילוצי החומרה (CPU, no GPU).
- 3B היה נותן 1-2 tokens/sec (לא שמיש ל-Agent עם 2-3 קריאות לכל query = 3-6 דקות per query).
- 1B נותן 5-8 tokens/sec (interactive use).
- המטלה אומרת "or a comparable lightweight instructor model" - זה מאפשר את הבחירה.

### 2. Inference Engine: **Ollama** (לא llama.cpp ולא vLLM)
**רציונל:**
- vLLM דורש NVIDIA GPU - אין לה.
- Ollama פשוט ויציב, תומך ב-CPU בלבד.
- API תואם OpenAI - קל לחבר ל-LangChain בעתיד.

### 3. מבנה הפרויקט: **`app/` Layout** (לא `src/`)
**רציונל:**
- `src/` הוא לספריות שמתפרסמות ב-PyPI (כמו pydantic, requests).
- `app/` הוא לאפליקציות web (FastAPI standard).
- הפרויקט שלנו הוא אפליקציה (שרת FastAPI) ולא ספרייה.
- התיעוד הרשמי של FastAPI ממליץ על `app/`.

### 4. ניהול תלויות: **`pyproject.toml`** (לא requirements.txt)
**רציונל:**
- pyproject.toml הוא הסטנדרט המודרני (PEP 621, 2020+).
- מאפשר metadata + dependencies + tool configs במקום אחד.
- Production-grade.

### 5. ניהול הגדרות: **Pydantic Settings + `.env`**
**רציונל:**
- Type-safe config (אם הערך לא valid - שגיאה ברורה).
- Single source of truth.
- ניתן לשנות הגדרות בלי לשנות קוד.
- 12-Factor App principle.

### 6. Deployment: **PowerShell script** ראשון, **Docker Compose** בהמשך
**רציונל:**
- מתחילים פשוט - shell script.
- אחרי שהבסיס עובד - מוסיפים Docker Compose כשכבת תאימות נוספת.

### 7. מיקום הפרויקט: `C:\Users\user\Desktop\agentic-edge-stack`
**רציונל:** נוח לגישה, יש מקום פנוי על C: (159GB).

### 8. מודל Embedding: **`BAAI/bge-small-en-v1.5`** (לא `all-MiniLM-L6-v2`)
**רציונל:**
- שני המודלים מותרים במשימה כדוגמאות.
- שניהם 384-dim, ~150MB בזיכרון, מהירים על CPU.
- BGE-small מקבל ~62.2 על MTEB, MiniLM-L6-v2 מקבל ~56.3 - הפרש משמעותי באיכות retrieval.
- מראה judgment בראיון: "בחרתי את החזק יותר באותו class גודל".
- Pooling: BGE משתמש ב-**CLS token** (לא mean) - גילינו זאת מ-`1_Pooling/config.json` של המודל.

### 9. Vector Store: **FAISS `IndexFlatIP`** (לא ChromaDB, לא HNSW)
**רציונל:**
- N קטן (60 chunks) - brute-force exact עדיף על approximate.
- IVF/HNSW מתחילים להשתלם רק מ-10K+ וקטורים.
- וקטורים מנורמלים L2 → inner product = cosine similarity (טריק סטנדרטי, חוסך אינדקס cosine ייעודי).
- ChromaDB יותר high-level אבל מוסיף תלויות (sqlite, posthog) ופחות "מרשים" ב-context של ML Systems Engineer.

### 10. Chunking: **`RecursiveCharacterTextSplitter`** עם separators מודעים ל-Markdown
**רציונל:**
- עוצר קודם על `\n## ` (כותרות), אחר כך `\n### `, אחר כך פסקאות, אחר כך משפטים.
- שומר על קוהרנטיות סמנטית של כל chunk.
- 500 תווים / 50 חפיפה - חלון יציב למודל BGE (שמכבד עד 512 tokens).

### 11. Cache טרנספרנטי לאינדקס
**רציונל:**
- אינדקס FAISS + chunks נשמרים ל-`data/cache/`.
- Manifest עם SHA-256 של תוכן הקורפוס + שם המודל + dim.
- כל שינוי ב-`data/` או החלפת מודל → cache invalidation אוטומטי.
- תוצאה: ריצה ראשונה ~5s, ריצה חוזרת ~0.4s (פי 15).

### 12. רשת חסומה (NetFree) - פתרון Path B מקביל ל-Llama
**רציונל:**
- HuggingFace החדש משתמש ב-`transfer.xethub.hf.co` שחסום.
- גם direct download מ-`huggingface.co` עובר SSL handshake timeout.
- כתבנו `scripts/import_embed_model.ps1` שמוריד את 10 הקבצים ב-`Invoke-WebRequest`.
- במקרה החסום במיוחד - העברה ידנית ממחשב פתוח (כמו שעשינו עם Llama).
- ב-`.env` הגדרנו `EMBED_MODEL_NAME=./models/bge-small-en-v1.5` (נתיב מקומי) - SentenceTransformer טוען מהדיסק בלי נגיעה ברשת.

---

## מבנה הפרויקט (לאחר Part 2)

```
C:\Users\user\Desktop\agentic-edge-stack\
│
├── app/                            # קוד האפליקציה
│   ├── core/
│   │   └── config.py               # Pydantic Settings - מורחב עם הגדרות RAG
│   │
│   ├── llm/                        # Part 1 (הושלם)
│   │   ├── ollama_client.py        # OllamaClient (לא השתנה)
│   │   ├── factory.py              # get_llm_client()
│   │   └── errors.py               # LLMClientError
│   │
│   ├── rag/                        # Part 2 (הושלם!)
│   │   ├── types.py                # Chunk, RetrievalHit, RetrievalResult
│   │   ├── errors.py               # RAGError, IngestionError, RetrievalError
│   │   ├── chunker.py              # RecursiveCharacterTextSplitter wrapper
│   │   ├── embeddings.py           # SentenceTransformer wrapper (BGE)
│   │   ├── vector_store.py         # FAISS IndexFlatIP + metadata sidecar
│   │   ├── retriever.py            # Orchestrator + on-disk cache
│   │   └── prompt_builder.py       # System + context + question prompt
│   │
│   ├── agent/                      # Part 3 - תיקייה ריקה
│   ├── api/                        # Part 4 - תיקייה ריקה
│   └── schemas/                    # Part 5 - תיקייה ריקה
│
├── data/                           # Corpus (Part 2)
│   ├── README.md                   # מסביר את הקורפוס
│   ├── 01_llama32_model_card.md    # ~1.5 עמ'
│   ├── 02_faiss_overview.md        # ~1.2 עמ'
│   ├── 03_sentence_transformers_and_bge.md   # ~1.3 עמ'
│   ├── 04_ollama_runtime.md        # ~1.2 עמ'
│   ├── 05_rag_concepts.md          # ~1.5 עמ'
│   └── cache/                      # gitignored - FAISS index + manifest
│
├── models/                         # gitignored
│   ├── llama-3.2-1b-instruct-q4_k_m.gguf   # ~770 MB (Part 1)
│   └── bge-small-en-v1.5/                  # ~134 MB (Part 2)
│       ├── 1_Pooling/config.json
│       ├── 2_Normalize/             # ריקה (Normalize layer חסר state)
│       ├── config.json
│       ├── config_sentence_transformers.json
│       ├── model.safetensors
│       ├── modules.json
│       ├── sentence_bert_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
│
├── tests/
│   ├── verify_ollama.py            # Part 1 (לא השתנה)
│   ├── verify_rag.py               # Part 2 - end-to-end + log
│   └── logs/                       # Test logs (committed - זה ה-deliverable)
│       └── rag_run_<timestamp>.txt
│
├── scripts/
│   ├── deploy.ps1                  # פריסה ראשונית
│   ├── import_model.ps1            # ייבוא GGUF ל-Ollama (Path B)
│   ├── import_embed_model.ps1      # ייבוא BGE מ-HF (Path B מקביל)
│   └── ingest.py                   # CLI - בנייה/ריענון של אינדקס FAISS
│
├── Modelfile                       # Llama 3.2 chat template (Part 1)
├── pyproject.toml                  # מורחב: sentence-transformers, faiss-cpu, langchain-text-splitters
├── .env / .env.example             # מורחבים: EMBED_*, RAG_*, DATA_DIR, CACHE_DIR
├── .gitignore                      # כולל models/, data/cache/, *.gguf
├── README.md                       # מורחב: סקציה ל-Part 2 + design notes
└── HANDOFF.md                      # המסמך הזה
```

---

## תכולת הקבצים החשובים

### `pyproject.toml`
```toml
[project]
name = "agentic-edge-stack"
version = "0.1.0"
description = "Locally hosted agentic AI assistant with RAG and streaming API"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    { name = "Your Name" }
]
license = { text = "MIT" }

dependencies = [
    "requests>=2.32.3",
    "pydantic-settings>=2.6.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["app*"]
```

### `.env.example`
```
# Ollama Server Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
REQUEST_TIMEOUT=120
```

### `app/core/config.py`
```python
"""Application configuration using Pydantic Settings.

All configuration is loaded from environment variables (or .env file).
This is the SINGLE source of truth for application settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings."""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    request_timeout: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
```

### `app/llm/ollama_client.py`
מחלקה עם 3 מתודות:
- `health_check()` - בדיקה שהשרת חי.
- `generate(prompt)` - מחזיר תשובה מלאה.
- `generate_stream(prompt)` - generator שמחזיר tokens (להשתמש בשלב 4).

המחלקה משתמשת ב-Dependency Injection (host, model, timeout מקבלים פרמטרים שיורשים מ-settings).

יש מחלקת שגיאה ייעודית: `OllamaError`.

### `tests/verify_ollama.py`
סקריפט שמדגים "Hello World":
1. יוצר OllamaClient.
2. בודק health.
3. שולח prompt: "Say hello in exactly one short sentence."
4. מודד זמן עם `time.perf_counter()`.
5. מציג: prompt, response, elapsed time.
6. exit code 0 בהצלחה / 1 בכישלון.

### `scripts/deploy.ps1`
5 שלבים:
1. בדיקת Python.
2. בדיקת Ollama.
3. יצירת venv + התקנת dependencies (pip install -e .).
4. `ollama pull llama3.2:1b` (עם בדיקת `$LASTEXITCODE`).
5. בדיקת שרת Ollama.

### `Modelfile` (root - חדש!)
מגדיר את הייבוא של ה-GGUF המקומי ל-Ollama. מכיל chat template רשמי של Llama 3.2 (Meta) ו-stop tokens (`<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>`). זה הקובץ שמבטיח שגם tool calling יעבוד ב-Part 3.

### `scripts/import_model.ps1` (חדש!)
מקביל לוגית ל-`pull_model.ps1` אבל מהמסלול המקומי: מקבל GGUF מ-`Downloads`, מעתיק ל-`models/` בשם נורמלי, ומריץ `ollama create llama3.2:1b -f Modelfile`. כולל verification של `ollama list`.

### `README.md`
כולל:
- כותרת ותיאור.
- **Model Selection Rationale** (סקציה מפורטת על למה llama3.2:1b ולא 3B).
- Architecture (ASCII tree).
- Requirements + Quick Start (3 צעדים).
- Configuration.
- Project Status (checkboxes לכל 5 השלבים).

---

## מצב נוכחי - מה כבר עובד ומה לא

### ✅ מה הושלם:
- כל מבנה התיקיות נוצר.
- כל 13 הקבצים נכתבו ועובדים.
- Python 3.13.3 מותקן.
- Ollama 0.22.0 מותקן (דרך `irm https://ollama.com/install.ps1 | iex`).
- Ollama server רץ על http://localhost:11434.
- venv נוצר.
- תלויות הותקנו (`pip install -e .`).

### ⏳ הבעיה ההיסטורית (רשת מקומית / NetFree):
- **הורדת משקלים דרך Ollama / HuggingFace** נחתכה או נחסמה; `ollama pull` נכשל ב-EOF גם למודלים קטנים.
- נבדק: GGUF מ-HF דרך curl → דף חסימה של NetFree (לא קובץ מודל).

### ✅ פתרון Part 1 (סופי - מומש!):
- **קובץ GGUF ידני (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 770 MB)** הועבר ממחשב אחר דרך Jumbo Mail.
- נכתב `Modelfile` ב-root של הפרויקט עם **chat template רשמי של Llama 3.2** + stop tokens (קריטי ל-Part 3 / Tool Calling).
- נכתב `scripts/import_model.ps1` שמעתיק את הקובץ מ-Downloads ל-`models/`, מריץ `ollama create`, ומאמת.
- `ollama list` מראה: `llama3.2:1b 807 MB` רשום.
- `python tests/verify_ollama.py` עובר ירוק: "Hello!" תוך ~7 שניות.
- **Part 1 הושלם רשמית.**

### 🧹 ניקיון - הוסרה תמיכת OpenRouter (החלטת המשתמשת):
- נמחקו: `app/llm/openrouter_client.py`, וסקריפטים מתים (`pull_model.ps1`, `download_gguf.ps1`, `download_model_hf.ps1`).
- נוקו: `config.py`, `factory.py`, `verify_ollama.py`, `.env`, `.env.example`, `README.md`.
- המערכת תומכת **רק ב-Ollama מקומי** - תואם את דרישת המטלה ("locally hosted").

---

## בדיקות שבוצעו על המחשב

```powershell
# 1. בדיקת חומרה
Get-CimInstance Win32_Processor          # i7-6500U, 2 cores
Get-CimInstance Win32_ComputerSystem     # 16 GB RAM
Get-CimInstance Win32_VideoController    # Intel HD 520 (no NVIDIA)

# 2. בדיקת מקום בדיסק
Get-PSDrive -PSProvider FileSystem       # C: 159 GB free

# 3. בדיקת Python
python --version                          # 3.13.3

# 4. בדיקת Ollama
ollama --version                          # 0.22.0
ollama list                               # ריק - מודל לא ירד

# 5. בדיקת Ollama server
Invoke-WebRequest http://localhost:11434/api/tags  # 200 OK

# 6. בדיקת אנטי-וירוס
Get-MpComputerStatus                      # Defender NOT running
Get-CimInstance Win32_Product...          # אין אנטי-וירוס צד שלישי
Get-NetFirewallProfile                    # Firewall enabled (לא חוסם HTTPS יוצא)
```

---

## הצעדים הבאים (ברורים!)

### ✅ Part 1 (הושלם)
- [x] מבנה פרויקט + תלויות.
- [x] Ollama מותקן ורץ.
- [x] Modelfile + import_model.ps1.
- [x] `llama3.2:1b` רשום ב-Ollama (807 MB).
- [x] `verify_ollama.py` עובר - "Hello World" עובד.

### ✅ Part 2 (הושלם!)
- [x] קורפוס של 5 מסמכי Markdown תחת `data/` (~6.7 עמ' טכניים).
- [x] תלויות חדשות ב-`pyproject.toml`: `sentence-transformers`, `faiss-cpu`, `langchain-text-splitters`, `numpy`.
- [x] 7 מודולים ב-`app/rag/` (types, errors, chunker, embeddings, vector_store, retriever, prompt_builder).
- [x] `app/core/config.py` הורחב עם הגדרות RAG (chunk size, top-K, threshold, model, dirs).
- [x] `scripts/import_embed_model.ps1` - הורדה אוטומטית של BGE מ-HF (Path A).
- [x] BGE-small-en-v1.5 הועבר ידנית ל-`models/bge-small-en-v1.5/` (Path B - בגלל NetFree).
- [x] `scripts/ingest.py` - CLI לבניית אינדקס FAISS עם cache.
- [x] `tests/verify_rag.py` - מריץ 6 שאילתות (5 in-domain + 1 OOD), מדפיס trace, שומר לוג ל-`tests/logs/`.
- [x] **לוג הריצה האחרונה**: `tests/logs/rag_run_20260505T105750Z.txt` - 5 שאילתות in-domain עם ציוני 0.75-0.86, OOD עם 0 hits ותשובת fallback.
- [x] README.md מורחב עם סקציה Part 2 + design notes.

### עכשיו: צעד GitHub (אם עוד לא בוצע)
1. ליצור repo פומבי בגיטהאב.
2. `git remote add origin <url>`.
3. `git add . && git commit -m "Part 2: in-memory FAISS RAG over BGE embeddings" && git push -u origin main`.

> שים/י לב: `models/`, `data/cache/`, ו-`.env` ב-gitignore. ה-corpus תחת `data/*.md` וה-test log תחת `tests/logs/` **כן** מועלים - הם חלק מה-deliverable של Part 2.

### צעד הבא: Part 3 - Agentic Orchestrator
- לעטוף את `Retriever.retrieve()` ככלי (Tool) בסכמה שמובנת ל-LangChain / LangGraph.
- Loop של agent שמחליט **מתי** להפעיל retrieval (לא תמיד) לפי השאילתה.
- Trace שמדגים turn אחד שכלל קריאה לכלי + שילוב התוצאה במענה.
- אופציה: לכתוב agent מינימלי משלנו (ReAct loop ידני) במקום LangChain - יותר שקוף, פחות תלויות, מרשים בראיון.

### צעד 4: Part 4 - FastAPI + SSE Streaming
- Endpoint `/chat` שמקבל query + chat_history.
- מפעיל את ה-agent מ-Part 3.
- מחזיר Server-Sent Events של tokens תוך כדי generation (משתמש ב-`OllamaClient.generate_stream` הקיים).

### צעד 5: בונוסים (לפי זמן שנותר)
1. **Structured Output** - Pydantic schema עבור תשובות מובנות (לדוגמה: `{"answer": str, "sources": list[str]}`).
2. **Quantization Profiling** - השוואה Q4_K_M vs Q5_K_M vs FP16 (אם נצליח להוריד נוסף): TPS, RAM, איכות.
3. **Production-Grade Vector DB** - הרצת Qdrant על k3s/minikube עם אותו corpus.

---

## נקודות חשובות לזכור

1. **גישת המשתמשת:** הסבר כל שורת קוד, כל החלטה - היא משווה למסמך המטלה.
2. **לחבר כל בחירה לדרישת מטלה ספציפית** - לא להמליץ על משהו בלי הסבר מקצועי.
3. **חומרה חלשה** - לא להציע פתרונות שיכבידו (vLLM, מודלים גדולים, K8s כבד).
4. **רשת איטיה** - להעדיף פתרונות בעלי retry ו-resumable downloads.
5. **כל שינוי בקוד צריך גם לעדכן את README** (במיוחד את "Project Status" ו"Configuration").
6. **השפה:** הכל בעברית למעט קוד, שמות קבצים, ומונחים טכניים.
7. **Code style ב-Python:** type hints על כל פונקציה, docstrings ב-Google style, מחלקות שגיאה ייעודיות, Dependency Injection.
8. **Code style ב-PowerShell:** Help blocks (`<# ... #>`), `$ErrorActionPreference = "Stop"`, צבעים לפלט (Cyan/Yellow/Green/Red), exit codes (0=הצלחה, 1=כישלון).

---

## הוראת המשך לצ'אט החדש

הצ'אט החדש צריך:
1. לקרוא את הקובץ הזה לפני כל פעולה.
2. לעבוד במצב Agent (לא Plan) כי כבר עברנו את שלב התכנון.
3. להמשיך מהצעד הבא: השלמת הורדת המודל.
4. אחרי שהמודל ירד - להריץ verify_ollama.py.
5. רק אחרי שזה עובד - להתחיל את Part 2.

**הקוד כבר נכתב, הסביבה כבר מוקמת. הצעד הבא הוא טכני בלבד: להוריד את המודל ולהריץ את הסקריפט.**
