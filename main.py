from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3, hashlib, secrets, json, os, smtplib, logging, hashlib as _hs
import itertools, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta
from groq import Groq
from typing import Optional, List
import pytz
import random

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

try:
    import requests as _req_lib
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    _req_lib = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("examai")

# ── MULTI-KEY CONFIG ──────────────────────────────────────────────────────────

# Groq key pool
GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
if not GROQ_KEYS:
    single = os.getenv("GROQ_API_KEY", "")
    if single:
        GROQ_KEYS = [single]
    else:
        raise RuntimeError("No GROQ API keys found! Set GROQ_API_KEYS in .env")

_groq_key_cycle = itertools.cycle(GROQ_KEYS)

def get_groq_client() -> Groq:
    return Groq(api_key=next(_groq_key_cycle))

# OpenRouter key pool + best models for multilingual exam AI
OPENROUTER_KEYS = [k.strip() for k in os.getenv("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
_or_key_cycle = itertools.cycle(OPENROUTER_KEYS) if OPENROUTER_KEYS else None

OPENROUTER_MODELS = [
     "openai/gpt-oss-120b"
]
_or_model_cycle = itertools.cycle(OPENROUTER_MODELS) if OPENROUTER_KEYS else None

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
APP_URL   = os.getenv("APP_URL", "http://localhost:8000")
TZ_NAME   = os.getenv("TIMEZONE", "Asia/Kolkata")
TZ        = pytz.timezone(TZ_NAME)

DB_PATH                   = "examai.db"
UNLOCK_REQUIRED_ATTEMPTS  = 20
UNLOCK_REQUIRED_ACCURACY  = 70
REQUIRED_SUB_ATTEMPTS     = 5
MAX_Q_PER_SUBCHAPTER      = 50   # ← Increased from 20
SUBCHAPTERS_PER_CHAPTER   = 8    # ← Increased from 5 for richer coverage
DYNAMIC_MAX_PER_CALL      = 4
COACH_MEMORY_LIMIT        = 30
DAILY_CHALLENGE_XP        = 75
COINS_PER_CORRECT         = 2
HINT_COST_COINS           = 5
FIFTY_FIFTY_COST_COINS    = 8
LOBBY_SECONDS             = 120
INTERVIEW_MAX_QUESTIONS   = 15

LATENCY_WARNING_THRESHOLD = 60
LATENCY_SLOW_SINGLE       = 90
LATENCY_FAST_THRESHOLD    = 20

# ── LANGUAGE DETECTION ────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "japanese": {
        "patterns": ["japanese", "日本語", "jlpt", " n5", " n4", " n3", " n2", " n1",
                     "nihongo", "kanji", "hiragana", "katakana"],
        "native_name": "日本語",
        "instruction": (
            "CRITICAL LANGUAGE RULE: This is a Japanese language exam. "
            "ALL vocabulary questions MUST include: Japanese text (in hiragana/katakana/kanji) | "
            "Romaji reading | English meaning. "
            "Grammar questions MUST use actual Japanese sentences with English translations. "
            "Chapter names, sub-topic names, and lesson content MUST be bilingual "
            "(Japanese + English). "
            "For JLPT N5: focus on ~800 basic words, hiragana, katakana, and ~100 kanji. "
            "For JLPT N4: ~1500 words, ~300 kanji. "
            "Example vocab question format: What does 「食べる」(たべる) mean? "
            "Always produce Japanese text in every question."
        ),
    },
    "korean": {
        "patterns": ["korean", "한국어", "topik", "hangul", "korea"],
        "native_name": "한국어",
        "instruction": (
            "CRITICAL: This is a Korean language exam. Include Korean Hangul text in every question. "
            "Vocab questions must show: Korean word | Romanization | English meaning. "
            "Grammar questions must use actual Korean sentences."
        ),
    },
    "chinese": {
        "patterns": ["chinese", "中文", "mandarin", "hsk", "putonghua", "cantonese"],
        "native_name": "中文",
        "instruction": (
            "CRITICAL: This is a Chinese language exam. Include Chinese characters (汉字) in every question. "
            "Show: Chinese | Pinyin | English meaning for vocabulary."
        ),
    },
    "french": {
        "patterns": ["french", "français", "delf", "dalf", "tcf", "french language"],
        "native_name": "Français",
        "instruction": (
            "CRITICAL: This is a French language exam. Questions should be bilingual (French + English). "
            "Vocabulary and grammar questions must include actual French sentences/words."
        ),
    },
    "german": {
        "patterns": ["german", "deutsch", "goethe", "dsh", "german language"],
        "native_name": "Deutsch",
        "instruction": (
            "CRITICAL: This is a German language exam. Include actual German text in questions. "
            "Show: German word | Article (der/die/das) | English meaning."
        ),
    },
    "spanish": {
        "patterns": ["spanish", "español", "dele", "siele"],
        "native_name": "Español",
        "instruction": (
            "CRITICAL: This is a Spanish language exam. Include actual Spanish text in questions."
        ),
    },
    "hindi": {
        "patterns": ["hindi", "हिंदी", "hindi language"],
        "native_name": "हिंदी",
        "instruction": (
            "CRITICAL: This is a Hindi language exam. Include Hindi (Devanagari) text in questions. "
            "Show: Hindi word | Transliteration | English meaning."
        ),
    },
    "arabic": {
        "patterns": ["arabic", "عربي", "عربية", "arabic language"],
        "native_name": "العربية",
        "instruction": (
            "CRITICAL: This is an Arabic language exam. Include Arabic script in questions. "
            "Show: Arabic word | Transliteration | English meaning."
        ),
    },
    "tamil": {
        "patterns": ["tamil", "தமிழ்", "tamil language"],
        "native_name": "தமிழ்",
        "instruction": (
            "CRITICAL: This is a Tamil language exam. Include Tamil script in questions."
        ),
    },
}

def detect_exam_language(exam_type: str, exam_goal: str = "") -> Optional[str]:
    """Detect if the exam is a language-learning exam and return the language key."""
    text = (exam_type + " " + exam_goal).lower()
    for lang, info in LANGUAGE_MAP.items():
        if any(p in text for p in info["patterns"]):
            return lang
    return None

def get_language_instruction(lang_key: Optional[str]) -> str:
    if not lang_key or lang_key not in LANGUAGE_MAP:
        return ""
    return "\n\n" + LANGUAGE_MAP[lang_key]["instruction"] + "\n"

def get_language_chapter_hint(lang_key: Optional[str], exam_type: str) -> str:
    """Return hint for generating language-appropriate chapter names."""
    if not lang_key or lang_key not in LANGUAGE_MAP:
        return ""
    native = LANGUAGE_MAP[lang_key]["native_name"]
    return (
        f"\nIMPORTANT: Generate chapters specific to {exam_type} curriculum. "
        f"Chapter names should be descriptive and include relevant {native} text where appropriate. "
        f"Focus on: script/writing systems, vocabulary by category, grammar patterns, "
        f"reading comprehension, listening, cultural knowledge, and exam-specific skills."
    )

# ── EXAM QUESTION STYLES ──────────────────────────────────────────────────────

EXAM_QUESTION_STYLES = {
    "default": [
        "Application-level scenario questions",
        "Conceptual traps with plausible wrong answers",
        "Case-study based questions",
        "Data interpretation questions",
        "Exception/edge-case questions",
    ],
    "UPSC": [
        "Statement-based (Statement 1/2 correct?)",
        "Match the following columns",
        "Chronological ordering",
        "Assertion-Reason format",
        "Map-based and data interpretation",
    ],
    "JEE": [
        "Multi-concept integration problems",
        "Calculation-heavy numerical problems",
        "Graph/diagram interpretation",
        "Multiple correct options type",
        "Integer-type answer questions",
    ],
    "NEET": [
        "Diagram labeling questions",
        "Clinical scenario application",
        "Assertion-Reason format",
        "Exception questions (which is NOT correct)",
        "Compare and contrast structure/function",
    ],
    "GATE": [
        "NAT (Numerical Answer Type)",
        "MCQ with negative marking traps",
        "Algorithm trace/output prediction",
        "Design and analysis questions",
        "Previous year pattern questions",
    ],
    "CAT": [
        "Inference-based questions",
        "Data sufficiency",
        "Logical reasoning chains",
        "Reading comprehension traps",
        "Quantitative approximation",
    ],
    "LANGUAGE": [
        "Vocabulary meaning/usage in context",
        "Grammar rule application",
        "Sentence correction / error spotting",
        "Fill in the blank (choose correct form)",
        "Reading comprehension questions",
        "Choose the correct translation",
    ],
}

ACHIEVEMENTS = [
    {"id": "first_blood",         "name": "First Blood",          "icon": "🎯", "desc": "Complete your first question",                   "xp": 50},
    {"id": "hot_streak",          "name": "Hot Streak",           "icon": "🔥", "desc": "Answer 5 correct in a row",                      "xp": 100},
    {"id": "century",             "name": "Century",              "icon": "💯", "desc": "Complete 100 questions",                         "xp": 200},
    {"id": "accuracy_ace",        "name": "Accuracy Ace",         "icon": "🏹", "desc": "Achieve 90%+ accuracy in a test",                "xp": 150},
    {"id": "week_warrior",        "name": "Week Warrior",         "icon": "⚔️",  "desc": "7-day study streak",                            "xp": 300},
    {"id": "chapter_master",      "name": "Chapter Master",       "icon": "📚", "desc": "Complete all sub-chapters in a chapter",         "xp": 250},
    {"id": "comeback_king",       "name": "Comeback King",        "icon": "👑", "desc": "Improve score 20%+ from previous test",          "xp": 200},
    {"id": "weak_slayer",         "name": "Weak Slayer",          "icon": "⚡", "desc": "Score 80%+ in weak training session",            "xp": 175},
    {"id": "night_owl",           "name": "Night Owl",            "icon": "🦉", "desc": "Study after 10 PM",                             "xp": 75},
    {"id": "early_bird",          "name": "Early Bird",           "icon": "🌅", "desc": "Study before 7 AM",                             "xp": 75},
    {"id": "perfectionist",       "name": "Perfectionist",        "icon": "💎", "desc": "Score 100% in any test",                        "xp": 500},
    {"id": "grind_mode",          "name": "Grind Mode",           "icon": "🦾", "desc": "Complete 10 tests total",                       "xp": 150},
    {"id": "chapter_unlock",      "name": "Unlocked",             "icon": "🔓", "desc": "Unlock your second chapter",                    "xp": 100},
    {"id": "speed_demon",         "name": "Speed Demon",          "icon": "💨", "desc": "Answer 10 questions under 5 min avg",           "xp": 125},
    {"id": "persistent",          "name": "Persistent",           "icon": "🧱", "desc": "Complete 5 weak training sessions",             "xp": 200},
    {"id": "dynamic_debut",       "name": "Dynamic Debut",        "icon": "🌀", "desc": "Complete your first dynamic session",           "xp": 100},
    {"id": "dynamic_marathon",    "name": "Marathon Mind",        "icon": "🏃", "desc": "Answer 50 questions in one dynamic session",    "xp": 300},
    {"id": "dynamic_ace",         "name": "Dynamic Ace",          "icon": "🃏", "desc": "80%+ accuracy in a dynamic session",            "xp": 200},
    {"id": "chapter_completed",   "name": "Chapter Champion",     "icon": "🏆", "desc": "Fully complete a chapter",                     "xp": 400},
    {"id": "coach_scholar",       "name": "Coach Scholar",        "icon": "🎓", "desc": "Have 10 coaching conversations",               "xp": 120},
    {"id": "weak_master",         "name": "Weak Area Conqueror",  "icon": "🛡️", "desc": "Turn 3 weak questions into correct answers",   "xp": 250},
    {"id": "teach_me_mode",       "name": "Teach Me Mode",        "icon": "📖", "desc": "Use AI coaching to learn from 5 wrong answers", "xp": 180},
    {"id": "real_exam_ready",     "name": "Exam Ready",           "icon": "🎖️", "desc": "Score 80%+ in dynamic real-exam mode",         "xp": 350},
    {"id": "thousand_questions",  "name": "Question Crusher",     "icon": "💪", "desc": "Answer 1000 total questions",                  "xp": 600},
    {"id": "mood_tracker",        "name": "Self-Aware Scholar",   "icon": "🧠", "desc": "Log mood for 7 days",                         "xp": 100},
    {"id": "three_chapters",      "name": "Triple Threat",        "icon": "🔱", "desc": "Complete 3 chapters",                         "xp": 500},
    {"id": "coin_collector",      "name": "Coin Collector",       "icon": "🪙", "desc": "Earn 100 coins",                              "xp": 80},
    {"id": "bookworm",            "name": "Bookworm",             "icon": "📌", "desc": "Bookmark 10 questions",                       "xp": 60},
    {"id": "challenge_champion",  "name": "Challenge Champion",   "icon": "🌟", "desc": "Complete 7 daily challenges",                 "xp": 280},
    {"id": "perfect_streak_5",    "name": "On Fire",              "icon": "🔥", "desc": "5-question correct streak",                   "xp": 100},
    {"id": "perfect_streak_10",   "name": "Unstoppable",          "icon": "⚡", "desc": "10-question correct streak",                  "xp": 250},
    {"id": "speed_solver",        "name": "Speed Solver",         "icon": "⚡", "desc": "Average under 20s per question in a session",  "xp": 200},
    {"id": "time_improver",       "name": "Time Optimizer",       "icon": "⏱️", "desc": "Improve avg decision time by 20%",            "xp": 150},
    {"id": "global_debut",        "name": "Global Debut",         "icon": "🌍", "desc": "Participate in your first global test",       "xp": 200},
    {"id": "global_podium",       "name": "Podium Finish",        "icon": "🥉", "desc": "Finish in top 3 in a global test",           "xp": 500},
    {"id": "global_champion",     "name": "Global Champion",      "icon": "🥇", "desc": "Win a global test",                          "xp": 1000},
    {"id": "interview_debut",     "name": "Interview Debut",      "icon": "🎤", "desc": "Complete your first interview session",       "xp": 150},
    {"id": "interview_ace",       "name": "Interview Ace",        "icon": "🏅", "desc": "Score 80%+ in an interview session",         "xp": 300},
    {"id": "interview_veteran",   "name": "Interview Veteran",    "icon": "🎖️", "desc": "Complete 10 interview sessions",              "xp": 400},
]

XP_LEVELS = [0, 100, 250, 500, 900, 1400, 2100, 3000, 4200, 5700, 7500,
             10000, 13000, 17000, 22000]

INTERVIEW_TYPES = {
    "conceptual": {
        "label": "Conceptual Deep Dive",
        "icon": "🧠",
        "desc": "Tests your understanding of core concepts with open-ended questions",
        "q_style": "Ask conceptual 'explain', 'why', 'how' questions. Expect detailed answers.",
        "scoring": "Score 0-10 based on accuracy, depth, and clarity of explanation.",
    },
    "rapid_fire": {
        "label": "Rapid Fire MCQ",
        "icon": "⚡",
        "desc": "Fast-paced multiple choice questions to test breadth of knowledge",
        "q_style": "Generate MCQ questions with 4 options. Quick factual questions.",
        "scoring": "Binary: correct = 10, wrong = 0.",
    },
    "case_study": {
        "label": "Case Study Analysis",
        "icon": "📊",
        "desc": "Real-world scenarios requiring applied knowledge and analysis",
        "q_style": "Present realistic scenarios. Ask the student to analyze, diagnose, or recommend.",
        "scoring": "Score 0-10 based on analytical depth and correctness.",
    },
    "viva": {
        "label": "Viva / Oral Style",
        "icon": "🎤",
        "desc": "Progressive questioning — follow-up questions based on your answers",
        "q_style": "Start broad, then drill deeper based on previous answers. Build on their responses.",
        "scoring": "Score 0-10. Reward correct follow-up answers more.",
    },
    "technical": {
        "label": "Technical Problem Solving",
        "icon": "🔧",
        "desc": "Step-by-step problem solving and calculation questions",
        "q_style": "Numerical, derivation, code-tracing, or multi-step problem questions.",
        "scoring": "Score 0-10 with partial credit for correct approach even if final answer wrong.",
    },
}

# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ExamAI API v5")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── DATABASE ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def _columns(conn, table: str) -> set:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r["name"] for r in rows}
    except Exception:
        return set()

SAFE_TO_DROP = {"questions", "test_attempts", "daily_tests", "weak_sessions", "sub_chapters"}

ADDABLE_COLS = [
    ("users",              "exam_type",             "TEXT"),
    ("users",              "exam_goal",             "TEXT"),
    ("users",              "chapters",              "TEXT"),
    ("users",              "setup_done",            "INTEGER DEFAULT 0"),
    ("users",              "receive_reminders",     "INTEGER DEFAULT 0"),
    ("users",              "xp",                   "INTEGER DEFAULT 0"),
    ("users",              "level",                "INTEGER DEFAULT 1"),
    ("users",              "achievements",         "TEXT DEFAULT '[]'"),
    ("users",              "mood_log",             "TEXT DEFAULT '[]'"),
    ("users",              "consecutive_correct",  "INTEGER DEFAULT 0"),
    ("users",              "coins",                "INTEGER DEFAULT 0"),
    ("users",              "avatar_color",         "TEXT DEFAULT '#00e5ff'"),
    ("users",              "bio",                  "TEXT DEFAULT ''"),
    ("users",              "daily_test_chapter_idx","INTEGER DEFAULT 0"),
    ("users",              "exam_language",        "TEXT"),
    ("questions",          "sub_chapter",           "TEXT"),
    ("questions",          "difficulty",            "TEXT DEFAULT 'medium'"),
    ("questions",          "report_count",          "INTEGER DEFAULT 0"),
    ("test_attempts",      "time_taken",            "INTEGER DEFAULT 0"),
    ("test_attempts",      "session_type",          "TEXT DEFAULT 'daily'"),
    ("daily_tests",        "answers",               "TEXT"),
    ("daily_tests",        "score",                 "INTEGER DEFAULT 0"),
    ("daily_tests",        "total",                 "INTEGER DEFAULT 0"),
    ("daily_tests",        "completed",             "INTEGER DEFAULT 0"),
    ("daily_tests",        "completed_at",          "TEXT"),
    ("daily_tests",        "chapter_name",          "TEXT"),
    ("weak_sessions",      "answers",               "TEXT"),
    ("weak_sessions",      "score",                 "INTEGER DEFAULT 0"),
    ("weak_sessions",      "total",                 "INTEGER DEFAULT 0"),
    ("weak_sessions",      "completed",             "INTEGER DEFAULT 0"),
    ("dynamic_sessions",   "chapter",               "TEXT"),
    ("dynamic_sessions",   "exam_type",             "TEXT"),
    ("dynamic_sessions",   "question_pool",         "TEXT DEFAULT '[]'"),
    ("coach_messages",     "question_context",      "TEXT"),
]

REQUIRED_COLS = {
    "users":            {"id", "name", "email", "password_hash"},
    "sessions":         {"id", "user_id", "token"},
    "sub_chapters":     {"id", "user_id", "parent_chapter", "name", "order_index"},
    "questions":        {"id", "user_id", "chapter", "question", "options", "correct_answer", "explanation"},
    "test_attempts":    {"id", "user_id", "question_id", "user_answer", "is_correct"},
    "daily_tests":      {"id", "user_id", "test_date", "question_ids"},
    "weak_sessions":    {"id", "user_id", "question_ids"},
    "dynamic_sessions": {"id", "user_id", "seen_hashes", "score", "total", "is_active"},
    "dynamic_attempts": {"id", "session_id", "user_id", "question_text", "options",
                         "correct_answer", "explanation", "is_correct"},
}


def init_db():
    conn = get_db()

    for table, required in REQUIRED_COLS.items():
        if table not in SAFE_TO_DROP:
            continue
        existing = _columns(conn, table)
        if existing and not required.issubset(existing):
            missing = required - existing
            logger.warning("Table '%s' missing columns %s — dropping for rebuild.", table, missing)
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            name                    TEXT NOT NULL,
            email                   TEXT UNIQUE NOT NULL,
            password_hash           TEXT NOT NULL,
            exam_type               TEXT,
            exam_goal               TEXT,
            exam_language           TEXT,
            chapters                TEXT,
            setup_done              INTEGER DEFAULT 0,
            receive_reminders       INTEGER DEFAULT 0,
            xp                      INTEGER DEFAULT 0,
            level                   INTEGER DEFAULT 1,
            achievements            TEXT DEFAULT '[]',
            mood_log                TEXT DEFAULT '[]',
            consecutive_correct     INTEGER DEFAULT 0,
            coins                   INTEGER DEFAULT 0,
            avatar_color            TEXT DEFAULT '#00e5ff',
            bio                     TEXT DEFAULT '',
            daily_test_chapter_idx  INTEGER DEFAULT 0,
            created_at              TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            token      TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sub_chapters (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            parent_chapter TEXT NOT NULL,
            name           TEXT NOT NULL,
            order_index    INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS questions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            chapter        TEXT NOT NULL,
            sub_chapter    TEXT,
            question       TEXT NOT NULL,
            options        TEXT NOT NULL,
            correct_answer INTEGER NOT NULL,
            explanation    TEXT NOT NULL,
            difficulty     TEXT DEFAULT 'medium',
            report_count   INTEGER DEFAULT 0,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS test_attempts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            question_id  INTEGER NOT NULL,
            user_answer  INTEGER,
            is_correct   INTEGER NOT NULL DEFAULT 0,
            time_taken   INTEGER DEFAULT 0,
            session_type TEXT DEFAULT 'daily',
            attempted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS daily_tests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            test_date    TEXT NOT NULL,
            question_ids TEXT NOT NULL,
            answers      TEXT,
            score        INTEGER DEFAULT 0,
            total        INTEGER DEFAULT 0,
            completed    INTEGER DEFAULT 0,
            completed_at TEXT,
            chapter_name TEXT
        );
        CREATE TABLE IF NOT EXISTS weak_sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            question_ids TEXT NOT NULL,
            answers      TEXT,
            score        INTEGER DEFAULT 0,
            total        INTEGER DEFAULT 0,
            completed    INTEGER DEFAULT 0,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS mood_checkins (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            mood         INTEGER NOT NULL,
            energy       INTEGER NOT NULL,
            note         TEXT,
            checkin_date TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS coach_messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            message          TEXT NOT NULL,
            msg_type         TEXT DEFAULT 'coach',
            question_context TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS coach_memory (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL UNIQUE,
            summary      TEXT NOT NULL,
            updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS dynamic_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            chapter       TEXT,
            exam_type     TEXT,
            seen_hashes   TEXT DEFAULT '[]',
            question_pool TEXT DEFAULT '[]',
            score         INTEGER DEFAULT 0,
            total         INTEGER DEFAULT 0,
            is_active     INTEGER DEFAULT 1,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            ended_at      TEXT
        );
        CREATE TABLE IF NOT EXISTS dynamic_attempts (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER NOT NULL,
            user_id        INTEGER NOT NULL,
            question_text  TEXT NOT NULL,
            options        TEXT NOT NULL,
            correct_answer INTEGER NOT NULL,
            explanation    TEXT NOT NULL,
            user_answer    INTEGER,
            is_correct     INTEGER DEFAULT 0,
            time_taken     INTEGER DEFAULT 0,
            source         TEXT DEFAULT 'ai',
            attempted_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chapter_lessons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            chapter      TEXT NOT NULL,
            content      TEXT NOT NULL,
            structured   TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, chapter)
        );
        CREATE TABLE IF NOT EXISTS chapter_completions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            chapter        TEXT NOT NULL,
            completed_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, chapter)
        );
        CREATE TABLE IF NOT EXISTS teach_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            question_id     INTEGER,
            question_text   TEXT NOT NULL,
            correct_answer  INTEGER NOT NULL,
            user_answer     INTEGER,
            options         TEXT NOT NULL,
            explanation     TEXT NOT NULL,
            chapter         TEXT,
            completed       INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS study_plans (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL UNIQUE,
            plan         TEXT NOT NULL,
            updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bookmarks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            question_id  INTEGER NOT NULL,
            note         TEXT DEFAULT '',
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, question_id)
        );
        CREATE TABLE IF NOT EXISTS daily_challenges (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            challenge_date TEXT NOT NULL,
            question_text  TEXT NOT NULL,
            options        TEXT NOT NULL,
            correct_answer INTEGER NOT NULL,
            explanation    TEXT NOT NULL,
            chapter        TEXT,
            user_answer    INTEGER,
            completed      INTEGER DEFAULT 0,
            xp_reward      INTEGER DEFAULT 75,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, challenge_date)
        );
        CREATE TABLE IF NOT EXISTS question_reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            question_id  INTEGER NOT NULL,
            reason       TEXT NOT NULL,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS powerup_uses (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            powerup_type TEXT NOT NULL,
            question_id  INTEGER,
            result       TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS question_latency (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            question_id    INTEGER,
            session_type   TEXT NOT NULL DEFAULT 'daily',
            time_taken_sec INTEGER NOT NULL DEFAULT 0,
            is_correct     INTEGER DEFAULT 0,
            chapter        TEXT,
            difficulty     TEXT DEFAULT 'medium',
            recorded_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS latency_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            session_type    TEXT NOT NULL,
            session_ref_id  INTEGER,
            total_questions INTEGER DEFAULT 0,
            avg_time_sec    REAL DEFAULT 0,
            slowest_sec     INTEGER DEFAULT 0,
            fastest_sec     INTEGER DEFAULT 0,
            slow_count      INTEGER DEFAULT 0,
            fast_count      INTEGER DEFAULT 0,
            chapter         TEXT,
            recorded_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS global_tests (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            title            TEXT NOT NULL,
            exam_type        TEXT NOT NULL,
            topic            TEXT NOT NULL,
            question_ids     TEXT NOT NULL DEFAULT '[]',
            scheduled_at     TEXT NOT NULL,
            starts_at        TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL DEFAULT 60,
            status           TEXT NOT NULL DEFAULT 'draft',
            created_by       TEXT NOT NULL DEFAULT 'admin',
            created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at         TEXT,
            winner_user_id   INTEGER,
            winner_name      TEXT,
            winner_score     INTEGER
        );
        CREATE TABLE IF NOT EXISTS global_questions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            global_test_id INTEGER NOT NULL,
            question       TEXT NOT NULL,
            options        TEXT NOT NULL,
            correct_answer INTEGER NOT NULL,
            explanation    TEXT NOT NULL,
            difficulty     TEXT DEFAULT 'medium',
            topic_tag      TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS global_participants (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            global_test_id INTEGER NOT NULL,
            user_id        INTEGER NOT NULL,
            joined_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            started_at     TEXT,
            submitted_at   TEXT,
            answers        TEXT DEFAULT '{}',
            score          INTEGER DEFAULT 0,
            total          INTEGER DEFAULT 0,
            time_taken_sec INTEGER DEFAULT 0,
            rank           INTEGER,
            UNIQUE(global_test_id, user_id)
        );

        -- ── Interview Mode ─────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS interview_sessions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            chapter          TEXT,
            exam_type        TEXT,
            interview_type   TEXT NOT NULL DEFAULT 'conceptual',
            total_questions  INTEGER DEFAULT 10,
            questions_asked  INTEGER DEFAULT 0,
            total_score      REAL DEFAULT 0,
            max_score        REAL DEFAULT 0,
            is_active        INTEGER DEFAULT 1,
            summary          TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            ended_at         TEXT
        );
        CREATE TABLE IF NOT EXISTS interview_qa (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER NOT NULL,
            user_id         INTEGER NOT NULL,
            question_number INTEGER NOT NULL DEFAULT 1,
            question        TEXT NOT NULL,
            question_type   TEXT DEFAULT 'open',
            options         TEXT,
            correct_answer  INTEGER,
            user_answer     TEXT,
            ai_feedback     TEXT,
            score           REAL DEFAULT 0,
            max_score       REAL DEFAULT 10,
            is_correct      INTEGER DEFAULT 0,
            time_taken      INTEGER DEFAULT 0,
            asked_at        TEXT DEFAULT CURRENT_TIMESTAMP,
            answered_at     TEXT
        );
    """)
    conn.commit()

    for table, column, col_def in ADDABLE_COLS:
        if column not in _columns(conn, table):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
                conn.commit()
                logger.info("Migration: added %s.%s", table, column)
            except Exception as exc:
                logger.debug("Migration skip %s.%s — %s", table, column, exc)

    conn.close()


init_db()

# ── WEB SEARCH ────────────────────────────────────────────────────────────────

def search_topic_context(topic: str, exam_type: str = "", deep: bool = False) -> str:
    if not HAS_REQUESTS:
        return ""
    limit = 6000 if deep else 3500
    try:
        wiki_resp = _req_lib.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": topic, "prop": "extracts",
                    "exintro": False, "explaintext": True, "format": "json",
                    "redirects": 1, "exchars": limit},
            timeout=8,
            headers={"User-Agent": "ExamAI/5.0"},
        )
        if wiki_resp.status_code == 200:
            data  = wiki_resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if text and len(text) > 300 and "-1" not in str(page.get("pageid", "")):
                    return text[:limit]
    except Exception as e:
        logger.debug("Wikipedia search failed: %s", e)
    return ""

# ── CORE HELPERS ──────────────────────────────────────────────────────────────

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def q_hash(text: str) -> str:
    return _hs.md5(text.strip().lower().encode()).hexdigest()[:16]

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Not authenticated — missing Authorization header")
    authorization = authorization.strip()
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated — expected 'Bearer <token>'")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Not authenticated — empty token")
    conn  = get_db()
    try:
        row = conn.execute(
            "SELECT s.*, u.id as uid, u.name, u.email, u.exam_type, u.exam_goal, "
            "u.exam_language, u.chapters, u.setup_done, u.receive_reminders, u.xp, u.level, "
            "u.achievements, u.mood_log, u.consecutive_correct, u.coins, "
            "u.avatar_color, u.bio, u.daily_test_chapter_idx "
            "FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=?", (token,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(401, "Invalid or expired token — please log in again")
    return dict(row)

def parse_json(text: str):
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").lstrip("JSON").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    try:
        return json.loads(text)
    except Exception:
        pass
    for s, e in [('[', ']'), ('{', '}')]:
        i = text.find(s)
        j = text.rfind(e) + 1
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j])
            except Exception:
                pass
    raise ValueError("Cannot parse JSON from AI response")

def fmt_q(q):
    if q is None:
        return None
    keys = q.keys() if hasattr(q, "keys") else q
    return {
        "id": q["id"], "chapter": q["chapter"],
        "sub_chapter": q["sub_chapter"] if "sub_chapter" in keys else None,
        "question": q["question"],
        "options": json.loads(q["options"]) if isinstance(q["options"], str) else q["options"],
        "correct_answer": q["correct_answer"], "explanation": q["explanation"],
        "difficulty": q["difficulty"] if "difficulty" in keys else "medium",
    }

def safe_answer_int(ua):
    if ua is None:
        return None
    try:
        return int(ua)
    except (TypeError, ValueError):
        return None

def get_exam_style_hints(exam_type: str) -> str:
    # Check if it's a language exam
    lang = detect_exam_language(exam_type)
    if lang:
        return "\n".join(f"  - {s}" for s in EXAM_QUESTION_STYLES["LANGUAGE"])
    for key in EXAM_QUESTION_STYLES:
        if key == "LANGUAGE":
            continue
        if key.upper() in (exam_type or "").upper():
            styles = EXAM_QUESTION_STYLES[key]
            return "\n".join(f"  - {s}" for s in styles)
    return "\n".join(f"  - {s}" for s in EXAM_QUESTION_STYLES["default"])

def normalize_correct_answer(ca, options: list) -> Optional[int]:
    if isinstance(ca, int) and 0 <= ca <= 3:
        return ca
    if isinstance(ca, str):
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3,
                      "a": 0, "b": 1, "c": 2, "d": 3}
        clean = ca.strip().rstrip(".")
        if clean in letter_map:
            return letter_map[clean]
        if clean in {"0", "1", "2", "3"}:
            return int(clean)
        for i, opt in enumerate(options or []):
            if str(opt).strip().lower() == clean.lower():
                return i
        if len(clean) >= 2 and clean[0].upper() in letter_map:
            return letter_map[clean[0].upper()]
    return None

def validate_question_dict(q: dict) -> Optional[dict]:
    if not isinstance(q, dict):
        return None
    question_text = str(q.get("question", "")).strip()
    if not question_text or len(question_text) < 10:
        return None
    options = q.get("options", [])
    if not isinstance(options, list):
        return None
    options = [str(o).strip() for o in options if str(o).strip()]
    if len(options) != 4:
        return None
    ca_raw = q.get("correct_answer")
    ca = normalize_correct_answer(ca_raw, options)
    if ca is None:
        return None
    if len(set(o.lower() for o in options)) < 4:
        return None
    explanation = str(q.get("explanation", "")).strip()
    if not explanation or len(explanation) < 10:
        explanation = f"The correct answer is option {ca + 1}: {options[ca]}."
    difficulty = str(q.get("difficulty", "medium")).lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    return {
        "question": question_text,
        "options": options,
        "correct_answer": ca,
        "explanation": explanation,
        "difficulty": difficulty,
    }

def build_question_prompt(topic: str, exam_type: str, count: int,
                           style_hints: str, ctx_block: str = "",
                           seen_count: int = 0,
                           lang_instruction: str = "") -> str:
    diversity_note = (
        f"\nIMPORTANT: You have already generated {seen_count} questions on this topic. "
        "Make these questions on DIFFERENT sub-aspects — no repeats."
    ) if seen_count > 0 else ""

    return f"""You are an expert question setter for {exam_type} exam.{lang_instruction}{ctx_block}{diversity_note}

Generate exactly {count} multiple-choice questions about: "{topic}"

Question styles to use (mix these):
{style_hints}

OUTPUT FORMAT — You MUST follow this EXACTLY:
{{
  "questions": [
    {{
      "question": "Write the full question here",
      "options": [
        "First option text",
        "Second option text",
        "Third option text",
        "Fourth option text"
      ],
      "correct_answer": 2,
      "explanation": "Why option 3 (index 2) is correct. Why others are wrong.",
      "difficulty": "medium"
    }}
  ]
}}

STRICT RULES for correct_answer:
- It MUST be an INTEGER: 0, 1, 2, or 3
- 0 = first option is correct, 1 = second, 2 = third, 3 = fourth
- Double-check: options[correct_answer] must actually BE the right answer
- VARY the correct index — do not always use 0 or 1

STRICT RULES for options:
- All 4 options must be plausible (no obviously wrong distractors)
- Options must be distinct — no duplicates
- Do not include "All of the above" or "None of the above"

difficulty: must be exactly "easy", "medium", or "hard"

Return ONLY the JSON. No extra text before or after."""

# ── AI WRAPPERS WITH KEY ROTATION + OPENROUTER FALLBACK ───────────────────────

def openrouter_chat(messages: list, temperature: float = 0.7,
                    json_mode: bool = False) -> str:
    """Call OpenRouter as fallback when Groq is rate-limited."""
    if not OPENROUTER_KEYS or not HAS_REQUESTS or not _or_key_cycle:
        raise Exception("OpenRouter not configured")
    key   = next(_or_key_cycle)
    model = next(_or_model_cycle)
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": "ExamAI",
    }
    resp = _req_lib.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def groq_chat(prompt: str, temperature: float = 0.6, json_mode: bool = True) -> str:
    last_error = None
    for attempt in range(len(GROQ_KEYS)):
        try:
            groq_client = get_groq_client()
            kwargs: dict = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 4096,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = groq_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "quota" in err:
                logger.warning("Groq key %d/%d rate-limited, rotating...", attempt + 1, len(GROQ_KEYS))
                time.sleep(0.5)
                continue
            raise

    # All Groq keys exhausted → try OpenRouter fallback
    if OPENROUTER_KEYS:
        logger.warning("All Groq keys rate-limited, falling back to OpenRouter...")
        try:
            return openrouter_chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature, json_mode=json_mode)
        except Exception as or_err:
            logger.error("OpenRouter fallback also failed: %s", or_err)

    logger.error("All AI providers exhausted: %s", last_error)
    raise HTTPException(503, "AI is busy right now — all providers are rate-limited. Please retry in a moment.")


def groq_chat_with_history(messages: list, temperature: float = 0.7) -> str:
    last_error = None
    for attempt in range(len(GROQ_KEYS)):
        try:
            groq_client = get_groq_client()
            resp = groq_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "quota" in err:
                logger.warning("Groq key %d/%d rate-limited, rotating...", attempt + 1, len(GROQ_KEYS))
                time.sleep(0.5)
                continue
            raise

    if OPENROUTER_KEYS:
        logger.warning("All Groq keys rate-limited for history chat, using OpenRouter...")
        try:
            return openrouter_chat(messages, temperature=temperature)
        except Exception as or_err:
            logger.error("OpenRouter fallback failed: %s", or_err)

    raise HTTPException(503, "AI is busy right now. Please retry in a moment.")

# ── CHAPTER / PROGRESS HELPERS ────────────────────────────────────────────────

def chapter_stats(conn, user_id, chapter_name):
    s = conn.execute("""
        SELECT
            COUNT(DISTINCT q.id)                                          AS total,
            COUNT(DISTINCT CASE WHEN ta.id IS NOT NULL THEN q.id END)    AS attempted,
            COUNT(DISTINCT CASE WHEN ta.is_correct=1  THEN q.id END)     AS correct,
            SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END)             AS correct_attempts,
            COUNT(ta.id)                                                  AS total_attempts
        FROM questions q
        LEFT JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
        WHERE q.user_id=? AND q.chapter=?
    """, (user_id, user_id, chapter_name)).fetchone()

    if s is None:
        return 0, 0, 0
    attempted        = s["attempted"]        or 0
    correct_attempts = s["correct_attempts"] or 0
    total_attempts   = s["total_attempts"]   or 0
    acc = round(correct_attempts / max(total_attempts, 1) * 100, 1) if total_attempts > 0 else 0
    correct = s["correct"] or 0
    return attempted, correct, acc

def get_sub_chapter_progress(conn, user_id, chapter_name):
    subs = conn.execute(
        "SELECT name FROM sub_chapters WHERE user_id=? AND parent_chapter=? ORDER BY order_index",
        (user_id, chapter_name)).fetchall()
    if not subs:
        return 0, 0
    practiced = 0
    for sub in subs:
        cnt = conn.execute("""
            SELECT COUNT(ta.id) as cnt FROM questions q
            JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
            WHERE q.user_id=? AND q.sub_chapter=?
        """, (user_id, user_id, sub["name"])).fetchone()["cnt"] or 0
        if cnt >= REQUIRED_SUB_ATTEMPTS:
            practiced += 1
    return practiced, len(subs)

def is_unlocked(conn, user_id, chapters, idx):
    if idx == 0:
        return True
    prev = chapters[idx - 1]
    att, cor, acc = chapter_stats(conn, user_id, prev)
    subs_practiced, subs_total = get_sub_chapter_progress(conn, user_id, prev)
    if subs_total == 0:
        return att >= UNLOCK_REQUIRED_ATTEMPTS and acc >= UNLOCK_REQUIRED_ACCURACY
    return (subs_practiced >= subs_total
            and att >= UNLOCK_REQUIRED_ATTEMPTS
            and acc >= UNLOCK_REQUIRED_ACCURACY)

def is_chapter_complete(conn, user_id, chapters, idx) -> bool:
    ch = chapters[idx]
    att, cor, acc = chapter_stats(conn, user_id, ch)
    subs_p, subs_t = get_sub_chapter_progress(conn, user_id, ch)
    if subs_t == 0:
        return False
    return (subs_p >= subs_t
            and att >= UNLOCK_REQUIRED_ATTEMPTS
            and acc >= UNLOCK_REQUIRED_ACCURACY)

def get_active_daily_chapter(conn, user_id: int, chapters: list) -> tuple:
    if not chapters:
        return None, 0
    for i, ch in enumerate(chapters):
        if not is_unlocked(conn, user_id, chapters, i):
            idx = max(0, i - 1)
            return chapters[idx], idx
        if not is_chapter_complete(conn, user_id, chapters, i):
            return ch, i
    last = len(chapters) - 1
    return chapters[last], last

# ── LATENCY HELPERS ───────────────────────────────────────────────────────────

def compute_latency_stats(conn, user_id: int, limit_days: int = 30) -> dict:
    cutoff = (datetime.now() - timedelta(days=limit_days)).isoformat()
    rows = conn.execute(
        "SELECT time_taken_sec, is_correct, difficulty, chapter FROM question_latency "
        "WHERE user_id=? AND recorded_at >= ? AND time_taken_sec > 0",
        (user_id, cutoff)).fetchall()
    if not rows:
        return {"avg_time": 0, "median_time": 0, "slow_count": 0, "fast_count": 0,
                "total_tracked": 0, "slowest_chapter": None, "trend": "no_data"}

    times = sorted([r["time_taken_sec"] for r in rows])
    n = len(times)
    avg = sum(times) / n
    median = times[n // 2]
    slow = sum(1 for t in times if t > LATENCY_SLOW_SINGLE)
    fast = sum(1 for t in times if t < LATENCY_FAST_THRESHOLD)

    ch_times: dict = {}
    for r in rows:
        ch = r["chapter"] or "Unknown"
        if ch not in ch_times:
            ch_times[ch] = []
        ch_times[ch].append(r["time_taken_sec"])
    ch_avgs = {ch: sum(ts) / len(ts) for ch, ts in ch_times.items()}
    slowest_ch = max(ch_avgs, key=lambda x: ch_avgs[x]) if ch_avgs else None

    recent_cut = (datetime.now() - timedelta(days=7)).isoformat()
    prev_cut   = (datetime.now() - timedelta(days=14)).isoformat()
    recent_rows = conn.execute(
        "SELECT time_taken_sec FROM question_latency WHERE user_id=? AND recorded_at>=? AND time_taken_sec>0",
        (user_id, recent_cut)).fetchall()
    prev_rows = conn.execute(
        "SELECT time_taken_sec FROM question_latency WHERE user_id=? AND recorded_at>=? AND recorded_at<? AND time_taken_sec>0",
        (user_id, prev_cut, recent_cut)).fetchall()
    recent_avg = sum(r["time_taken_sec"] for r in recent_rows) / max(len(recent_rows), 1)
    prev_avg   = sum(r["time_taken_sec"] for r in prev_rows)   / max(len(prev_rows), 1)
    if not prev_rows:
        trend = "no_data"
    elif recent_avg < prev_avg * 0.85:
        trend = "improving"
    elif recent_avg > prev_avg * 1.15:
        trend = "slowing"
    else:
        trend = "stable"

    return {
        "avg_time": round(avg, 1),
        "median_time": median,
        "slow_count": slow,
        "fast_count": fast,
        "total_tracked": n,
        "slowest_chapter": slowest_ch,
        "slowest_chapter_avg": round(ch_avgs.get(slowest_ch, 0), 1) if slowest_ch else 0,
        "chapter_breakdown": {ch: round(v, 1) for ch, v in ch_avgs.items()},
        "trend": trend,
        "recent_avg": round(recent_avg, 1),
        "prev_avg": round(prev_avg, 1),
        "times_percentile_90": times[int(n * 0.9)] if n >= 10 else max(times),
    }

def get_latency_alerts(stats: dict) -> list:
    alerts = []
    if stats["avg_time"] > LATENCY_WARNING_THRESHOLD:
        alerts.append({
            "type": "slow_average", "severity": "high",
            "message": f"Your average decision time is {stats['avg_time']}s — above the {LATENCY_WARNING_THRESHOLD}s target.",
            "tip": "Try reading options before the full question to eliminate obvious wrong answers faster."
        })
    if stats["slow_count"] > stats["total_tracked"] * 0.3 and stats["total_tracked"] >= 10:
        alerts.append({
            "type": "too_many_slow", "severity": "medium",
            "message": f"{stats['slow_count']} questions took over {LATENCY_SLOW_SINGLE}s.",
            "tip": "If you're stuck after 30s, mark it and move on. Return if time allows."
        })
    if stats["trend"] == "slowing":
        alerts.append({
            "type": "slowing_trend", "severity": "medium",
            "message": f"Decision speed slowing — recent avg {stats['recent_avg']}s vs {stats['prev_avg']}s.",
            "tip": "Review your weakest chapter under a timer — simulate real exam pressure."
        })
    if stats["slowest_chapter"] and stats["slowest_chapter_avg"] > LATENCY_WARNING_THRESHOLD:
        alerts.append({
            "type": "slow_chapter", "severity": "medium",
            "message": f"Slowest chapter: '{stats['slowest_chapter']}' averaging {stats['slowest_chapter_avg']}s.",
            "tip": f"Spend 15 min doing a flashcard review of '{stats['slowest_chapter']}' core concepts."
        })
    if stats["trend"] == "improving":
        alerts.append({
            "type": "improving", "severity": "info",
            "message": f"Your speed is improving! Recent avg {stats['recent_avg']}s vs {stats['prev_avg']}s. 🚀",
            "tip": "Keep the momentum — try dynamic mode to push further."
        })
    return alerts

# ── GAMIFICATION ─────────────────────────────────────────────────────────────

def get_level(xp):
    for i, threshold in enumerate(XP_LEVELS):
        if xp < threshold:
            return max(1, i)
    return len(XP_LEVELS)

def award_xp(conn, user_id, amount, reason=""):
    conn.execute("UPDATE users SET xp = xp + ? WHERE id = ?", (amount, user_id))
    conn.commit()
    row = conn.execute("SELECT xp FROM users WHERE id=?", (user_id,)).fetchone()
    new_xp = row["xp"] if row else 0
    new_level = get_level(new_xp)
    conn.execute("UPDATE users SET level=? WHERE id=?", (new_level, user_id))
    conn.commit()
    return new_xp, new_level

def award_coins(conn, user_id, amount):
    conn.execute("UPDATE users SET coins = coins + ? WHERE id = ?", (amount, user_id))
    conn.commit()
    row = conn.execute("SELECT coins FROM users WHERE id=?", (user_id,)).fetchone()
    return row["coins"] if row else 0

def spend_coins(conn, user_id, amount) -> bool:
    row = conn.execute("SELECT coins FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or (row["coins"] or 0) < amount:
        return False
    conn.execute("UPDATE users SET coins = coins - ? WHERE id = ?", (amount, user_id))
    conn.commit()
    return True

def update_consecutive_correct(conn, user_id, is_correct: bool) -> int:
    if is_correct:
        conn.execute("UPDATE users SET consecutive_correct = consecutive_correct + 1 WHERE id=?", (user_id,))
    else:
        conn.execute("UPDATE users SET consecutive_correct = 0 WHERE id=?", (user_id,))
    conn.commit()
    row = conn.execute("SELECT consecutive_correct FROM users WHERE id=?", (user_id,)).fetchone()
    return row["consecutive_correct"] if row else 0

def get_streak(conn, user_id: int) -> int:
    streak = 0
    hist = conn.execute(
        "SELECT test_date FROM daily_tests WHERE user_id=? AND completed=1 ORDER BY test_date DESC",
        (user_id,)).fetchall()
    if hist:
        check = date.today()
        for r in hist:
            try:
                d = date.fromisoformat(r["test_date"])
            except Exception:
                break
            if d == check or d == check - timedelta(1):
                streak += 1
                check = d - timedelta(1)
            else:
                break
    return streak

def check_and_award_achievements(conn, user_id, context: dict) -> list:
    row = conn.execute("SELECT achievements FROM users WHERE id=?", (user_id,)).fetchone()
    earned = json.loads(row["achievements"] or "[]") if row else []
    earned_ids = {a["id"] for a in earned}
    new_achievements = []

    total_att    = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=?", (user_id,)).fetchone()[0]
    tests_done   = conn.execute("SELECT COUNT(*) FROM daily_tests WHERE user_id=? AND completed=1", (user_id,)).fetchone()[0]
    weak_done    = conn.execute("SELECT COUNT(*) FROM weak_sessions WHERE user_id=? AND completed=1", (user_id,)).fetchone()[0]
    dyn_done     = conn.execute("SELECT COUNT(*) FROM dynamic_sessions WHERE user_id=? AND is_active=0", (user_id,)).fetchone()[0]
    coach_count  = conn.execute("SELECT COUNT(*) FROM coach_messages WHERE user_id=? AND msg_type='user'", (user_id,)).fetchone()[0]
    mood_days    = conn.execute("SELECT COUNT(DISTINCT date(checkin_date)) FROM mood_checkins WHERE user_id=?", (user_id,)).fetchone()[0]
    teach_done   = conn.execute("SELECT COUNT(*) FROM teach_sessions WHERE user_id=? AND completed=1", (user_id,)).fetchone()[0]
    chapters_done= conn.execute("SELECT COUNT(*) FROM chapter_completions WHERE user_id=?", (user_id,)).fetchone()[0]
    bookmark_cnt = conn.execute("SELECT COUNT(*) FROM bookmarks WHERE user_id=?", (user_id,)).fetchone()[0]
    challenge_cnt= conn.execute("SELECT COUNT(*) FROM daily_challenges WHERE user_id=? AND completed=1", (user_id,)).fetchone()[0]
    coins_row    = conn.execute("SELECT COALESCE(coins, 0) as coins FROM users WHERE id=?", (user_id,)).fetchone()
    total_coins_ever = (coins_row["coins"] if coins_row else 0)
    interview_done = conn.execute("SELECT COUNT(*) FROM interview_sessions WHERE user_id=? AND is_active=0", (user_id,)).fetchone()[0]
    interview_ace_cnt = conn.execute(
        "SELECT COUNT(*) FROM interview_sessions WHERE user_id=? AND is_active=0 "
        "AND max_score > 0 AND (total_score / max_score) >= 0.8", (user_id,)).fetchone()[0]

    global_parts  = conn.execute("SELECT COUNT(*) FROM global_participants WHERE user_id=? AND submitted_at IS NOT NULL", (user_id,)).fetchone()[0]
    global_wins   = conn.execute("SELECT COUNT(*) FROM global_participants WHERE user_id=? AND rank=1", (user_id,)).fetchone()[0]
    global_podium = conn.execute("SELECT COUNT(*) FROM global_participants WHERE user_id=? AND rank<=3", (user_id,)).fetchone()[0]

    user_chapters_row = conn.execute("SELECT chapters FROM users WHERE id=?", (user_id,)).fetchone()
    user_chapters = json.loads(user_chapters_row["chapters"] or "[]") if user_chapters_row and user_chapters_row["chapters"] else []
    has_chapter_master = any(
        get_sub_chapter_progress(conn, user_id, ch)[1] > 0 and
        get_sub_chapter_progress(conn, user_id, ch)[0] >= get_sub_chapter_progress(conn, user_id, ch)[1]
        for ch in user_chapters
    )

    weak_turned = conn.execute("""
        SELECT COUNT(DISTINCT q.id) FROM questions q
        WHERE q.user_id=?
          AND EXISTS (SELECT 1 FROM test_attempts ta WHERE ta.question_id=q.id AND ta.user_id=? AND ta.is_correct=0)
          AND EXISTS (SELECT 1 FROM test_attempts ta2 WHERE ta2.question_id=q.id AND ta2.user_id=? AND ta2.is_correct=1
                      AND ta2.attempted_at > (SELECT MAX(ta3.attempted_at) FROM test_attempts ta3
                                              WHERE ta3.question_id=q.id AND ta3.user_id=? AND ta3.is_correct=0))
    """, (user_id, user_id, user_id, user_id)).fetchone()[0]

    streak = get_streak(conn, user_id)
    hour     = datetime.now().hour
    score    = context.get("score", 0)
    total    = context.get("total", 1)
    pct      = score / max(total, 1) * 100
    prev_pct = context.get("prev_pct", 0)
    mode     = context.get("mode", "")
    dyn_total= context.get("dyn_total", 0)
    consec   = conn.execute("SELECT consecutive_correct FROM users WHERE id=?", (user_id,)).fetchone()
    consec_n = consec["consecutive_correct"] if consec else 0

    lat_stats = compute_latency_stats(conn, user_id, limit_days=30)
    is_speed_solver = (lat_stats["avg_time"] < LATENCY_FAST_THRESHOLD and lat_stats["total_tracked"] >= 10)
    is_time_improver = (lat_stats["trend"] == "improving" and lat_stats["prev_avg"] > 0 and
                        (lat_stats["prev_avg"] - lat_stats["recent_avg"]) / lat_stats["prev_avg"] >= 0.2)

    checks = [
        ("first_blood",        total_att >= 1),
        ("century",            total_att >= 100),
        ("thousand_questions", total_att >= 1000),
        ("accuracy_ace",       pct >= 90 and total >= 5),
        ("perfectionist",      pct >= 100 and total >= 5),
        ("week_warrior",       streak >= 7),
        ("comeback_king",      pct - prev_pct >= 20 and prev_pct > 0),
        ("weak_slayer",        mode == "weak" and pct >= 80),
        ("night_owl",          hour >= 22),
        ("early_bird",         hour < 7),
        ("grind_mode",         tests_done >= 10),
        ("persistent",         weak_done >= 5),
        ("dynamic_debut",      dyn_done >= 1),
        ("dynamic_marathon",   mode == "dynamic" and dyn_total >= 50),
        ("dynamic_ace",        mode == "dynamic" and pct >= 80 and total >= 10),
        ("real_exam_ready",    mode == "dynamic" and pct >= 80 and total >= 20),
        ("chapter_completed",  context.get("chapter_completed", False)),
        ("chapter_master",     has_chapter_master),
        ("coach_scholar",      coach_count >= 10),
        ("weak_master",        weak_turned >= 3),
        ("teach_me_mode",      teach_done >= 5),
        ("mood_tracker",       mood_days >= 7),
        ("three_chapters",     chapters_done >= 3),
        ("hot_streak",         consec_n >= 5),
        ("coin_collector",     total_coins_ever >= 100),
        ("bookworm",           bookmark_cnt >= 10),
        ("challenge_champion", challenge_cnt >= 7),
        ("perfect_streak_5",   consec_n >= 5),
        ("perfect_streak_10",  consec_n >= 10),
        ("speed_solver",       is_speed_solver),
        ("time_improver",      is_time_improver),
        ("global_debut",       global_parts >= 1),
        ("global_podium",      global_podium >= 1),
        ("global_champion",    global_wins >= 1),
        ("interview_debut",    interview_done >= 1),
        ("interview_ace",      interview_ace_cnt >= 1),
        ("interview_veteran",  interview_done >= 10),
    ]

    for ach_id, condition in checks:
        if condition and ach_id not in earned_ids:
            ach_def = next((a for a in ACHIEVEMENTS if a["id"] == ach_id), None)
            if ach_def:
                ach_entry = {**ach_def, "earned_at": datetime.now().isoformat()}
                earned.append(ach_entry)
                earned_ids.add(ach_id)
                new_achievements.append(ach_entry)
                award_xp(conn, user_id, ach_def["xp"], f"achievement:{ach_id}")

    conn.execute("UPDATE users SET achievements=? WHERE id=?", (json.dumps(earned), user_id))
    conn.commit()
    return new_achievements

# ── COACHING TRIGGER HELPER ───────────────────────────────────────────────────

def build_coaching_trigger(question_data: dict, user_answer: int) -> dict:
    opts = question_data.get("options", [])
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            opts = []
    return {
        "should_coach": True,
        "question_id": question_data.get("id"),
        "question_text": question_data.get("question", ""),
        "options": opts,
        "correct_answer": question_data.get("correct_answer"),
        "user_answer": user_answer,
        "explanation": question_data.get("explanation", ""),
        "chapter": question_data.get("chapter", ""),
        "message": "You got this wrong — your AI coach can help you understand it now!",
    }

# ── EMAIL ─────────────────────────────────────────────────────────────────────

def send_email(to_email: str, subject: str, html_body: str):
    if not SMTP_USER or not SMTP_PASS:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"ExamAI <{SMTP_USER}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception as exc:
        logger.error("Email error: %s", exc)
        return False

# ── SCHEDULER ─────────────────────────────────────────────────────────────────

def job_morning_reminders():
    conn  = get_db()
    users = conn.execute("SELECT name, email FROM users WHERE receive_reminders=1 AND setup_done=1").fetchall()
    conn.close()
    for u in users:
        html = f"""<div style="font-family:sans-serif;max-width:520px;margin:0 auto;background:#050a14;color:#d0eeff;padding:2rem;border-radius:12px">
          <h2 style="color:#00e5ff">Good Morning, {u['name']}! 🌅</h2>
          <p>Your daily test is ready. Stay consistent — champions are built daily.</p>
          <a href="{APP_URL}" style="display:inline-block;background:linear-gradient(135deg,#00e5ff,#00b8d4);color:#000;font-weight:700;padding:.85rem 2rem;border-radius:100px;text-decoration:none">⚡ Start Today's Test</a>
        </div>"""
        send_email(u["email"], "Your Daily ExamAI Test is Ready!", html)

def job_evening_results():
    conn  = get_db()
    today = date.today().isoformat()
    users = conn.execute("SELECT id, name, email FROM users WHERE receive_reminders=1 AND setup_done=1").fetchall()
    for u in users:
        test = conn.execute(
            "SELECT score, total FROM daily_tests WHERE user_id=? AND test_date=? AND completed=1",
            (u["id"], today)).fetchone()
        sc  = test["score"] if test else 0
        tot = test["total"] if test else 0
        pct = round(sc / max(tot, 1) * 100, 1) if tot else 0
        color = "#00ff88" if pct >= 75 else "#ffd700" if pct >= 50 else "#ff2d6e"
        html = f"""<div style="font-family:sans-serif;max-width:520px;margin:0 auto;background:#050a14;color:#d0eeff;padding:2rem;border-radius:12px">
          <h2 style="color:#00e5ff">Daily Results — {u['name']}</h2>
          <div style="text-align:center;padding:1.5rem;background:#0a2035;border-radius:10px;margin:1rem 0">
            <div style="font-size:3rem;font-weight:900;color:{color}">{pct}%</div>
            <div style="color:#6a9ab8">{sc} / {tot} correct</div>
          </div>
          <a href="{APP_URL}" style="display:inline-block;background:linear-gradient(135deg,#ffd700,#cc9900);color:#000;font-weight:700;padding:.75rem 1.75rem;border-radius:100px;text-decoration:none">View Analytics →</a>
        </div>"""
        send_email(u["email"], f"ExamAI Results — {pct}% today", html)
    conn.close()

def job_compress_coach_memory():
    conn  = get_db()
    users = conn.execute("SELECT id, name, exam_type FROM users WHERE setup_done=1").fetchall()
    for u in users:
        _compress_coach_memory(conn, u["id"], u["name"], u["exam_type"] or "exam")
    conn.close()

if HAS_SCHEDULER:
    scheduler = BackgroundScheduler(timezone=TZ)
    scheduler.add_job(job_morning_reminders,    CronTrigger(hour=8,  minute=0, timezone=TZ))
    scheduler.add_job(job_evening_results,       CronTrigger(hour=20, minute=0, timezone=TZ))
    scheduler.add_job(job_compress_coach_memory, CronTrigger(hour=2,  minute=0, timezone=TZ))
    scheduler.start()

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────

class RegisterReq(BaseModel):
    name: str; email: str; password: str

class LoginReq(BaseModel):
    email: str; password: str

class UpdateProfileReq(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    avatar_color: Optional[str] = None
    receive_reminders: Optional[bool] = None

class ExamSetupReq(BaseModel):
    exam_type: str
    exam_goal: str = ""
    custom_chapters: Optional[List[str]] = None

class GenQuestionsReq(BaseModel):
    chapter: str; sub_chapter: Optional[str] = None; count: int = 10

class SubmitTestReq(BaseModel):
    test_id: int; answers: dict; time_taken: dict = {}; session_type: str = "daily"

class WeakSessionSubmitReq(BaseModel):
    session_id: int; answers: dict; time_taken: dict = {}

class ExplainReq(BaseModel):
    question_id: int; user_answer: int

class LearnReq(BaseModel):
    chapter: str; sub_chapter: str

class NotifReq(BaseModel):
    receive_reminders: bool

class ChapterPracticeReq(BaseModel):
    answers: dict; time_taken: dict = {}; session_type: str = "chapter"

class MoodCheckinReq(BaseModel):
    mood: int; energy: int; note: str = ""

class CoachChatReq(BaseModel):
    message: str
    context: dict = {}
    question_context: Optional[dict] = None

class TeachMeReq(BaseModel):
    question_id: Optional[int] = None
    question_text: Optional[str] = None
    options: Optional[list] = None
    correct_answer: Optional[int] = None
    user_answer: Optional[int] = None
    explanation: Optional[str] = None
    chapter: Optional[str] = None

class TeachMeReplyReq(BaseModel):
    session_id: int
    message: str

class DynamicStartReq(BaseModel):
    chapter: Optional[str] = None

class DynamicAnswerReq(BaseModel):
    attempt_id: int
    user_answer: int
    time_taken: int = 0

class DynamicStopReq(BaseModel):
    session_id: int

class StudyPlanReq(BaseModel):
    exam_date: Optional[str] = None
    daily_hours: Optional[float] = 2.0
    focus_chapters: Optional[list] = None

class BookmarkReq(BaseModel):
    question_id: int
    note: str = ""

class ReportQuestionReq(BaseModel):
    question_id: int
    reason: str

class UseHintReq(BaseModel):
    question_id: int
    hint_type: str

class DailyChallengeSubmitReq(BaseModel):
    user_answer: int

class GlobalTestJoinReq(BaseModel):
    global_test_id: int

class GlobalTestSubmitReq(BaseModel):
    global_test_id: int
    answers: dict
    time_taken_sec: int = 0

class LatencyCoachReq(BaseModel):
    chapter: Optional[str] = None

# Interview Mode models
class InterviewStartReq(BaseModel):
    chapter: Optional[str] = None
    interview_type: str = "conceptual"
    total_questions: int = 10

class InterviewAnswerReq(BaseModel):
    session_id: int
    qa_id: int
    user_answer: str
    time_taken: int = 0

class InterviewEndReq(BaseModel):
    session_id: int

# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(req: RegisterReq):
    if not req.name or not req.name.strip():
        raise HTTPException(400, "Name is required")
    if not req.email or "@" not in req.email:
        raise HTTPException(400, "Valid email is required")
    if not req.password or len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (name,email,password_hash) VALUES (?,?,?)",
                     (req.name.strip(), req.email.lower().strip(), hash_password(req.password)))
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        token   = secrets.token_hex(32)
        conn.execute("INSERT INTO sessions (user_id,token) VALUES (?,?)", (user_id, token))
        conn.commit()
        return {"token": token, "user": {"id": user_id, "name": req.name.strip(),
                "email": req.email.lower().strip(), "setup_done": False, "xp": 0, "level": 1,
                "coins": 0, "avatar_color": "#00e5ff"}}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")
    finally:
        conn.close()

@app.post("/api/auth/login")
def login(req: LoginReq):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=? AND password_hash=?",
                        (req.email.lower().strip(), hash_password(req.password))).fetchone()
    if not user:
        conn.close()
        raise HTTPException(401, "Invalid email or password")
    user  = dict(user)
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO sessions (user_id,token) VALUES (?,?)", (user["id"], token))
    conn.commit(); conn.close()
    return {"token": token, "user": {
        "id": user["id"], "name": user["name"], "email": user["email"],
        "exam_type": user["exam_type"], "exam_language": user.get("exam_language"),
        "setup_done": bool(user["setup_done"]),
        "receive_reminders": bool(user["receive_reminders"]),
        "xp": user.get("xp", 0), "level": user.get("level", 1),
        "coins": user.get("coins", 0), "avatar_color": user.get("avatar_color", "#00e5ff"),
        "bio": user.get("bio", ""),
        "achievements": json.loads(user.get("achievements") or "[]"),
    }}

@app.post("/api/auth/logout")
def logout(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return {"success": True}
    token = authorization.split(" ", 1)[1].strip()
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit(); conn.close()
    return {"success": True, "message": "Logged out"}

# ── USER PROFILE ──────────────────────────────────────────────────────────────

@app.get("/api/user/profile")
def profile(u=Depends(get_current_user)):
    conn = get_db()
    uid = u["uid"]
    streak = get_streak(conn, uid)
    total_att = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=?", (uid,)).fetchone()[0]
    total_cor = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=? AND is_correct=1", (uid,)).fetchone()[0]
    acc = round(total_cor / max(total_att, 1) * 100, 1) if total_att else 0
    best_day = conn.execute("""
        SELECT test_date, score, total,
               ROUND(CAST(score AS FLOAT)/MAX(total,1)*100, 1) as pct
        FROM daily_tests WHERE user_id=? AND completed=1
        ORDER BY pct DESC LIMIT 1
    """, (uid,)).fetchone()
    member_since = conn.execute("SELECT created_at FROM users WHERE id=?", (uid,)).fetchone()
    bookmarks = conn.execute("SELECT COUNT(*) FROM bookmarks WHERE user_id=?", (uid,)).fetchone()[0]
    chapters_done = conn.execute("SELECT COUNT(*) FROM chapter_completions WHERE user_id=?", (uid,)).fetchone()[0]
    interview_done = conn.execute("SELECT COUNT(*) FROM interview_sessions WHERE user_id=? AND is_active=0", (uid,)).fetchone()[0]
    xp = u.get("xp", 0)
    level = u.get("level", 1)
    xp_for_next = XP_LEVELS[min(level, len(XP_LEVELS)-1)] if level < len(XP_LEVELS) else None
    xp_prev = XP_LEVELS[max(0, level-1)]
    xp_progress = round((xp - xp_prev) / max(1, (xp_for_next or xp+1) - xp_prev) * 100, 1) if xp_for_next else 100
    global_tests_done = conn.execute(
        "SELECT COUNT(*) FROM global_participants WHERE user_id=? AND submitted_at IS NOT NULL", (uid,)).fetchone()[0]
    best_global_rank = conn.execute(
        "SELECT MIN(rank) FROM global_participants WHERE user_id=? AND rank IS NOT NULL", (uid,)).fetchone()[0]
    challenge_streak = 0
    ch_hist = conn.execute(
        "SELECT challenge_date FROM daily_challenges WHERE user_id=? AND completed=1 ORDER BY challenge_date DESC",
        (uid,)).fetchall()
    if ch_hist:
        check = date.today()
        for r in ch_hist:
            try:
                d = date.fromisoformat(r["challenge_date"])
                if d == check or d == check - timedelta(1):
                    challenge_streak += 1; check = d - timedelta(1)
                else:
                    break
            except Exception:
                break
    lat_stats = compute_latency_stats(conn, uid, 30)
    conn.close()
    return {
        "id": uid, "name": u["name"], "email": u["email"],
        "exam_type": u["exam_type"], "exam_goal": u.get("exam_goal"),
        "exam_language": u.get("exam_language"),
        "setup_done": bool(u["setup_done"]),
        "receive_reminders": bool(u["receive_reminders"]),
        "chapters": json.loads(u["chapters"]) if u["chapters"] else [],
        "xp": xp, "level": level, "xp_progress": xp_progress, "xp_for_next": xp_for_next,
        "achievements": json.loads(u.get("achievements") or "[]"),
        "consecutive_correct": u.get("consecutive_correct", 0),
        "coins": u.get("coins", 0),
        "avatar_color": u.get("avatar_color", "#00e5ff"),
        "bio": u.get("bio", ""),
        "stats": {
            "total_attempted": total_att, "total_correct": total_cor, "overall_accuracy": acc,
            "study_streak": streak, "challenge_streak": challenge_streak,
            "bookmarks": bookmarks, "chapters_completed": chapters_done,
            "interview_sessions": interview_done,
            "best_day": dict(best_day) if best_day else None,
            "global_tests_completed": global_tests_done, "best_global_rank": best_global_rank,
            "avg_decision_time_sec": lat_stats.get("avg_time", 0),
        },
        "member_since": member_since["created_at"] if member_since else None,
    }

@app.put("/api/user/profile")
def update_profile(req: UpdateProfileReq, u=Depends(get_current_user)):
    conn = get_db()
    updates, values = [], []
    if req.name is not None and req.name.strip():
        updates.append("name=?"); values.append(req.name.strip()[:80])
    if req.bio is not None:
        updates.append("bio=?"); values.append(req.bio[:200])
    if req.avatar_color is not None:
        color = req.avatar_color.strip()
        if color.startswith("#") and len(color) in (4, 7):
            updates.append("avatar_color=?"); values.append(color)
    if req.receive_reminders is not None:
        updates.append("receive_reminders=?"); values.append(1 if req.receive_reminders else 0)
    if updates:
        values.append(u["uid"])
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", values)
        conn.commit()
    conn.close()
    return {"success": True, "message": "Profile updated"}

@app.post("/api/user/notifications")
def set_notifications(req: NotifReq, u=Depends(get_current_user)):
    conn = get_db()
    conn.execute("UPDATE users SET receive_reminders=? WHERE id=?",
                 (1 if req.receive_reminders else 0, u["uid"]))
    conn.commit(); conn.close()
    return {"receive_reminders": req.receive_reminders}

# ── EXAM SETUP ────────────────────────────────────────────────────────────────

@app.post("/api/exam/setup")
def setup_exam(req: ExamSetupReq, u=Depends(get_current_user)):
    exam_type = (req.exam_type or "").strip()
    if not exam_type:
        raise HTTPException(400, "exam_type is required")

    # Detect language for multilingual exams
    lang_key = detect_exam_language(exam_type, req.exam_goal or "")
    lang_instruction = get_language_instruction(lang_key)
    lang_chapter_hint = get_language_chapter_hint(lang_key, exam_type)

    if req.custom_chapters and isinstance(req.custom_chapters, list):
        chapters = [str(c).strip() for c in req.custom_chapters if str(c).strip()]
        if len(chapters) < 3:
            raise HTTPException(400, "Please provide at least 3 chapters")
        chapters = chapters[:15]
    else:
        prompt = f"""You are an expert exam coach. The student is preparing for: {exam_type}
{("Student's goal: " + req.exam_goal) if req.exam_goal else ""}
{lang_chapter_hint}

Generate exactly 12 ordered main chapters/topics for this exam syllabus.

RULES:
- Make them SPECIFIC to the {exam_type} syllabus and exam format
- Order from foundational to advanced
- If this is a language exam, include chapters for writing systems, vocabulary themes, grammar, reading, etc.
- Be precise — no vague chapters like "Miscellaneous"

Return ONLY this JSON, no extra text:
{{"chapters": ["Chapter 1 name", "Chapter 2 name", "Chapter 3 name", ..., "Chapter 12 name"]}}"""

        try:
            raw = groq_chat(prompt, temperature=0.4)
            data = parse_json(raw)
            chapters = data.get("chapters", data) if isinstance(data, dict) else data
            if not isinstance(chapters, list):
                chapters = []
            chapters = [str(c).strip() for c in chapters if str(c).strip()][:12]
        except Exception as exc:
            logger.error("Chapter generation failed: %s", exc)
            chapters = []

        if len(chapters) < 3:
            logger.warning("AI returned insufficient chapters for '%s', using fallback", exam_type)
            if lang_key == "japanese":
                chapters = [
                    "Hiragana & Katakana (ひらがな・カタカナ)",
                    "Basic Vocabulary (基本語彙) — Numbers, Time, Colors",
                    "Nouns & Pronouns (名詞・代名詞)",
                    "Core Grammar Patterns (文法)",
                    "Particles (助詞) — は, が, を, に, で",
                    "Verbs (動詞) — Groups & Conjugation",
                    "Adjectives (形容詞) — い・な adjectives",
                    "Sentence Structure (文構造)",
                    "Daily Expressions (日常表現)",
                    "Reading Comprehension (読解)",
                    "Listening Practice (聴解)",
                    "Kanji (漢字) — Key Characters",
                ]
            else:
                chapters = [
                    f"{exam_type} — Fundamentals",
                    f"{exam_type} — Core Concepts",
                    f"{exam_type} — Intermediate Topics",
                    f"{exam_type} — Advanced Concepts",
                    f"{exam_type} — Problem Solving",
                    f"{exam_type} — Applications",
                    f"{exam_type} — Case Studies",
                    f"{exam_type} — Practice & Revision",
                ]

    conn = get_db()
    conn.execute(
        "UPDATE users SET exam_type=?,exam_goal=?,exam_language=?,chapters=?,setup_done=1,daily_test_chapter_idx=0 WHERE id=?",
        (exam_type, req.exam_goal, lang_key, json.dumps(chapters), u["uid"]))
    conn.commit()
    award_xp(conn, u["uid"], 50, "setup_complete")
    conn.close()
    return {
        "chapters": chapters,
        "exam_type": exam_type,
        "detected_language": lang_key,
        "language_learning_mode": lang_key is not None,
    }

# ── CHAPTERS ──────────────────────────────────────────────────────────────────

@app.get("/api/exam/chapters")
def get_chapters(u=Depends(get_current_user)):
    conn = get_db()
    row  = conn.execute("SELECT chapters FROM users WHERE id=?", (u["uid"],)).fetchone()
    if not row or not row["chapters"]:
        conn.close()
        return {"chapters": []}
    chapters = json.loads(row["chapters"])
    result   = []
    for i, ch in enumerate(chapters):
        att, cor, acc    = chapter_stats(conn, u["uid"], ch)
        unlocked         = is_unlocked(conn, u["uid"], chapters, i)
        complete         = is_chapter_complete(conn, u["uid"], chapters, i)
        total_qs         = conn.execute(
            "SELECT COUNT(DISTINCT id) FROM questions WHERE user_id=? AND chapter=?",
            (u["uid"], ch)).fetchone()[0]
        subs_p, subs_t   = get_sub_chapter_progress(conn, u["uid"], ch)
        if complete:
            conn.execute("INSERT OR IGNORE INTO chapter_completions (user_id, chapter) VALUES (?,?)",
                         (u["uid"], ch))
            conn.commit()
        result.append({
            "index": i, "name": ch, "total_questions": total_qs,
            "attempted": att, "correct": cor, "accuracy": acc,
            "unlocked": unlocked, "complete": complete,
            "subs_practiced": subs_p, "subs_total": subs_t,
            "unlock_threshold_attempts": UNLOCK_REQUIRED_ATTEMPTS,
            "unlock_threshold_accuracy": UNLOCK_REQUIRED_ACCURACY,
            "status": ("complete" if complete else
                       "weak" if acc < 50 and att > 0 else
                       "moderate" if acc < 75 and att > 0 else
                       "strong" if att > 0 else "untouched"),
        })
    conn.close()
    return {"chapters": result}

# ── STRUCTURED CHAPTER LESSON ──────────────────────────────────────────────────

@app.get("/api/chapters/{chapter_name}/lesson")
def get_chapter_lesson(chapter_name: str, refresh: bool = False, u=Depends(get_current_user)):
    conn = get_db()
    if not refresh:
        cached = conn.execute(
            "SELECT content, structured FROM chapter_lessons WHERE user_id=? AND chapter=?",
            (u["uid"], chapter_name)).fetchone()
        if cached:
            conn.close()
            structured = None
            if cached["structured"]:
                try:
                    structured = json.loads(cached["structured"])
                except Exception:
                    pass
            return {"chapter": chapter_name, "content": cached["content"],
                    "structured": structured, "cached": True}

    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)

    subs = conn.execute(
        "SELECT name FROM sub_chapters WHERE user_id=? AND parent_chapter=? ORDER BY order_index",
        (u["uid"], chapter_name)).fetchall()
    conn.close()

    sub_names   = [s["name"] for s in subs] if subs else []
    web_ctx     = search_topic_context(chapter_name, exam_type, deep=True)
    ctx_block   = f'\nREFERENCE MATERIAL:\n"""\n{web_ctx[:4000]}\n"""\n' if web_ctx else ""
    sub_block   = ("\nThis chapter covers sub-topics: " + ", ".join(sub_names) + ".\n") if sub_names else ""
    style_hints = get_exam_style_hints(exam_type)

    # Structured JSON lesson prompt
    prompt = f"""You are a world-class exam tutor for {exam_type}.{lang_instruction}{ctx_block}{sub_block}

Create a COMPREHENSIVE STRUCTURED LESSON for the chapter: "{chapter_name}"

Return ONLY this JSON structure (no markdown wrapper, no extra text):
{{
  "overview": {{
    "summary": "2-3 paragraph overview of this chapter and why it matters for {exam_type}",
    "importance": "Why this chapter is crucial for the exam (1-2 sentences)",
    "prerequisite_knowledge": ["prerequisite 1", "prerequisite 2"]
  }},
  "learning_objectives": [
    "By the end of this chapter, you will be able to: ...",
    "Understand ...",
    "Apply ...",
    "Analyze ..."
  ],
  "core_concepts": [
    {{
      "title": "Concept Name",
      "explanation": "Clear 2-3 sentence explanation",
      "example": "Concrete example or analogy",
      "exam_relevance": "How this appears in exam questions"
    }}
  ],
  "key_facts": [
    "Important fact or data point 1",
    "Important fact or data point 2",
    "Important fact or data point 3",
    "Important fact or data point 4",
    "Important fact or data point 5"
  ],
  "common_mistakes": [
    {{
      "mistake": "What students often get wrong",
      "correction": "The correct understanding"
    }}
  ],
  "memory_tricks": [
    {{
      "trick": "Mnemonic or memory device",
      "what_it_covers": "What this helps you remember"
    }}
  ],
  "exam_strategy": {{
    "question_types": {json.dumps(style_hints.split(chr(10))[:4])},
    "time_allocation": "How to approach time management for this chapter",
    "high_priority_topics": ["Topic A", "Topic B", "Topic C"],
    "traps_to_avoid": ["Common trap 1", "Common trap 2"]
  }},
  "sub_topic_roadmap": {json.dumps([{"name": s, "focus": "Key area to master"} for s in (sub_names or ["Core concepts", "Applications", "Practice"])])},
  "connections": {{
    "related_chapters": ["Chapter it connects to 1", "Chapter 2"],
    "real_world_applications": ["Application 1", "Application 2"]
  }},
  "quick_self_check": [
    {{
      "question": "Can you explain [key concept] in your own words?",
      "what_to_check": "Key points your answer should include"
    }}
  ]
}}

RULES:
- Include AT LEAST 4 core_concepts
- Include AT LEAST 3 common_mistakes  
- Include AT LEAST 2 memory_tricks
- For language exams: include actual {lang_key or 'target'} language text in examples
- Be specific and exam-focused, not generic
- Return ONLY valid JSON"""

    try:
        raw = groq_chat(prompt, temperature=0.5, json_mode=True)
        structured = parse_json(raw)
        # Build flat content string from structured data for backward compatibility
        content_parts = []
        if "overview" in structured:
            content_parts.append(f"### 🗺️ Chapter Overview\n{structured['overview'].get('summary','')}")
        if "learning_objectives" in structured:
            objs = "\n".join(f"- {o}" for o in structured["learning_objectives"])
            content_parts.append(f"### 🎯 Learning Objectives\n{objs}")
        if "core_concepts" in structured:
            cc_parts = []
            for cc in structured["core_concepts"]:
                cc_parts.append(f"**{cc['title']}**: {cc['explanation']}\n*Example*: {cc.get('example','')}")
            content_parts.append("### 🧱 Core Concepts\n" + "\n\n".join(cc_parts))
        if "key_facts" in structured:
            facts = "\n".join(f"- {f}" for f in structured["key_facts"])
            content_parts.append(f"### 📊 Key Facts\n{facts}")
        if "exam_strategy" in structured:
            es = structured["exam_strategy"]
            tips = "\n".join(f"- {t}" for t in es.get("traps_to_avoid", []))
            content_parts.append(f"### ⚡ Exam Strategy\n{tips}")
        if "memory_tricks" in structured:
            mt_parts = [f"**{mt['trick']}**: {mt.get('what_it_covers','')}" for mt in structured["memory_tricks"]]
            content_parts.append("### 🧠 Memory Tricks\n" + "\n".join(mt_parts))
        content = "\n\n".join(content_parts)
    except Exception as exc:
        logger.error("Structured lesson generation failed, falling back to text: %s", exc)
        structured = None
        # Text fallback
        text_prompt = f"""You are a world-class exam tutor for {exam_type}.{lang_instruction}{ctx_block}{sub_block}
Create a comprehensive lesson for "{chapter_name}".
Use these section headers with ###:
### 🗺️ Chapter Overview
### 🎯 Learning Objectives
### 🧱 Core Concepts
### 📊 Key Facts & Figures
### ⚡ Exam Strategy
### 🧠 Memory Tricks
### ⚠️ Common Mistakes
### 🔗 Connections
700-900 words. Plain text with headers ONLY."""
        content = groq_chat(text_prompt, temperature=0.5, json_mode=False)

    conn3 = get_db()
    conn3.execute(
        "INSERT OR REPLACE INTO chapter_lessons (user_id, chapter, content, structured) VALUES (?,?,?,?)",
        (u["uid"], chapter_name, content, json.dumps(structured) if structured else None))
    conn3.commit(); conn3.close()

    return {
        "chapter": chapter_name, "content": content, "structured": structured,
        "cached": False, "used_web": bool(web_ctx),
        "language": lang_key,
    }

# ── SUB-CHAPTERS ──────────────────────────────────────────────────────────────

@app.get("/api/chapters/{chapter_name}/sub-chapters")
def get_sub_chapters(chapter_name: str, u=Depends(get_current_user)):
    conn = get_db()
    subs = conn.execute(
        "SELECT * FROM sub_chapters WHERE user_id=? AND parent_chapter=? ORDER BY order_index",
        (u["uid"], chapter_name)).fetchall()

    if not subs:
        exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
        exam_type = exam_row["exam_type"] if exam_row else "General"
        lang_key  = exam_row["exam_language"] if exam_row else None
        lang_instruction = get_language_instruction(lang_key)

        prompt = (
            f'For the {exam_type} exam chapter "{chapter_name}", '
            f'create exactly {SUBCHAPTERS_PER_CHAPTER} focused sub-topics.\n'
            f'{lang_instruction}\n'
            f'Sub-topics should be specific, testable areas within this chapter.\n'
            f'Return ONLY this JSON:\n'
            f'{{"sub_chapters": ["Sub-topic 1", "Sub-topic 2", ..., "Sub-topic {SUBCHAPTERS_PER_CHAPTER}"]}}\n'
        )
        try:
            raw = groq_chat(prompt, temperature=0.4)
            data = parse_json(raw)
            sub_names = data.get("sub_chapters", data) if isinstance(data, dict) else data
            if not isinstance(sub_names, list) or len(sub_names) < 1:
                sub_names = [f"{chapter_name} — Part {i + 1}" for i in range(SUBCHAPTERS_PER_CHAPTER)]
        except Exception as exc:
            logger.error("Sub-chapter generation failed: %s", exc)
            sub_names = [f"{chapter_name} — Part {i + 1}" for i in range(SUBCHAPTERS_PER_CHAPTER)]

        for i, name in enumerate(sub_names[:SUBCHAPTERS_PER_CHAPTER]):
            conn.execute(
                "INSERT INTO sub_chapters (user_id,parent_chapter,name,order_index) VALUES (?,?,?,?)",
                (u["uid"], chapter_name, str(name).strip(), i))
        conn.commit()
        subs = conn.execute(
            "SELECT * FROM sub_chapters WHERE user_id=? AND parent_chapter=? ORDER BY order_index",
            (u["uid"], chapter_name)).fetchall()

    result = []
    for sub in subs:
        sq = conn.execute("""
            SELECT COUNT(DISTINCT q.id) AS total,
                   COUNT(DISTINCT CASE WHEN ta.id IS NOT NULL THEN q.id END) AS attempted,
                   COUNT(ta.id) AS total_attempts,
                   SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END) AS correct_attempts
            FROM questions q
            LEFT JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
            WHERE q.user_id=? AND q.sub_chapter=?
        """, (u["uid"], u["uid"], sub["name"])).fetchone()
        att     = sq["attempted"]        or 0 if sq else 0
        tot     = sq["total"]            or 0 if sq else 0
        tot_att = sq["total_attempts"]   or 0 if sq else 0
        cor_att = sq["correct_attempts"] or 0 if sq else 0
        acc     = round(cor_att / max(tot_att, 1) * 100, 1) if tot_att > 0 else 0
        practiced = tot_att >= REQUIRED_SUB_ATTEMPTS
        result.append({
            "id": sub["id"], "name": sub["name"], "order_index": sub["order_index"],
            "total_questions": tot, "attempted": att, "accuracy": acc,
            "total_attempts": tot_att, "practiced": practiced,
            "required_attempts": REQUIRED_SUB_ATTEMPTS,
            "can_generate_more": tot < MAX_Q_PER_SUBCHAPTER,
            "max_questions": MAX_Q_PER_SUBCHAPTER,
        })
    conn.close()
    return {"sub_chapters": result, "chapter": chapter_name,
            "required_sub_attempts": REQUIRED_SUB_ATTEMPTS,
            "total_sub_chapters": SUBCHAPTERS_PER_CHAPTER,
            "max_questions_per_sub": MAX_Q_PER_SUBCHAPTER}

# ── CHAPTER COMPLETION ────────────────────────────────────────────────────────

@app.get("/api/chapters/{chapter_name}/completion")
def check_chapter_completion(chapter_name: str, u=Depends(get_current_user)):
    conn = get_db()
    row  = conn.execute("SELECT chapters FROM users WHERE id=?", (u["uid"],)).fetchone()
    if not row or not row["chapters"]:
        conn.close()
        return {"complete": False, "next_unlocked": False}
    chapters = json.loads(row["chapters"])
    try:
        idx = chapters.index(chapter_name)
    except ValueError:
        conn.close()
        raise HTTPException(404, "Chapter not found")
    att, cor, acc  = chapter_stats(conn, u["uid"], chapter_name)
    subs_p, subs_t = get_sub_chapter_progress(conn, u["uid"], chapter_name)
    complete       = is_chapter_complete(conn, u["uid"], chapters, idx)
    next_unlocked  = (idx + 1 < len(chapters) and is_unlocked(conn, u["uid"], chapters, idx + 1))
    next_chapter   = chapters[idx + 1] if idx + 1 < len(chapters) else None
    new_achs = []
    if complete:
        conn.execute("INSERT OR IGNORE INTO chapter_completions (user_id, chapter) VALUES (?,?)",
            (u["uid"], chapter_name))
        conn.commit()
        current_idx = u.get("daily_test_chapter_idx", 0) or 0
        if idx == current_idx and idx + 1 < len(chapters):
            conn.execute("UPDATE users SET daily_test_chapter_idx=? WHERE id=?", (idx + 1, u["uid"]))
            conn.commit()
        new_achs = check_and_award_achievements(conn, u["uid"], {"chapter_completed": True, "mode": "completion"})
    conn.close()
    return {
        "chapter": chapter_name, "complete": complete,
        "attempted": att, "accuracy": acc,
        "subs_practiced": subs_p, "subs_total": subs_t,
        "required_attempts": UNLOCK_REQUIRED_ATTEMPTS,
        "required_accuracy": UNLOCK_REQUIRED_ACCURACY,
        "next_chapter": next_chapter, "next_unlocked": next_unlocked,
        "new_achievements": new_achs,
    }

# ── QUESTION GENERATION ───────────────────────────────────────────────────────

@app.post("/api/questions/generate")
def generate_questions(req: GenQuestionsReq, u=Depends(get_current_user)):
    conn      = get_db()
    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)

    filter_clause = "AND sub_chapter=?" if req.sub_chapter else "AND (sub_chapter IS NULL OR sub_chapter=?)"
    filter_val    = req.sub_chapter or ""
    existing_cnt = conn.execute(
        f"SELECT COUNT(*) as cnt FROM questions WHERE user_id=? AND chapter=? {filter_clause}",
        (u["uid"], req.chapter, filter_val)).fetchone()["cnt"]

    if existing_cnt >= MAX_Q_PER_SUBCHAPTER:
        qs = conn.execute(
            f"SELECT * FROM questions WHERE user_id=? AND chapter=? {filter_clause} ORDER BY RANDOM() LIMIT ?",
            (u["uid"], req.chapter, filter_val, req.count)).fetchall()
        conn.close()
        return {"questions": [fmt_q(q) for q in qs], "generated": False,
                "total_available": existing_cnt, "max_per_subchapter": MAX_Q_PER_SUBCHAPTER}

    need        = min(req.count, MAX_Q_PER_SUBCHAPTER - existing_cnt, 15)
    topic       = f'{req.sub_chapter} (part of {req.chapter})' if req.sub_chapter else req.chapter
    web_ctx     = search_topic_context(req.sub_chapter or req.chapter, exam_type, deep=True)
    ctx_block   = f'\nREAL-WORLD REFERENCE:\n"""\n{web_ctx[:4000]}\n"""\n' if web_ctx else ""
    style_hints = get_exam_style_hints(exam_type)
    prompt = build_question_prompt(topic, exam_type, need, style_hints, ctx_block,
                                    existing_cnt, lang_instruction)
    raw    = groq_chat(prompt, temperature=0.75)
    data   = parse_json(raw)
    raw_qs = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(raw_qs, list):
        raw_qs = []

    stored = []
    for q in raw_qs:
        vq = validate_question_dict(q)
        if not vq:
            continue
        try:
            conn.execute(
                "INSERT INTO questions (user_id,chapter,sub_chapter,question,options,correct_answer,explanation,difficulty) VALUES (?,?,?,?,?,?,?,?)",
                (u["uid"], req.chapter, req.sub_chapter, vq["question"],
                 json.dumps(vq["options"]), vq["correct_answer"], vq["explanation"], vq["difficulty"]))
            conn.commit()
            qid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            stored.append({"id": qid, "chapter": req.chapter, "sub_chapter": req.sub_chapter, **vq})
        except Exception as exc:
            logger.error("Question insert error: %s", exc)

    if len(stored) < req.count and existing_cnt > 0:
        stored_ids = [q["id"] for q in stored] or [0]
        ph = ",".join("?" * len(stored_ids))
        old_qs = conn.execute(
            f"SELECT * FROM questions WHERE user_id=? AND chapter=? {filter_clause} AND id NOT IN ({ph}) ORDER BY RANDOM() LIMIT ?",
            [u["uid"], req.chapter, filter_val] + stored_ids + [req.count - len(stored)]).fetchall()
        stored = [fmt_q(q) for q in old_qs] + stored

    total_now = conn.execute(
        f"SELECT COUNT(*) FROM questions WHERE user_id=? AND chapter=? {filter_clause}",
        (u["uid"], req.chapter, filter_val)).fetchone()[0]
    conn.close()
    return {"questions": stored[:req.count], "generated": True,
            "total_available": total_now,
            "can_generate_more": total_now < MAX_Q_PER_SUBCHAPTER,
            "max_per_subchapter": MAX_Q_PER_SUBCHAPTER,
            "used_web_search": bool(web_ctx),
            "language": lang_key}

# ── LEARN (Sub-chapter lesson) ────────────────────────────────────────────────

@app.post("/api/learn")
def generate_lesson(req: LearnReq, u=Depends(get_current_user)):
    conn      = get_db()
    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)
    conn.close()

    web_ctx     = search_topic_context(req.sub_chapter, exam_type, deep=True)
    ctx_block   = f'\nREAL-WORLD REFERENCE:\n"""\n{web_ctx[:4000]}\n"""\n' if web_ctx else ""
    style_hints = get_exam_style_hints(exam_type)

    prompt = f"""You are a world-class exam tutor for {exam_type}.{lang_instruction}{ctx_block}
Create a structured lesson on "{req.sub_chapter}" (chapter: "{req.chapter}").

Return ONLY this JSON:
{{
  "title": "{req.sub_chapter}",
  "key_concepts": [
    {{"term": "Term", "definition": "Clear definition", "example": "Example"}}
  ],
  "explanation": "Full 3-4 paragraph explanation of this sub-topic",
  "important_points": ["Point 1", "Point 2", "Point 3", "Point 4"],
  "exam_tips": ["Tip 1", "Tip 2", "Tip 3"],
  "memory_trick": "One memorable trick or mnemonic",
  "practice_question": {{
    "question": "A practice question on this topic",
    "answer": "The answer and explanation"
  }},
  "connections": ["How this connects to other topics"]
}}

For language exams: include actual {lang_key or 'target language'} text, vocabulary, and grammar examples.
Return ONLY valid JSON."""

    try:
        raw = groq_chat(prompt, temperature=0.55, json_mode=True)
        structured = parse_json(raw)
        # Build flat content for backward compat
        parts = []
        if "explanation" in structured:
            parts.append(f"### 🎯 Explanation\n{structured['explanation']}")
        if "key_concepts" in structured:
            kc = "\n".join(f"**{c['term']}**: {c['definition']}" for c in structured["key_concepts"])
            parts.append(f"### 📋 Key Concepts\n{kc}")
        if "exam_tips" in structured:
            tips = "\n".join(f"- {t}" for t in structured["exam_tips"])
            parts.append(f"### ⚡ Exam Tips\n{tips}")
        if "memory_trick" in structured:
            parts.append(f"### 🧠 Memory Trick\n{structured['memory_trick']}")
        content = "\n\n".join(parts)
        return {"content": content, "structured": structured,
                "chapter": req.chapter, "sub_chapter": req.sub_chapter,
                "used_web": bool(web_ctx), "language": lang_key}
    except Exception as exc:
        logger.warning("Structured sub-lesson failed, using text: %s", exc)
        text_prompt = f"""You are a world-class exam tutor for {exam_type}.{lang_instruction}{ctx_block}
Create a lesson on "{req.sub_chapter}" (chapter: "{req.chapter}").
Sections: ### 🎯 Key Concepts | ### 📋 Important Facts | ### 📖 Explanation | ### 🧠 Memory Tricks | ### ⚡ Exam Tips | ### 🔗 Connections
600-800 words. Plain text with ### headers."""
        content = groq_chat(text_prompt, temperature=0.5, json_mode=False)
        return {"content": content, "structured": None,
                "chapter": req.chapter, "sub_chapter": req.sub_chapter,
                "used_web": bool(web_ctx), "language": lang_key}

# ── CHAPTER PRACTICE SUBMIT ───────────────────────────────────────────────────

@app.post("/api/test/chapter-practice")
def submit_chapter_practice(req: ChapterPracticeReq, u=Depends(get_current_user)):
    conn  = get_db()
    score = 0
    results = []
    coaching_triggers = []

    for qid_str, ua in req.answers.items():
        try:
            qid = int(qid_str)
        except (ValueError, TypeError):
            continue
        q = conn.execute("SELECT * FROM questions WHERE id=? AND user_id=?", (qid, u["uid"])).fetchone()
        if not q:
            continue
        q = dict(q)
        ua_int     = safe_answer_int(ua)
        time_t     = int(req.time_taken.get(str(qid), 0) or 0)
        is_correct = 1 if ua_int is not None and ua_int == q["correct_answer"] else 0
        if is_correct:
            score += 1
            award_coins(conn, u["uid"], COINS_PER_CORRECT)
        else:
            coaching_triggers.append(build_coaching_trigger(
                {"id": qid, "question": q["question"],
                 "options": json.loads(q["options"]),
                 "correct_answer": q["correct_answer"],
                 "explanation": q["explanation"],
                 "chapter": q["chapter"]}, ua_int if ua_int is not None else -1))
        update_consecutive_correct(conn, u["uid"], bool(is_correct))
        conn.execute(
            "INSERT INTO test_attempts (user_id,question_id,user_answer,is_correct,time_taken,session_type) VALUES (?,?,?,?,?,?)",
            (u["uid"], qid, ua_int, is_correct, time_t, req.session_type or "chapter"))
        if time_t > 0:
            conn.execute(
                "INSERT INTO question_latency (user_id,question_id,session_type,time_taken_sec,is_correct,chapter,difficulty) VALUES (?,?,?,?,?,?,?)",
                (u["uid"], qid, req.session_type or "chapter", time_t, is_correct, q["chapter"], q["difficulty"]))
        results.append({
            "question_id": qid, "question": q["question"],
            "options": json.loads(q["options"]), "correct_answer": q["correct_answer"],
            "user_answer": ua_int, "is_correct": bool(is_correct), "explanation": q["explanation"],
        })
    conn.commit()
    total = len(results)
    pct   = round(score / max(total, 1) * 100, 1)
    xp_earned = score * 5 + (25 if pct >= 75 else 10)
    award_xp(conn, u["uid"], xp_earned, "chapter_practice")
    lat_stats  = compute_latency_stats(conn, u["uid"], 30)
    lat_alerts = get_latency_alerts(lat_stats)
    new_achs = check_and_award_achievements(conn, u["uid"], {"score": score, "total": total, "mode": "chapter"})
    conn.close()
    return {"score": score, "total": total, "percentage": pct, "results": results,
            "saved": True, "xp_earned": xp_earned, "new_achievements": new_achs,
            "coins_earned": score * COINS_PER_CORRECT, "latency_alerts": lat_alerts[:2],
            "coaching_triggers": coaching_triggers,
            "has_wrong_answers": len(coaching_triggers) > 0,
            "first_coaching_trigger": coaching_triggers[0] if coaching_triggers else None}

# ── DAILY TEST ────────────────────────────────────────────────────────────────

@app.get("/api/test/daily")
def daily_test(u=Depends(get_current_user)):
    today = date.today().isoformat()
    conn  = get_db()
    uid   = u["uid"]

    existing = conn.execute(
        "SELECT * FROM daily_tests WHERE user_id=? AND test_date=?", (uid, today)).fetchone()
    if existing:
        existing = dict(existing)
        q_ids    = json.loads(existing["question_ids"])
        qs       = [fmt_q(conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone())
                    for qid in q_ids]
        qs = [q for q in qs if q]
        answers = json.loads(existing["answers"]) if existing.get("answers") else {}
        conn.close()
        total = existing.get("total") or len(qs)
        score = existing.get("score") or 0
        return {"test_id": existing["id"], "completed": bool(existing.get("completed")),
                "score": score, "total": total,
                "percentage": round(score / max(total, 1) * 100, 1),
                "questions": qs, "answers": answers,
                "chapter_name": existing.get("chapter_name")}

    chapters_row = conn.execute("SELECT chapters FROM users WHERE id=?", (uid,)).fetchone()
    chapters = json.loads(chapters_row["chapters"]) if chapters_row and chapters_row["chapters"] else []
    active_chapter, active_idx = get_active_daily_chapter(conn, uid, chapters)

    if not active_chapter:
        conn.close()
        return {"test_id": None, "completed": False, "questions": [],
                "message": "Complete exam setup first to get a daily test."}

    answered_today = set(
        r["question_id"] for r in conn.execute(
            "SELECT DISTINCT question_id FROM test_attempts WHERE user_id=? AND date(attempted_at)=?",
            (uid, today)).fetchall())

    new_q_ids = [r["id"] for r in conn.execute("""
        SELECT id FROM questions WHERE user_id=? AND chapter=?
        AND id NOT IN (SELECT DISTINCT question_id FROM test_attempts WHERE user_id=?)
        ORDER BY RANDOM() LIMIT 15
    """, (uid, active_chapter, uid)).fetchall() if r["id"] not in answered_today]

    weak_ids = [r["id"] for r in conn.execute("""
        SELECT q.id FROM questions q JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
        WHERE q.user_id=? AND q.chapter=?
        GROUP BY q.id HAVING SUM(CASE WHEN ta.is_correct=0 THEN 1 ELSE 0 END) > SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END)
        ORDER BY RANDOM() LIMIT 10
    """, (uid, uid, active_chapter)).fetchall() if r["id"] not in answered_today]

    all_ids = list(dict.fromkeys(new_q_ids + weak_ids))[:20]

    if len(all_ids) < 10:
        extra = [r["id"] for r in conn.execute(
            "SELECT id FROM questions WHERE user_id=? AND chapter=? ORDER BY RANDOM() LIMIT 20",
            (uid, active_chapter)).fetchall()
        if r["id"] not in answered_today and r["id"] not in all_ids]
        all_ids = list(dict.fromkeys(all_ids + extra))[:20]

    if not all_ids:
        conn.close()
        return {"test_id": None, "completed": False, "questions": [],
                "message": f"Generate questions for '{active_chapter}' first.",
                "current_chapter": active_chapter}

    conn.execute(
        "INSERT INTO daily_tests (user_id,test_date,question_ids,total,chapter_name) VALUES (?,?,?,?,?)",
        (uid, today, json.dumps(all_ids), len(all_ids), active_chapter))
    conn.commit()
    test_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    qs = [fmt_q(conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()) for qid in all_ids]
    qs = [q for q in qs if q]
    conn.close()
    return {"test_id": test_id, "completed": False, "questions": qs, "total": len(qs),
            "current_chapter": active_chapter, "chapter_index": active_idx,
            "chapter_name": active_chapter}

@app.post("/api/test/submit")
def submit_test(req: SubmitTestReq, u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]
    coaching_triggers = []

    def _process_answers(q_ids, answers, time_taken, session_type):
        score = 0; results = []
        for qid in q_ids:
            q = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
            if not q: continue
            q = dict(q)
            ua_int     = safe_answer_int(answers.get(str(qid)))
            time_t     = int(time_taken.get(str(qid), 0) or 0)
            is_correct = 1 if ua_int is not None and ua_int == q["correct_answer"] else 0
            if is_correct:
                score += 1
                award_coins(conn, uid, COINS_PER_CORRECT)
            else:
                coaching_triggers.append(build_coaching_trigger(
                    {"id": qid, "question": q["question"],
                     "options": json.loads(q["options"]),
                     "correct_answer": q["correct_answer"],
                     "explanation": q["explanation"],
                     "chapter": q["chapter"]}, ua_int if ua_int is not None else -1))
            update_consecutive_correct(conn, uid, bool(is_correct))
            conn.execute(
                "INSERT INTO test_attempts (user_id,question_id,user_answer,is_correct,time_taken,session_type) VALUES (?,?,?,?,?,?)",
                (uid, qid, ua_int, is_correct, time_t, session_type))
            if time_t > 0:
                conn.execute(
                    "INSERT INTO question_latency (user_id,question_id,session_type,time_taken_sec,is_correct,chapter,difficulty) VALUES (?,?,?,?,?,?,?)",
                    (uid, qid, session_type, time_t, is_correct, q["chapter"], q["difficulty"]))
            results.append({"question_id": qid, "question": q["question"],
                            "options": json.loads(q["options"]), "correct_answer": q["correct_answer"],
                            "user_answer": ua_int, "is_correct": bool(is_correct),
                            "explanation": q["explanation"], "time_taken": time_t})
        return score, results

    if req.test_id == -1:
        q_ids  = [int(k) for k in req.answers.keys() if str(k).lstrip('-').isdigit()]
        score, results = _process_answers(q_ids, req.answers, req.time_taken, req.session_type or "chapter")
        conn.commit()
        xp_earned = score * 5 + 10
        award_xp(conn, uid, xp_earned, "chapter")
        lat_stats  = compute_latency_stats(conn, uid, 30)
        lat_alerts = get_latency_alerts(lat_stats)
        new_achs = check_and_award_achievements(conn, uid, {"score": score, "total": len(results), "mode": "chapter"})
        conn.close()
        return {"score": score, "total": len(results),
                "percentage": round(score / max(len(results), 1) * 100, 1),
                "results": results, "xp_earned": xp_earned, "new_achievements": new_achs,
                "coins_earned": score * COINS_PER_CORRECT, "latency_alerts": lat_alerts[:2],
                "coaching_triggers": coaching_triggers,
                "has_wrong_answers": len(coaching_triggers) > 0,
                "first_coaching_trigger": coaching_triggers[0] if coaching_triggers else None}

    test_row = conn.execute("SELECT * FROM daily_tests WHERE id=? AND user_id=?",
                            (req.test_id, uid)).fetchone()
    if not test_row:
        conn.close()
        raise HTTPException(404, "Test not found")
    test = dict(test_row)

    q_ids = json.loads(test["question_ids"])
    if test["completed"]:
        results = []
        saved_answers = json.loads(test["answers"]) if test.get("answers") else {}
        for qid in q_ids:
            q = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
            if not q: continue
            q = dict(q)
            ua_int = safe_answer_int(saved_answers.get(str(qid)))
            results.append({"question_id": qid, "question": q["question"],
                            "options": json.loads(q["options"]), "correct_answer": q["correct_answer"],
                            "user_answer": ua_int,
                            "is_correct": ua_int is not None and ua_int == q["correct_answer"],
                            "explanation": q["explanation"]})
        conn.close()
        return {"score": test.get("score") or 0, "total": test.get("total") or len(q_ids),
                "percentage": round((test.get("score") or 0) / max(test.get("total") or 1, 1) * 100, 1),
                "results": results, "already_submitted": True,
                "chapter_name": test.get("chapter_name")}

    score, results = _process_answers(q_ids, req.answers, req.time_taken, req.session_type or "daily")
    conn.execute(
        "UPDATE daily_tests SET completed=1,score=?,answers=?,total=?,completed_at=? WHERE id=?",
        (score, json.dumps({str(k): v for k, v in req.answers.items()}),
         len(q_ids), datetime.now().isoformat(), req.test_id))
    conn.commit()
    xp_earned = score * 10 + (50 if score / max(len(q_ids), 1) >= 0.75 else 20)
    award_xp(conn, uid, xp_earned, "daily_test")
    prev = conn.execute(
        "SELECT score, total FROM daily_tests WHERE user_id=? AND completed=1 AND id!=? ORDER BY id DESC LIMIT 1",
        (uid, req.test_id)).fetchone()
    prev_pct = round((prev["score"] or 0) / max(prev["total"] or 1, 1) * 100, 1) if prev else 0

    chapter_name = test.get("chapter_name")
    chapters_row = conn.execute("SELECT chapters FROM users WHERE id=?", (uid,)).fetchone()
    chapters = json.loads(chapters_row["chapters"]) if chapters_row and chapters_row["chapters"] else []
    if chapter_name and chapter_name in chapters:
        ch_idx = chapters.index(chapter_name)
        if is_chapter_complete(conn, uid, chapters, ch_idx):
            conn.execute("INSERT OR IGNORE INTO chapter_completions (user_id,chapter) VALUES (?,?)",
                         (uid, chapter_name))
            current_idx = u.get("daily_test_chapter_idx", 0) or 0
            if ch_idx == current_idx and ch_idx + 1 < len(chapters):
                conn.execute("UPDATE users SET daily_test_chapter_idx=? WHERE id=?", (ch_idx + 1, uid))
            conn.commit()

    lat_stats  = compute_latency_stats(conn, uid, 30)
    lat_alerts = get_latency_alerts(lat_stats)
    new_achs = check_and_award_achievements(conn, uid, {
        "score": score, "total": len(q_ids), "mode": "daily", "prev_pct": prev_pct})
    conn.close()
    return {"score": score, "total": len(q_ids),
            "percentage": round(score / max(len(q_ids), 1) * 100, 1),
            "results": results, "xp_earned": xp_earned, "new_achievements": new_achs,
            "coins_earned": score * COINS_PER_CORRECT, "chapter_name": chapter_name,
            "latency_alerts": lat_alerts[:2],
            "coaching_triggers": coaching_triggers,
            "has_wrong_answers": len(coaching_triggers) > 0,
            "first_coaching_trigger": coaching_triggers[0] if coaching_triggers else None}

# ── WEAK SESSION ──────────────────────────────────────────────────────────────

@app.get("/api/test/weak-session")
def get_weak_session(chapter: Optional[str] = None, u=Depends(get_current_user)):
    conn = get_db()
    conn.execute("UPDATE weak_sessions SET completed=1 WHERE user_id=? AND completed=0", (u["uid"],))
    conn.commit()
    chapter_filter = "AND q.chapter = ?" if chapter else ""

    weak_qs = conn.execute(f"""
        SELECT q.id, q.chapter, q.sub_chapter,
               SUM(CASE WHEN ta.is_correct=0 THEN 1 ELSE 0 END) AS wrong_count,
               SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END) AS right_count,
               (SELECT ta2.is_correct FROM test_attempts ta2
                WHERE ta2.question_id=q.id AND ta2.user_id=?
                ORDER BY ta2.attempted_at DESC LIMIT 1) AS last_result
        FROM questions q JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
        WHERE q.user_id=? {chapter_filter}
        GROUP BY q.id HAVING last_result = 0
        ORDER BY wrong_count DESC LIMIT 15
    """, [u["uid"], u["uid"], u["uid"]] + ([chapter] if chapter else [])).fetchall()

    if not weak_qs or len(weak_qs) < 3:
        weak_qs = conn.execute(f"""
            SELECT q.id, q.chapter, q.sub_chapter,
                   SUM(CASE WHEN ta.is_correct=0 THEN 1 ELSE 0 END) AS wrong_count,
                   SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END) AS right_count,
                   (SELECT ta2.is_correct FROM test_attempts ta2
                    WHERE ta2.question_id=q.id AND ta2.user_id=?
                    ORDER BY ta2.attempted_at DESC LIMIT 1) AS last_result
            FROM questions q JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
            WHERE q.user_id=? {chapter_filter} AND ta.is_correct=0
            GROUP BY q.id HAVING wrong_count > COALESCE(right_count, 0) AND last_result = 0
            ORDER BY wrong_count DESC LIMIT 15
        """, [u["uid"], u["uid"], u["uid"]] + ([chapter] if chapter else [])).fetchall()

    if not weak_qs:
        conn.close()
        return {"session_id": None, "questions": [],
                "message": "No weak areas found! All practiced questions are mastered. 🎉"}

    q_ids = [r["id"] for r in weak_qs]
    conn.execute("INSERT INTO weak_sessions (user_id,question_ids,total) VALUES (?,?,?)",
                 (u["uid"], json.dumps(q_ids), len(q_ids)))
    conn.commit()
    session_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    qs = [fmt_q(conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()) for qid in q_ids]
    qs = [q for q in qs if q]
    conn.close()
    return {"session_id": session_id, "questions": qs, "total": len(qs)}

@app.post("/api/test/weak-session/submit")
def submit_weak_session(req: WeakSessionSubmitReq, u=Depends(get_current_user)):
    conn    = get_db()
    session = conn.execute("SELECT * FROM weak_sessions WHERE id=? AND user_id=?",
                           (req.session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    session = dict(session)
    if session["completed"]:
        conn.close()
        return {"score": session.get("score") or 0, "total": session.get("total") or 0,
                "percentage": round((session.get("score") or 0) / max(session.get("total") or 1, 1) * 100, 1),
                "results": [], "already_submitted": True}
    q_ids = json.loads(session["question_ids"])
    score = 0; results = []; coaching_triggers = []
    for qid in q_ids:
        q = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        if not q: continue
        q = dict(q)
        ua_int     = safe_answer_int(req.answers.get(str(qid)))
        time_t     = int(req.time_taken.get(str(qid), 0) or 0)
        is_correct = 1 if ua_int is not None and ua_int == q["correct_answer"] else 0
        if is_correct:
            score += 1
            award_coins(conn, u["uid"], COINS_PER_CORRECT)
        else:
            coaching_triggers.append(build_coaching_trigger(
                {"id": qid, "question": q["question"], "options": json.loads(q["options"]),
                 "correct_answer": q["correct_answer"], "explanation": q["explanation"],
                 "chapter": q["chapter"]}, ua_int if ua_int is not None else -1))
        update_consecutive_correct(conn, u["uid"], bool(is_correct))
        conn.execute(
            "INSERT INTO test_attempts (user_id,question_id,user_answer,is_correct,time_taken,session_type) VALUES (?,?,?,?,?,?)",
            (u["uid"], qid, ua_int, is_correct, time_t, "weak"))
        if time_t > 0:
            conn.execute(
                "INSERT INTO question_latency (user_id,question_id,session_type,time_taken_sec,is_correct,chapter,difficulty) VALUES (?,?,?,?,?,?,?)",
                (u["uid"], qid, "weak", time_t, is_correct, q["chapter"], q["difficulty"]))
        results.append({"question_id": qid, "question": q["question"],
                        "options": json.loads(q["options"]), "correct_answer": q["correct_answer"],
                        "user_answer": ua_int, "is_correct": bool(is_correct), "explanation": q["explanation"]})
    conn.execute("UPDATE weak_sessions SET completed=1,score=?,answers=?,total=? WHERE id=?",
                 (score, json.dumps({str(k): v for k, v in req.answers.items()}),
                  len(q_ids), req.session_id))
    conn.commit()
    xp_earned = score * 8 + 20
    award_xp(conn, u["uid"], xp_earned, "weak_session")
    lat_stats  = compute_latency_stats(conn, u["uid"], 30)
    lat_alerts = get_latency_alerts(lat_stats)
    new_achs = check_and_award_achievements(conn, u["uid"], {"score": score, "total": len(q_ids), "mode": "weak"})
    conn.close()
    return {"score": score, "total": len(q_ids),
            "percentage": round(score / max(len(q_ids), 1) * 100, 1),
            "results": results, "improved": score, "xp_earned": xp_earned,
            "new_achievements": new_achs, "coins_earned": score * COINS_PER_CORRECT,
            "latency_alerts": lat_alerts[:2],
            "coaching_triggers": coaching_triggers,
            "has_wrong_answers": len(coaching_triggers) > 0,
            "first_coaching_trigger": coaching_triggers[0] if coaching_triggers else None}

# ── INTERVIEW MODE ────────────────────────────────────────────────────────────

@app.get("/api/interview/types")
def get_interview_types():
    return {"types": [{"id": k, **v} for k, v in INTERVIEW_TYPES.items()]}

@app.post("/api/interview/start")
def start_interview(req: InterviewStartReq, u=Depends(get_current_user)):
    if req.interview_type not in INTERVIEW_TYPES:
        raise HTTPException(400, f"Invalid interview_type. Choose from: {list(INTERVIEW_TYPES.keys())}")
    if not 3 <= req.total_questions <= INTERVIEW_MAX_QUESTIONS:
        raise HTTPException(400, f"total_questions must be between 3 and {INTERVIEW_MAX_QUESTIONS}")

    conn = get_db()
    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)

    # Close any existing active interview sessions
    conn.execute("UPDATE interview_sessions SET is_active=0, ended_at=? WHERE user_id=? AND is_active=1",
                 (datetime.now().isoformat(), u["uid"]))

    conn.execute(
        "INSERT INTO interview_sessions (user_id, chapter, exam_type, interview_type, total_questions) VALUES (?,?,?,?,?)",
        (u["uid"], req.chapter, exam_type, req.interview_type, req.total_questions))
    conn.commit()
    session_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Generate first question immediately
    itype = INTERVIEW_TYPES[req.interview_type]
    topic = req.chapter or exam_type

    if req.interview_type == "rapid_fire":
        # MCQ style — generate as structured question
        prompt = f"""You are an interviewer for {exam_type}.{lang_instruction}
Generate 1 rapid-fire MCQ interview question about "{topic}".
Return ONLY this JSON:
{{
  "question": "Question text",
  "type": "mcq",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": 2,
  "difficulty": "medium",
  "topic_tag": "sub-topic"
}}"""
        raw = groq_chat(prompt, temperature=0.7, json_mode=True)
        q_data = parse_json(raw)
        vq = validate_question_dict(q_data)
        if vq:
            qa_id = _save_interview_question(conn, session_id, u["uid"], 1, q_data["question"],
                                              "mcq", json.dumps(vq["options"]), vq["correct_answer"])
            conn.close()
            return {
                "session_id": session_id, "qa_id": qa_id,
                "question_number": 1, "total_questions": req.total_questions,
                "question": q_data["question"], "type": "mcq",
                "options": vq["options"], "interview_type": req.interview_type,
                "interview_label": itype["label"], "icon": itype["icon"],
                "topic": topic, "exam_type": exam_type,
                "message": f"Interview started! {req.total_questions} questions, {itype['label']} style.",
            }
    else:
        prompt = f"""You are a {exam_type} examiner conducting a {itype['label']} interview.{lang_instruction}
Topic: "{topic}"
{itype['q_style']}

Generate question #1 of {req.total_questions}. This is the OPENING question — broad, to warm up.
Return ONLY this JSON:
{{
  "question": "Your interview question here",
  "type": "open",
  "difficulty": "easy",
  "what_to_assess": "What a good answer should cover",
  "topic_tag": "sub-topic being tested"
}}"""
        raw = groq_chat(prompt, temperature=0.7, json_mode=True)
        q_data = parse_json(raw)
        qa_id = _save_interview_question(conn, session_id, u["uid"], 1,
                                          q_data.get("question", "Introduce yourself and your understanding of " + topic),
                                          "open", None, None)
        conn.close()
        return {
            "session_id": session_id, "qa_id": qa_id,
            "question_number": 1, "total_questions": req.total_questions,
            "question": q_data.get("question", ""), "type": "open",
            "options": None, "what_to_assess": q_data.get("what_to_assess", ""),
            "interview_type": req.interview_type, "interview_label": itype["label"],
            "icon": itype["icon"], "topic": topic, "exam_type": exam_type,
            "message": f"Interview started! {req.total_questions} questions ahead. Take your time.",
        }

def _save_interview_question(conn, session_id, user_id, q_num, question, q_type, options, correct_answer):
    conn.execute(
        "INSERT INTO interview_qa (session_id, user_id, question_number, question, question_type, options, correct_answer) VALUES (?,?,?,?,?,?,?)",
        (session_id, user_id, q_num, question, q_type, options, correct_answer))
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

@app.post("/api/interview/answer")
def submit_interview_answer(req: InterviewAnswerReq, u=Depends(get_current_user)):
    conn = get_db()
    session = conn.execute("SELECT * FROM interview_sessions WHERE id=? AND user_id=?",
                           (req.session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Interview session not found")
    session = dict(session)
    if not session["is_active"]:
        conn.close()
        raise HTTPException(400, "Interview session has ended")

    qa = conn.execute("SELECT * FROM interview_qa WHERE id=? AND session_id=?",
                      (req.qa_id, req.session_id)).fetchone()
    if not qa:
        conn.close()
        raise HTTPException(404, "Question not found")
    qa = dict(qa)

    if qa.get("answered_at"):
        conn.close()
        return {"already_answered": True, "feedback": qa.get("ai_feedback", ""), "score": qa.get("score", 0)}

    exam_type = session["exam_type"] or "General"
    itype     = INTERVIEW_TYPES.get(session["interview_type"], INTERVIEW_TYPES["conceptual"])
    q_num     = qa["question_number"]
    total     = session["total_questions"]
    lang_key  = u.get("exam_language")
    lang_instruction = get_language_instruction(lang_key)

    # Score the answer
    if qa["question_type"] == "mcq":
        try:
            user_ans_int = int(req.user_answer)
        except (ValueError, TypeError):
            user_ans_int = -1
        is_correct = user_ans_int == qa["correct_answer"]
        score = 10.0 if is_correct else 0.0
        opts = json.loads(qa["options"]) if qa["options"] else []
        correct_opt = opts[qa["correct_answer"]] if qa["correct_answer"] is not None and qa["correct_answer"] < len(opts) else "N/A"
        user_opt = opts[user_ans_int] if 0 <= user_ans_int < len(opts) else "No answer"
        feedback = (
            f"✅ **Correct!** {correct_opt} is right.\n\n"
            if is_correct else
            f"❌ **Incorrect.** You chose: {user_opt}\n✅ **Correct answer**: {correct_opt}\n\n"
        )
        feedback += f"*Score: {int(score)}/10*"
    else:
        # AI evaluates open-ended answer
        eval_prompt = f"""You are evaluating a {exam_type} interview answer.{lang_instruction}

QUESTION: {qa['question']}
STUDENT ANSWER: {req.user_answer}

{itype['scoring']}

Evaluate and return ONLY this JSON:
{{
  "score": 7,
  "feedback": "2-3 sentences: what was good, what was missing",
  "key_points_covered": ["point 1", "point 2"],
  "key_points_missed": ["missed point 1"],
  "model_answer_hint": "Brief hint at the ideal answer (1-2 sentences)",
  "follow_up_tip": "One specific thing to study"
}}

Be encouraging but honest. Score strictly 0-10."""
        try:
            raw = groq_chat(eval_prompt, temperature=0.5, json_mode=True)
            eval_data = parse_json(raw)
            score = float(eval_data.get("score", 5))
            score = max(0, min(10, score))
            feedback = f"**Score: {score}/10**\n\n{eval_data.get('feedback', '')}"
            if eval_data.get("key_points_missed"):
                missed = ", ".join(eval_data["key_points_missed"])
                feedback += f"\n\n**To improve**: {missed}"
            if eval_data.get("model_answer_hint"):
                feedback += f"\n\n**Hint**: {eval_data['model_answer_hint']}"
            if eval_data.get("follow_up_tip"):
                feedback += f"\n\n📖 *Study tip*: {eval_data['follow_up_tip']}"
        except Exception as exc:
            logger.error("Interview evaluation failed: %s", exc)
            score = 5.0
            feedback = "Your answer was recorded. Keep developing your understanding of this topic."

    is_correct_flag = 1 if score >= 5 else 0
    conn.execute(
        "UPDATE interview_qa SET user_answer=?, ai_feedback=?, score=?, max_score=10, is_correct=?, answered_at=?, time_taken=? WHERE id=?",
        (req.user_answer, feedback, score, is_correct_flag, datetime.now().isoformat(), req.time_taken, req.qa_id))

    new_total_score  = (session["total_score"] or 0) + score
    new_max_score    = (session["max_score"] or 0) + 10
    new_q_asked      = (session["questions_asked"] or 0) + 1

    conn.execute("UPDATE interview_sessions SET total_score=?, max_score=?, questions_asked=? WHERE id=?",
                 (new_total_score, new_max_score, new_q_asked, req.session_id))
    conn.commit()

    # Check if interview is done
    is_last = new_q_asked >= total
    next_qa_id   = None
    next_question = None
    next_type    = None
    next_options  = None

    if not is_last:
        # Generate next question
        next_q_num = new_q_asked + 1
        difficulty = ("easy" if next_q_num <= 2 else
                      "hard" if next_q_num >= total - 1 else "medium")
        prev_answers_summary = f"So far answered {new_q_asked} questions. Score so far: {new_total_score}/{new_max_score}."

        if session["interview_type"] == "rapid_fire":
            next_prompt = f"""Generate 1 {difficulty} rapid-fire MCQ about "{session['chapter'] or exam_type}" for {exam_type}.{lang_instruction}
{prev_answers_summary} This is question {next_q_num}/{total}.
Return ONLY JSON: {{"question":"...","type":"mcq","options":["A","B","C","D"],"correct_answer":2,"difficulty":"{difficulty}"}}"""
            raw = groq_chat(next_prompt, temperature=0.8, json_mode=True)
            q_data = parse_json(raw)
            vq = validate_question_dict(q_data)
            if vq:
                next_qa_id = _save_interview_question(conn, req.session_id, u["uid"], next_q_num,
                                                       q_data["question"], "mcq",
                                                       json.dumps(vq["options"]), vq["correct_answer"])
                next_question = q_data["question"]
                next_type = "mcq"
                next_options = vq["options"]
        elif session["interview_type"] == "viva":
            # Follow-up question based on previous answer
            follow_prompt = f"""You are conducting a viva for {exam_type}.{lang_instruction}
Previous question: "{qa['question']}"
Student answered: "{req.user_answer}"
Score given: {score}/10

Generate a follow-up question that:
- If score < 6: clarifies or simplifies the same concept
- If score >= 6: digs deeper or explores a related concept

Question {next_q_num}/{total}, difficulty: {difficulty}.
Return ONLY JSON: {{"question":"...","type":"open","what_to_assess":"..."}}"""
            raw = groq_chat(follow_prompt, temperature=0.7, json_mode=True)
            q_data = parse_json(raw)
            next_qa_id = _save_interview_question(conn, req.session_id, u["uid"], next_q_num,
                                                   q_data.get("question", ""), "open", None, None)
            next_question = q_data.get("question", "")
            next_type = "open"
        else:
            next_prompt = f"""You are interviewing for {exam_type} on "{session['chapter'] or exam_type}".{lang_instruction}
{itype['q_style']}
{prev_answers_summary}
Question {next_q_num}/{total}, difficulty: {difficulty}.
Return ONLY JSON: {{"question":"...","type":"open","what_to_assess":"..."}}"""
            raw = groq_chat(next_prompt, temperature=0.7, json_mode=True)
            q_data = parse_json(raw)
            next_qa_id = _save_interview_question(conn, req.session_id, u["uid"], next_q_num,
                                                   q_data.get("question", ""), "open", None, None)
            next_question = q_data.get("question", "")
            next_type = "open"

    response = {
        "qa_id": req.qa_id, "feedback": feedback, "score": score,
        "is_correct": bool(is_correct_flag),
        "questions_answered": new_q_asked, "total_questions": total,
        "running_score": new_total_score, "running_max": new_max_score,
        "running_pct": round(new_total_score / max(new_max_score, 1) * 100, 1),
        "is_last_question": is_last,
        "next_qa_id": next_qa_id,
        "next_question": next_question,
        "next_type": next_type,
        "next_options": next_options,
        "next_question_number": new_q_asked + 1 if not is_last else None,
    }

    if is_last:
        # Auto-end the session
        final = _end_interview_session(conn, u, session, new_total_score, new_max_score, new_q_asked)
        response.update(final)
        conn.close()
        return response

    conn.commit()
    conn.close()
    return response


def _end_interview_session(conn, u, session, total_score, max_score, q_asked):
    pct = round(total_score / max(max_score, 1) * 100, 1) if max_score else 0
    performance = ("excellent" if pct >= 85 else "good" if pct >= 65 else
                   "fair" if pct >= 45 else "needs_practice")
    perf_emoji  = {"excellent": "🏆", "good": "✅", "fair": "📈", "needs_practice": "💪"}

    # Generate AI summary
    all_qa = conn.execute(
        "SELECT question, user_answer, ai_feedback, score, max_score FROM interview_qa "
        "WHERE session_id=? ORDER BY question_number", (session["id"],)).fetchall()
    qa_summary = "\n".join(
        f"Q{i+1}: {r['question'][:80]}... | Score: {r['score']}/10"
        for i, r in enumerate(all_qa[:8]))

    lang_key = u.get("exam_language")
    summary_prompt = f"""Write a SHORT interview performance summary (4-5 sentences) for {u.get('name','Student')} after a {session['interview_type']} interview on "{session['chapter'] or session['exam_type']}".
Score: {total_score:.0f}/{max_score:.0f} ({pct}%) — Performance: {performance}
Questions summary:
{qa_summary}

Cover: 1 strength, 1 area to improve, 1 specific study recommendation. Be encouraging."""
    try:
        summary_text = groq_chat(summary_prompt, temperature=0.7, json_mode=False)
    except Exception:
        summary_text = f"Interview complete! You scored {pct}%. {'Great job!' if pct >= 70 else 'Keep practicing to improve.'}"

    conn.execute(
        "UPDATE interview_sessions SET is_active=0, ended_at=?, total_score=?, max_score=?, questions_asked=?, summary=? WHERE id=?",
        (datetime.now().isoformat(), total_score, max_score, q_asked, summary_text, session["id"]))
    conn.commit()

    xp_earned = int(total_score * 2) + (50 if pct >= 75 else 15)
    award_xp(conn, u["uid"], xp_earned, "interview_complete")
    award_coins(conn, u["uid"], int(total_score))
    new_achs = check_and_award_achievements(conn, u["uid"], {
        "score": int(total_score), "total": int(max_score / 10) if max_score else 1,
        "mode": "interview"})

    return {
        "session_ended": True,
        "final_score": total_score, "final_max": max_score, "final_pct": pct,
        "performance": performance,
        "performance_emoji": perf_emoji.get(performance, "📊"),
        "summary": summary_text,
        "xp_earned": xp_earned,
        "coins_earned": int(total_score),
        "new_achievements": new_achs,
    }


@app.post("/api/interview/{session_id}/end")
def end_interview(session_id: int, u=Depends(get_current_user)):
    conn = get_db()
    session = conn.execute("SELECT * FROM interview_sessions WHERE id=? AND user_id=?",
                           (session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Interview session not found")
    session = dict(session)
    if not session["is_active"]:
        conn.close()
        return {"message": "Session already ended", "summary": session.get("summary"),
                "final_score": session["total_score"], "final_max": session["max_score"],
                "final_pct": round((session["total_score"] or 0) / max(session["max_score"] or 1, 1) * 100, 1)}
    result = _end_interview_session(conn, u, session,
                                     session["total_score"] or 0,
                                     session["max_score"] or 0,
                                     session["questions_asked"] or 0)
    conn.close()
    return result


@app.get("/api/interview/{session_id}/results")
def get_interview_results(session_id: int, u=Depends(get_current_user)):
    conn = get_db()
    session = conn.execute("SELECT * FROM interview_sessions WHERE id=? AND user_id=?",
                           (session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Interview session not found")
    session = dict(session)
    qa_rows = conn.execute(
        "SELECT * FROM interview_qa WHERE session_id=? ORDER BY question_number",
        (session_id,)).fetchall()
    conn.close()
    pct = round((session["total_score"] or 0) / max(session["max_score"] or 1, 1) * 100, 1)
    return {
        "session_id": session_id,
        "chapter": session.get("chapter"),
        "exam_type": session.get("exam_type"),
        "interview_type": session.get("interview_type"),
        "total_questions": session["total_questions"],
        "questions_answered": session["questions_asked"],
        "total_score": session["total_score"] or 0,
        "max_score": session["max_score"] or 0,
        "percentage": pct,
        "performance": ("excellent" if pct >= 85 else "good" if pct >= 65 else "fair" if pct >= 45 else "needs_practice"),
        "summary": session.get("summary"),
        "is_active": bool(session["is_active"]),
        "created_at": session["created_at"],
        "ended_at": session.get("ended_at"),
        "questions": [
            {"qa_id": r["id"], "question_number": r["question_number"],
             "question": r["question"], "type": r["question_type"],
             "options": json.loads(r["options"]) if r["options"] else None,
             "correct_answer": r["correct_answer"],
             "user_answer": r["user_answer"],
             "ai_feedback": r["ai_feedback"],
             "score": r["score"], "max_score": r["max_score"],
             "is_correct": bool(r["is_correct"]), "time_taken": r["time_taken"]}
            for r in qa_rows
        ],
    }


@app.get("/api/interview/history")
def interview_history(u=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, chapter, exam_type, interview_type, total_questions, questions_asked, "
        "total_score, max_score, is_active, summary, created_at, ended_at "
        "FROM interview_sessions WHERE user_id=? ORDER BY id DESC LIMIT 20",
        (u["uid"],)).fetchall()
    conn.close()
    return {"sessions": [
        {"session_id": r["id"], "chapter": r["chapter"], "exam_type": r["exam_type"],
         "interview_type": r["interview_type"],
         "interview_label": INTERVIEW_TYPES.get(r["interview_type"], {}).get("label", r["interview_type"]),
         "total_questions": r["total_questions"], "questions_answered": r["questions_asked"] or 0,
         "total_score": r["total_score"] or 0, "max_score": r["max_score"] or 0,
         "percentage": round((r["total_score"] or 0) / max(r["max_score"] or 1, 1) * 100, 1),
         "is_active": bool(r["is_active"]), "summary": r["summary"],
         "created_at": r["created_at"], "ended_at": r["ended_at"]}
        for r in rows
    ]}

# ── LATENCY STATS ─────────────────────────────────────────────────────────────

@app.get("/api/latency/stats")
def get_latency_stats(days: int = Query(30, ge=1, le=90), u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]
    stats = compute_latency_stats(conn, uid, days)
    alerts = get_latency_alerts(stats)
    sessions = conn.execute("""
        SELECT session_type, COUNT(*) as q_count, AVG(time_taken_sec) as avg_t,
               MAX(time_taken_sec) as max_t, MIN(time_taken_sec) as min_t,
               SUM(CASE WHEN time_taken_sec > ? THEN 1 ELSE 0 END) as slow_q,
               date(recorded_at) as day
        FROM question_latency
        WHERE user_id=? AND time_taken_sec > 0 AND recorded_at >= ?
        GROUP BY date(recorded_at), session_type ORDER BY recorded_at DESC LIMIT 14
    """, (LATENCY_SLOW_SINGLE, uid, (datetime.now() - timedelta(days=days)).isoformat())).fetchall()
    diff_lat = conn.execute("""
        SELECT difficulty, AVG(time_taken_sec) as avg_t, COUNT(*) as cnt
        FROM question_latency WHERE user_id=? AND time_taken_sec > 0 GROUP BY difficulty
    """, (uid,)).fetchall()
    conn.close()
    return {
        "stats": stats, "alerts": alerts,
        "thresholds": {"fast": LATENCY_FAST_THRESHOLD, "slow": LATENCY_SLOW_SINGLE,
                       "target_avg": LATENCY_WARNING_THRESHOLD},
        "session_history": [
            {"day": r["day"], "session_type": r["session_type"], "questions": r["q_count"],
             "avg_time": round(r["avg_t"], 1), "max_time": r["max_t"],
             "min_time": r["min_t"], "slow_questions": r["slow_q"]}
            for r in sessions],
        "by_difficulty": [{"difficulty": r["difficulty"], "avg_time": round(r["avg_t"], 1),
                            "question_count": r["cnt"]} for r in diff_lat],
    }

@app.post("/api/latency/coach-advice")
def latency_coach_advice(req: LatencyCoachReq, u=Depends(get_current_user)):
    conn  = get_db()
    uid   = u["uid"]
    stats = compute_latency_stats(conn, uid, 30)
    alerts = get_latency_alerts(stats)
    memory = _get_coach_memory(conn, uid)
    chapter_times = stats.get("chapter_breakdown", {})
    slowest_ch = stats.get("slowest_chapter", "unknown chapter")
    top3_slow = sorted(chapter_times.items(), key=lambda x: x[1], reverse=True)[:3]
    slow_ch_str = ", ".join(f"{ch} ({t}s avg)" for ch, t in top3_slow) if top3_slow else "none identified"
    focus = req.chapter or slowest_ch or "exam topics"
    prompt = f"""You are ExamAI Time Coach for {u.get('name','Student')} ({u.get('exam_type','General')}).
LATENCY DATA: avg={stats.get('avg_time',0)}s, slow_q={stats.get('slow_count',0)}/{stats.get('total_tracked',0)}, trend={stats.get('trend','unknown')}, slowest: {slow_ch_str}
ALERTS: {[a['message'] for a in alerts[:2]]}
{f'Student context: {memory[:200]}' if memory else ''}
Write 4-5 sentences of personalized time-efficiency coaching: 1. acknowledge their timing pattern 2. ONE concrete technique for "{focus}" 3. ONE drill RIGHT NOW (time-boxed) 4. encouragement. Direct, energetic. Use actual numbers."""
    advice = groq_chat(prompt, temperature=0.75, json_mode=False)
    conn.execute("INSERT INTO coach_messages (user_id,message,msg_type) VALUES (?,?,'coach')", (uid, advice))
    conn.commit(); conn.close()
    return {"advice": advice, "stats_summary": {"avg_time": stats.get("avg_time",0),
            "trend": stats.get("trend","no_data"), "slow_count": stats.get("slow_count",0),
            "total_tracked": stats.get("total_tracked",0), "slowest_chapter": slowest_ch},
            "alerts": alerts, "drill": f"Try: answer 10 questions on '{focus}' with a strict 8-minute timer."}

@app.get("/api/latency/leaderboard")
def latency_leaderboard(u=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("""
        SELECT ql.user_id, u.name, u.avatar_color, AVG(ql.time_taken_sec) as avg_t, COUNT(*) as q_cnt
        FROM question_latency ql JOIN users u ON ql.user_id = u.id
        WHERE ql.time_taken_sec > 0 GROUP BY ql.user_id HAVING q_cnt >= 20 ORDER BY avg_t ASC LIMIT 20
    """).fetchall()
    my_rank = next((i + 1 for i, r in enumerate(rows) if r["user_id"] == u["uid"]), None)
    my_stats = compute_latency_stats(conn, u["uid"], 30)
    conn.close()
    return {
        "leaderboard": [{"rank": i + 1, "name": r["name"], "avatar_color": r["avatar_color"],
                          "avg_time_sec": round(r["avg_t"], 1), "questions_tracked": r["q_cnt"],
                          "is_me": r["user_id"] == u["uid"]} for i, r in enumerate(rows)],
        "my_rank": my_rank, "my_avg_time": my_stats.get("avg_time", 0),
        "my_questions_tracked": my_stats.get("total_tracked", 0),
        "note": "Rankings by average decision time (lower is better). Min 20 questions required.",
    }

# ── GLOBAL TESTS ──────────────────────────────────────────────────────────────

def _parse_dt(dt_str: str) -> datetime:
    dt_str = (dt_str or "").strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError:
        dt = datetime.fromisoformat(dt_str[:19])
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    else:
        dt = dt.astimezone(pytz.utc)
    return dt

@app.get("/api/global-tests")
def list_global_tests_for_user(u=Depends(get_current_user)):
    conn = get_db()
    exam_type = (u.get("exam_type") or "").strip()
    VISIBLE = ('draft', 'generating', 'ready', 'lobby', 'live', 'ended')
    ph = ",".join("?" * len(VISIBLE))
    if exam_type:
        rows = conn.execute(
            f"SELECT gt.*, (SELECT COUNT(*) FROM global_participants WHERE global_test_id=gt.id) as participant_count "
            f"FROM global_tests gt WHERE gt.status IN ({ph}) "
            "AND (UPPER(gt.exam_type) LIKE ? OR UPPER(gt.exam_type)='GENERAL') ORDER BY gt.id DESC LIMIT 20",
            list(VISIBLE) + [f"%{exam_type.upper()}%"]).fetchall()
    else:
        rows = conn.execute(
            f"SELECT gt.*, (SELECT COUNT(*) FROM global_participants WHERE global_test_id=gt.id) as participant_count "
            f"FROM global_tests gt WHERE gt.status IN ({ph}) ORDER BY gt.id DESC LIMIT 20",
            list(VISIBLE)).fetchall()
    uid = u["uid"]
    result = []
    for r in rows:
        r = dict(r)
        part = conn.execute("SELECT * FROM global_participants WHERE global_test_id=? AND user_id=?",
                            (r["id"], uid)).fetchone()
        part = dict(part) if part else None
        try:
            q_count = len(json.loads(r.get("question_ids") or "[]"))
        except Exception:
            q_count = 0
        result.append({
            "id": r["id"], "title": r.get("title",""), "exam_type": r.get("exam_type",""),
            "topic": r.get("topic",""), "status": r.get("status","draft"),
            "scheduled_at": r.get("scheduled_at"), "starts_at": r.get("starts_at"),
            "duration_minutes": r.get("duration_minutes", 60),
            "participant_count": r.get("participant_count", 0), "question_count": q_count,
            "winner_name": r.get("winner_name"), "winner_score": r.get("winner_score"),
            "my_participation": {
                "joined": part is not None,
                "submitted": part["submitted_at"] is not None if part else False,
                "score": part["score"] if part else None,
                "rank": part["rank"] if part else None,
            } if part else {"joined": False, "submitted": False},
        })
    conn.close()
    return {"tests": result, "total": len(result)}

@app.post("/api/global-tests/join")
def join_global_test(req: GlobalTestJoinReq, u=Depends(get_current_user)):
    conn = get_db()
    test = conn.execute("SELECT * FROM global_tests WHERE id=?", (req.global_test_id,)).fetchone()
    if not test:
        conn.close()
        raise HTTPException(404, "Test not found")
    test = dict(test)
    if test["status"] not in ("draft", "ready", "lobby", "live"):
        conn.close()
        raise HTTPException(400, f"Test not open. Status: {test['status']}")
    now_iso = datetime.now().isoformat()
    existing = conn.execute("SELECT id, started_at FROM global_participants WHERE global_test_id=? AND user_id=?",
                            (req.global_test_id, u["uid"])).fetchone()
    if existing:
        existing = dict(existing)
        if test["status"] == "live" and not existing["started_at"]:
            conn.execute("UPDATE global_participants SET started_at=? WHERE global_test_id=? AND user_id=?",
                         (now_iso, req.global_test_id, u["uid"]))
            conn.commit()
        conn.close()
        return {"message": "Already joined", "already_joined": True,
                "starts_at": test["starts_at"], "duration_minutes": test["duration_minutes"]}
    started_at = now_iso if test["status"] == "live" else None
    conn.execute("INSERT INTO global_participants (global_test_id, user_id, joined_at, started_at) VALUES (?,?,?,?)",
                 (req.global_test_id, u["uid"], now_iso, started_at))
    conn.commit(); conn.close()
    return {"message": "Joined!", "starts_at": test["starts_at"],
            "lobby_opens_at": test["scheduled_at"], "duration_minutes": test["duration_minutes"],
            "test_is_live": test["status"] == "live"}

@app.get("/api/global-tests/{test_id}/questions")
def get_global_test_questions(test_id: int, u=Depends(get_current_user)):
    conn = get_db()
    test_row = conn.execute("SELECT * FROM global_tests WHERE id=?", (test_id,)).fetchone()
    if not test_row:
        conn.close()
        raise HTTPException(404, "Test not found")
    test = dict(test_row)
    if test["status"] not in ("live", "draft"):
        conn.close()
        raise HTTPException(400, f"Test not available. Status: '{test['status']}'.")
    now_iso = datetime.now().isoformat()
    part = conn.execute("SELECT * FROM global_participants WHERE global_test_id=? AND user_id=?",
                        (test_id, u["uid"])).fetchone()
    if not part:
        conn.execute("INSERT INTO global_participants (global_test_id, user_id, joined_at, started_at) VALUES (?,?,?,?)",
                     (test_id, u["uid"], now_iso, now_iso))
        conn.commit()
        part = conn.execute("SELECT * FROM global_participants WHERE global_test_id=? AND user_id=?",
                            (test_id, u["uid"])).fetchone()
    part = dict(part)
    if part.get("submitted_at"):
        conn.close()
        raise HTTPException(400, "Already submitted.")
    if not part.get("started_at"):
        conn.execute("UPDATE global_participants SET started_at=? WHERE global_test_id=? AND user_id=?",
                     (now_iso, test_id, u["uid"]))
        conn.commit()
    qs = conn.execute("SELECT id, question, options, difficulty, topic_tag FROM global_questions WHERE global_test_id=? ORDER BY id",
                      (test_id,)).fetchall()
    try:
        starts_dt = _parse_dt(test["starts_at"])
        ends_dt   = starts_dt + timedelta(minutes=test["duration_minutes"])
        now_utc   = datetime.now(pytz.utc)
        seconds_remaining = max(0, int((ends_dt - now_utc).total_seconds()))
        ends_at_iso = ends_dt.isoformat()
    except Exception as exc:
        logger.error("Time calc error: %s", exc)
        seconds_remaining = test["duration_minutes"] * 60
        ends_at_iso = None
    conn.close()
    questions_list = [
        {"id": q["id"], "question": q["question"],
         "options": json.loads(q["options"]) if isinstance(q["options"], str) else (q["options"] or []),
         "difficulty": q["difficulty"] or "medium", "topic_tag": q["topic_tag"]}
        for q in qs]
    return {"test_id": test_id, "title": test.get("title",""), "duration_minutes": test.get("duration_minutes",60),
            "seconds_remaining": seconds_remaining, "ends_at": ends_at_iso, "status": test.get("status"),
            "questions": questions_list, "total_questions": len(questions_list)}

@app.post("/api/global-tests/submit")
def submit_global_test(req: GlobalTestSubmitReq, u=Depends(get_current_user)):
    conn = get_db()
    test = conn.execute("SELECT * FROM global_tests WHERE id=?", (req.global_test_id,)).fetchone()
    if not test:
        conn.close()
        raise HTTPException(404, "Test not found")
    test = dict(test)
    if test["status"] not in ("live", "ended", "draft"):
        conn.close()
        raise HTTPException(400, f"Not accepting submissions. Status: {test['status']}")
    part = conn.execute("SELECT * FROM global_participants WHERE global_test_id=? AND user_id=?",
                        (req.global_test_id, u["uid"])).fetchone()
    if not part:
        now_iso = datetime.now().isoformat()
        conn.execute("INSERT INTO global_participants (global_test_id, user_id, joined_at, started_at) VALUES (?,?,?,?)",
                     (req.global_test_id, u["uid"], now_iso, now_iso))
        conn.commit()
        part = conn.execute("SELECT * FROM global_participants WHERE global_test_id=? AND user_id=?",
                            (req.global_test_id, u["uid"])).fetchone()
    part = dict(part)
    if part.get("submitted_at"):
        conn.close()
        return {"already_submitted": True, "score": part["score"], "total": part["total"],
                "percentage": round((part["score"] or 0) / max(part["total"] or 1, 1) * 100, 1),
                "rank": part.get("rank")}
    qs = conn.execute("SELECT id, correct_answer, explanation FROM global_questions WHERE global_test_id=?",
                      (req.global_test_id,)).fetchall()
    score = 0; total = len(qs); results = []
    for q in qs:
        q = dict(q)
        user_ans = req.answers.get(str(q["id"]))
        try:
            user_ans_int = int(user_ans) if user_ans is not None else None
        except (TypeError, ValueError):
            user_ans_int = None
        is_correct = user_ans_int is not None and user_ans_int == q["correct_answer"]
        if is_correct: score += 1
        results.append({"question_id": q["id"], "user_answer": user_ans_int,
                        "correct_answer": q["correct_answer"], "is_correct": is_correct,
                        "explanation": q["explanation"]})
    conn.execute(
        "UPDATE global_participants SET submitted_at=?, answers=?, score=?, total=?, time_taken_sec=? WHERE global_test_id=? AND user_id=?",
        (datetime.now().isoformat(), json.dumps({str(k): v for k, v in req.answers.items()}),
         score, total, req.time_taken_sec, req.global_test_id, u["uid"]))
    conn.commit()
    pct = round(score / max(total, 1) * 100, 1)
    xp_earned = score * 12 + (100 if pct >= 75 else 30)
    award_xp(conn, u["uid"], xp_earned, "global_test")
    if score > 0: award_coins(conn, u["uid"], score * 3)
    new_achs = check_and_award_achievements(conn, u["uid"], {"score": score, "total": total, "mode": "global"})
    current_rank = conn.execute(
        "SELECT COUNT(*) FROM global_participants WHERE global_test_id=? AND submitted_at IS NOT NULL AND score > ?",
        (req.global_test_id, score)).fetchone()[0] + 1
    conn.close()
    return {"score": score, "total": total, "percentage": pct, "provisional_rank": current_rank,
            "xp_earned": xp_earned, "coins_earned": score * 3, "results": results,
            "new_achievements": new_achs,
            "message": f"Submitted! {score}/{total} ({pct}%). Provisional rank: #{current_rank}"}

@app.get("/api/global-tests/{test_id}/leaderboard")
def global_test_leaderboard(test_id: int, u=Depends(get_current_user)):
    conn = get_db()
    test_row = conn.execute("SELECT * FROM global_tests WHERE id=?", (test_id,)).fetchone()
    if not test_row:
        conn.close()
        raise HTTPException(404, "Test not found")
    test = dict(test_row)
    rows = conn.execute(
        "SELECT gp.rank, gp.score, gp.total, gp.time_taken_sec, gp.submitted_at, u.name, u.avatar_color "
        "FROM global_participants gp JOIN users u ON gp.user_id=u.id WHERE gp.global_test_id=? "
        "AND gp.submitted_at IS NOT NULL ORDER BY gp.rank ASC, gp.score DESC, gp.time_taken_sec ASC LIMIT 50",
        (test_id,)).fetchall()
    my_part = conn.execute("SELECT rank, score, total, time_taken_sec FROM global_participants WHERE global_test_id=? AND user_id=?",
                           (test_id, u["uid"])).fetchone()
    total_p    = conn.execute("SELECT COUNT(*) FROM global_participants WHERE global_test_id=?", (test_id,)).fetchone()[0]
    submitted_c = conn.execute("SELECT COUNT(*) FROM global_participants WHERE global_test_id=? AND submitted_at IS NOT NULL", (test_id,)).fetchone()[0]
    conn.close()
    return {
        "test": {"id": test["id"], "title": test.get("title",""), "exam_type": test.get("exam_type",""),
                 "status": test.get("status",""), "winner_name": test.get("winner_name"),
                 "winner_score": test.get("winner_score")},
        "leaderboard": [{"rank": r["rank"] or "—", "name": r["name"], "avatar_color": r["avatar_color"],
                          "score": r["score"], "total": r["total"],
                          "pct": round((r["score"] or 0) / max(r["total"] or 1, 1) * 100, 1),
                          "time_taken_sec": r["time_taken_sec"]} for r in rows],
        "my_result": dict(my_part) if my_part else None,
        "total_participants": total_p, "submitted_count": submitted_c,
    }

@app.get("/api/global-tests/{test_id}/status")
def global_test_status(test_id: int, u=Depends(get_current_user)):
    conn = get_db()
    test_row = conn.execute(
        "SELECT id, title, status, scheduled_at, starts_at, duration_minutes, winner_name, winner_score "
        "FROM global_tests WHERE id=?", (test_id,)).fetchone()
    if not test_row:
        conn.close()
        raise HTTPException(404, "Test not found")
    test = dict(test_row)
    part_row = conn.execute("SELECT joined_at, submitted_at, score, rank FROM global_participants WHERE global_test_id=? AND user_id=?",
                            (test_id, u["uid"])).fetchone()
    part = dict(part_row) if part_row else None
    participant_count = conn.execute("SELECT COUNT(*) FROM global_participants WHERE global_test_id=?", (test_id,)).fetchone()[0]
    try:
        starts_dt = _parse_dt(test["starts_at"])
        ends_dt   = starts_dt + timedelta(minutes=test["duration_minutes"])
        now_utc   = datetime.now(pytz.utc)
        seconds_to_start  = max(0, int((starts_dt - now_utc).total_seconds()))
        seconds_remaining = max(0, int((ends_dt   - now_utc).total_seconds()))
    except Exception:
        seconds_to_start = seconds_remaining = 0
    conn.close()
    return {"test_id": test_id, "title": test.get("title",""), "status": test.get("status",""),
            "scheduled_at": test.get("scheduled_at"), "starts_at": test.get("starts_at"),
            "seconds_to_start": seconds_to_start,
            "seconds_remaining": seconds_remaining if test.get("status") == "live" else None,
            "participant_count": participant_count, "winner_name": test.get("winner_name"),
            "winner_score": test.get("winner_score"),
            "my_status": {"joined": part is not None,
                          "submitted": part["submitted_at"] is not None if part else False,
                          "score": part["score"] if part else None,
                          "rank": part["rank"] if part else None} if part else {"joined": False}}

# ── DYNAMIC SESSION ───────────────────────────────────────────────────────────

def _build_dynamic_prompt(exam_type, chapter, seen_count, web_ctx, lang_instruction=""):
    topic       = chapter or exam_type
    ctx_block   = f'\nREAL-WORLD REFERENCE:\n"""\n{web_ctx[:4000]}\n"""\n' if web_ctx else ""
    style_hints = get_exam_style_hints(exam_type)
    return build_question_prompt(topic, exam_type, DYNAMIC_MAX_PER_CALL,
                                  style_hints, ctx_block, seen_count, lang_instruction)

def _generate_dynamic_batch(exam_type, chapter, seen_hashes, lang_key=None):
    lang_instruction = get_language_instruction(lang_key)
    web_ctx = search_topic_context(chapter or exam_type, exam_type, deep=False)
    prompt  = _build_dynamic_prompt(exam_type, chapter, len(seen_hashes), web_ctx, lang_instruction)
    try:
        raw  = groq_chat(prompt, temperature=0.8)
        data = parse_json(raw)
        qs   = data.get("questions", []) if isinstance(data, dict) else data
    except Exception as exc:
        logger.error("Dynamic batch generation failed: %s", exc)
        return []
    results = []
    for q in (qs or []):
        vq = validate_question_dict(q)
        if not vq:
            continue
        h = q_hash(vq["question"])
        if h not in seen_hashes:
            results.append({**vq, "source": "web_ai" if web_ctx else "ai", "hash": h})
    return results

@app.post("/api/session/dynamic/start")
def dynamic_start(req: DynamicStartReq, u=Depends(get_current_user)):
    conn = get_db()
    conn.execute("UPDATE dynamic_sessions SET is_active=0, ended_at=? WHERE user_id=? AND is_active=1",
                 (datetime.now().isoformat(), u["uid"]))
    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    initial_pool = _generate_dynamic_batch(exam_type, req.chapter, set(), lang_key)
    seen = {q["hash"] for q in initial_pool}
    conn.execute(
        "INSERT INTO dynamic_sessions (user_id, chapter, exam_type, seen_hashes, question_pool) VALUES (?,?,?,?,?)",
        (u["uid"], req.chapter, exam_type, json.dumps(list(seen)), json.dumps(initial_pool)))
    conn.commit()
    session_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"session_id": session_id, "chapter": req.chapter, "exam_type": exam_type,
            "pool_ready": len(initial_pool),
            "message": f"Dynamic session started! {len(initial_pool)} questions pre-loaded."}

@app.get("/api/session/dynamic/{session_id}/next")
def dynamic_next(session_id: int, u=Depends(get_current_user)):
    conn    = get_db()
    session = conn.execute("SELECT * FROM dynamic_sessions WHERE id=? AND user_id=?",
                           (session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Dynamic session not found")
    if not session["is_active"]:
        conn.close()
        raise HTTPException(400, "Session has ended. Start a new one.")
    seen_hashes   = set(json.loads(session["seen_hashes"] or "[]"))
    question_pool = json.loads(session["question_pool"] or "[]")
    exam_type     = session["exam_type"] or "General"
    chapter       = session["chapter"]
    lang_key      = conn.execute("SELECT exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    lang_key      = lang_key["exam_language"] if lang_key else None
    q_data = None
    if question_pool:
        q_data = question_pool.pop(0)
    if len(question_pool) < 2:
        new_batch = _generate_dynamic_batch(exam_type, chapter, seen_hashes, lang_key)
        new_batch = [q for q in new_batch if q["hash"] not in seen_hashes]
        if not q_data and new_batch:
            q_data = new_batch.pop(0)
        question_pool.extend(new_batch)
    if not q_data:
        conn.close()
        raise HTTPException(503, "Could not generate a unique question. Please start a new session.")
    seen_hashes.add(q_data["hash"])
    conn.execute(
        "INSERT INTO dynamic_attempts (session_id, user_id, question_text, options, correct_answer, explanation, source) VALUES (?,?,?,?,?,?,?)",
        (session_id, u["uid"], q_data["question"], json.dumps(q_data["options"]),
         q_data["correct_answer"], q_data["explanation"], q_data.get("source", "ai")))
    conn.commit()
    attempt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("UPDATE dynamic_sessions SET seen_hashes=?, question_pool=? WHERE id=?",
                 (json.dumps(list(seen_hashes)), json.dumps(question_pool), session_id))
    conn.commit()
    score = session["score"] or 0; total = session["total"] or 0
    conn.close()
    return {"attempt_id": attempt_id, "question": q_data["question"], "options": q_data["options"],
            "difficulty": q_data.get("difficulty", "hard"),
            "running_score": score, "running_total": total,
            "running_pct": round(score / max(total, 1) * 100, 1) if total else 0}

@app.post("/api/session/dynamic/answer")
def dynamic_answer(req: DynamicAnswerReq, u=Depends(get_current_user)):
    conn    = get_db()
    attempt = conn.execute("SELECT * FROM dynamic_attempts WHERE id=? AND user_id=?",
                           (req.attempt_id, u["uid"])).fetchone()
    if not attempt:
        conn.close()
        raise HTTPException(404, "Attempt not found")
    attempt = dict(attempt)
    if attempt["user_answer"] is not None:
        conn.close()
        return {"message": "Already answered", "is_correct": bool(attempt["is_correct"]),
                "correct_answer": attempt["correct_answer"], "explanation": attempt["explanation"]}
    session = conn.execute("SELECT * FROM dynamic_sessions WHERE id=? AND user_id=?",
                           (attempt["session_id"], u["uid"])).fetchone()
    if not session or not session["is_active"]:
        conn.close()
        raise HTTPException(400, "Session is not active")
    is_correct = 1 if req.user_answer == attempt["correct_answer"] else 0
    conn.execute("UPDATE dynamic_attempts SET user_answer=?, is_correct=?, time_taken=? WHERE id=?",
                 (req.user_answer, is_correct, req.time_taken, req.attempt_id))
    new_score = (session["score"] or 0) + is_correct
    new_total = (session["total"] or 0) + 1
    conn.execute("UPDATE dynamic_sessions SET score=?, total=? WHERE id=?",
                 (new_score, new_total, attempt["session_id"]))
    conn.commit()
    if is_correct: award_coins(conn, u["uid"], COINS_PER_CORRECT)
    consec = update_consecutive_correct(conn, u["uid"], bool(is_correct))
    xp_gain = 15 if is_correct else 3
    award_xp(conn, u["uid"], xp_gain, "dynamic_question")
    if req.time_taken > 0:
        conn.execute(
            "INSERT INTO question_latency (user_id,question_id,session_type,time_taken_sec,is_correct,chapter,difficulty) VALUES (?,?,?,?,?,?,?)",
            (u["uid"], None, "dynamic", req.time_taken, is_correct, session["chapter"], "hard"))
        conn.commit()
    lat_alert = None
    if req.time_taken > LATENCY_SLOW_SINGLE:
        lat_alert = {"type": "slow_single",
                     "message": f"This took {req.time_taken}s. Aim for under {LATENCY_SLOW_SINGLE}s.",
                     "tip": "Eliminate 2 options first, then decide."}
    coaching_trigger = None
    if not is_correct:
        opts = json.loads(attempt["options"]) if isinstance(attempt["options"], str) else attempt["options"]
        coaching_trigger = {"should_coach": True, "question_id": None,
                             "question_text": attempt["question_text"], "options": opts,
                             "correct_answer": attempt["correct_answer"], "user_answer": req.user_answer,
                             "explanation": attempt["explanation"], "chapter": session["chapter"] or "",
                             "message": "You got this wrong — your AI coach can help you understand it!"}
    conn.close()
    return {"is_correct": bool(is_correct), "correct_answer": attempt["correct_answer"],
            "explanation": attempt["explanation"], "xp_earned": xp_gain,
            "coins_earned": COINS_PER_CORRECT if is_correct else 0,
            "running_score": new_score, "running_total": new_total,
            "running_pct": round(new_score / max(new_total, 1) * 100, 1),
            "consecutive_correct": consec, "latency_alert": lat_alert,
            "coaching_trigger": coaching_trigger}

@app.post("/api/session/dynamic/stop")
def dynamic_stop(req: DynamicStopReq, u=Depends(get_current_user)):
    conn    = get_db()
    session = conn.execute("SELECT * FROM dynamic_sessions WHERE id=? AND user_id=?",
                           (req.session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    session = dict(session)
    if not session["is_active"]:
        attempts = conn.execute("SELECT * FROM dynamic_attempts WHERE session_id=? ORDER BY id",
                                (req.session_id,)).fetchall()
        conn.close()
        return _dynamic_summary(session, attempts, [])
    conn.execute("UPDATE dynamic_sessions SET is_active=0, ended_at=?, question_pool='[]' WHERE id=?",
                 (datetime.now().isoformat(), req.session_id))
    conn.commit()
    attempts = conn.execute("SELECT * FROM dynamic_attempts WHERE session_id=? ORDER BY id",
                            (req.session_id,)).fetchall()
    score = session["score"] or 0; total = session["total"] or 0
    bonus_xp = min(total * 2, 100)
    award_xp(conn, u["uid"], bonus_xp, "dynamic_session_end")
    new_achs = check_and_award_achievements(conn, u["uid"], {
        "score": score, "total": total, "mode": "dynamic", "dyn_total": total})
    lat_stats  = compute_latency_stats(conn, u["uid"], 30)
    lat_alerts = get_latency_alerts(lat_stats)
    conn.close()
    return _dynamic_summary(session, attempts, new_achs, bonus_xp, lat_alerts)

def _dynamic_summary(session, attempts, new_achs, bonus_xp=0, lat_alerts=None):
    score = session["score"] or 0; total = session["total"] or 0
    pct   = round(score / max(total, 1) * 100, 1) if total else 0
    results = []
    for a in attempts:
        a = dict(a)
        if a["user_answer"] is None: continue
        results.append({"attempt_id": a["id"], "question": a["question_text"],
                        "options": json.loads(a["options"]) if isinstance(a["options"], str) else a["options"],
                        "correct_answer": a["correct_answer"], "user_answer": a["user_answer"],
                        "is_correct": bool(a["is_correct"]), "explanation": a["explanation"],
                        "time_taken": a["time_taken"]})
    return {"session_id": session["id"], "chapter": session["chapter"],
            "score": score, "total": total, "percentage": pct, "bonus_xp": bonus_xp,
            "results": results, "new_achievements": new_achs,
            "latency_alerts": (lat_alerts or [])[:2],
            "performance": ("excellent" if pct >= 85 else "good" if pct >= 65 else "needs_practice")}

@app.get("/api/session/dynamic/{session_id}/status")
def dynamic_status(session_id: int, u=Depends(get_current_user)):
    conn    = get_db()
    session = conn.execute("SELECT * FROM dynamic_sessions WHERE id=? AND user_id=?",
                           (session_id, u["uid"])).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    attempts = conn.execute("SELECT id, is_correct, user_answer, time_taken FROM dynamic_attempts WHERE session_id=? ORDER BY id",
                            (session_id,)).fetchall()
    conn.close()
    score = session["score"] or 0; total = session["total"] or 0
    pool  = json.loads(session["question_pool"] or "[]")
    return {"session_id": session_id, "is_active": bool(session["is_active"]),
            "chapter": session["chapter"], "score": score, "total": total,
            "percentage": round(score / max(total, 1) * 100, 1) if total else 0,
            "questions_seen": len(json.loads(session["seen_hashes"] or "[]")),
            "pool_remaining": len(pool), "attempts": [dict(a) for a in attempts]}

@app.get("/api/session/dynamic/history")
def dynamic_history(u=Depends(get_current_user)):
    conn  = get_db()
    rows  = conn.execute(
        "SELECT id, chapter, exam_type, score, total, is_active, created_at, ended_at "
        "FROM dynamic_sessions WHERE user_id=? ORDER BY id DESC LIMIT 20", (u["uid"],)).fetchall()
    conn.close()
    return {"sessions": [
        {"session_id": r["id"], "chapter": r["chapter"], "exam_type": r["exam_type"],
         "score": r["score"] or 0, "total": r["total"] or 0,
         "percentage": round((r["score"] or 0) / max(r["total"] or 1, 1) * 100, 1),
         "is_active": bool(r["is_active"]), "created_at": r["created_at"], "ended_at": r["ended_at"]}
        for r in rows]}

# ── AI COACH ──────────────────────────────────────────────────────────────────

def _get_coach_memory(conn, user_id: int) -> str:
    row = conn.execute("SELECT summary FROM coach_memory WHERE user_id=?", (user_id,)).fetchone()
    return row["summary"] if row else ""

def _compress_coach_memory(conn, user_id: int, name: str, exam_type: str):
    messages = conn.execute(
        "SELECT msg_type, message FROM coach_messages WHERE user_id=? ORDER BY id DESC LIMIT 60",
        (user_id,)).fetchall()
    if len(messages) < 5: return
    convo_text = "\n".join(f"[{m['msg_type'].upper()}]: {m['message'][:200]}" for m in reversed(messages))
    prompt = f"""Summarize this AI coach conversation for {name} ({exam_type} student) in 150 words.
Extract: 1.Topics struggled with 2.Confidence level 3.Learning style 4.Weak areas 5.Goals 6.Progress 7.Speed issues
Conversation:\n{convo_text}\nPlain text only."""
    try:
        summary = groq_chat(prompt, temperature=0.4, json_mode=False)
        conn.execute("INSERT OR REPLACE INTO coach_memory (user_id, summary, updated_at) VALUES (?,?,?)",
                     (user_id, summary, datetime.now().isoformat()))
        conn.commit()
    except Exception as exc:
        logger.error("Memory compression failed: %s", exc)

def _build_coach_system_prompt(u: dict, conn, memory: str, question_ctx: Optional[dict]) -> str:
    uid = u["uid"]
    total_att  = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=?", (uid,)).fetchone()[0]
    total_cor  = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=? AND is_correct=1", (uid,)).fetchone()[0]
    acc        = round(total_cor / max(total_att, 1) * 100, 1) if total_att else 0
    recent_tests = conn.execute("SELECT score, total FROM daily_tests WHERE user_id=? AND completed=1 ORDER BY id DESC LIMIT 5", (uid,)).fetchall()
    recent_scores = [round(t["score"] / max(t["total"], 1) * 100, 1) for t in recent_tests]
    weak_chapters = conn.execute("""
        SELECT q.chapter, SUM(CASE WHEN ta.is_correct=0 THEN 1 ELSE 0 END) as wc
        FROM test_attempts ta JOIN questions q ON ta.question_id=q.id WHERE ta.user_id=?
        GROUP BY q.chapter ORDER BY wc DESC LIMIT 3
    """, (uid,)).fetchall()
    weak_list = ", ".join(r["chapter"] for r in weak_chapters) if weak_chapters else "None yet"
    streak   = get_streak(conn, uid)
    consec   = conn.execute("SELECT consecutive_correct FROM users WHERE id=?", (uid,)).fetchone()
    consec_n = consec["consecutive_correct"] if consec else 0
    lat_stats = compute_latency_stats(conn, uid, 30)
    lat_note = f"\nDecision speed: avg {lat_stats['avg_time']}s/question (trend: {lat_stats['trend']})" if lat_stats["total_tracked"] >= 5 else ""
    memory_block = f"\n\nWhat I remember:\n{memory}" if memory else ""
    q_block = ""
    if question_ctx:
        opts = question_ctx.get("options", [])
        if isinstance(opts, str):
            try: opts = json.loads(opts)
            except Exception: opts = []
        ua = question_ctx.get("user_answer")
        ca = question_ctx.get("correct_answer")
        user_chose  = opts[ua] if ua is not None and isinstance(ua, int) and 0 <= ua < len(opts) else "No answer"
        correct_opt = opts[ca] if ca is not None and isinstance(ca, int) and 0 <= ca < len(opts) else "Unknown"
        q_block = f"\nTHE QUESTION THEY GOT WRONG:\nQ: {question_ctx.get('question','')}\nThey chose: \"{user_chose}\" ❌\nCorrect: \"{correct_opt}\" ✓\nExplanation: {question_ctx.get('explanation','')}"
    return f"""You are ExamAI Coach — friendly, supportive, exam-focused.
STUDENT: {u.get('name','Student')} | Exam: {u.get('exam_type','General')}
Stats: {total_att} questions | {acc}% accuracy | {streak}-day streak | {consec_n} in a row
Recent: {recent_scores or 'No tests yet'} | Weak: {weak_list}{lat_note}{memory_block}{q_block}
STYLE: Talk like a friendly teacher. 3-5 sentences max. Use actual data. Occasional emojis. Warm but honest."""

@app.post("/api/coach/chat")
def coach_chat(req: CoachChatReq, u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]
    memory = _get_coach_memory(conn, uid)
    system_prompt = _build_coach_system_prompt(u, conn, memory, req.question_context)
    recent = list(reversed(conn.execute(
        "SELECT msg_type, message FROM coach_messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (uid, COACH_MEMORY_LIMIT)).fetchall()))
    messages = [{"role": "system", "content": system_prompt}]
    for m in recent:
        messages.append({"role": "user" if m["msg_type"] == "user" else "assistant", "content": m["message"]})
    messages.append({"role": "user", "content": req.message})
    response_text = groq_chat_with_history(messages, temperature=0.75)
    q_ctx_str = json.dumps(req.question_context) if req.question_context else None
    conn.execute("INSERT INTO coach_messages (user_id, message, msg_type, question_context) VALUES (?,?,'user',?)",
                 (uid, req.message, q_ctx_str))
    conn.execute("INSERT INTO coach_messages (user_id, message, msg_type) VALUES (?,?,'coach')", (uid, response_text))
    conn.commit()
    msg_count = conn.execute("SELECT COUNT(*) FROM coach_messages WHERE user_id=?", (uid,)).fetchone()[0]
    if msg_count > 0 and msg_count % 40 == 0:
        _compress_coach_memory(conn, uid, u.get("name", "Student"), u.get("exam_type", "exam"))
    new_achs = check_and_award_achievements(conn, uid, {"mode": "coach"})
    conn.close()
    return {"response": response_text, "new_achievements": new_achs}

@app.get("/api/coach/history")
def coach_history(limit: int = Query(20, ge=1, le=100), u=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, msg_type, message, question_context, created_at FROM coach_messages "
        "WHERE user_id=? ORDER BY id DESC LIMIT ?", (u["uid"], limit)).fetchall()
    conn.close()
    return {"messages": [
        {"id": r["id"], "type": r["msg_type"], "message": r["message"],
         "question_context": json.loads(r["question_context"]) if r["question_context"] else None,
         "created_at": r["created_at"]}
        for r in reversed(rows)]}

@app.get("/api/coach/daily-insight")
def daily_insight(u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]
    total_att = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=?", (uid,)).fetchone()[0]
    total_cor = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=? AND is_correct=1", (uid,)).fetchone()[0]
    acc       = round(total_cor / max(total_att, 1) * 100, 1) if total_att else 0
    streak    = get_streak(conn, uid)
    today     = date.today().isoformat()
    today_test = conn.execute(
        "SELECT score, total FROM daily_tests WHERE user_id=? AND test_date=? AND completed=1",
        (uid, today)).fetchone()
    weak_chapters = conn.execute("""
        SELECT q.chapter, SUM(CASE WHEN ta.is_correct=0 THEN 1 ELSE 0 END) as wc
        FROM test_attempts ta JOIN questions q ON ta.question_id=q.id WHERE ta.user_id=?
        GROUP BY q.chapter ORDER BY wc DESC LIMIT 3
    """, (uid,)).fetchall()
    weak_list = ", ".join(r["chapter"] for r in weak_chapters) if weak_chapters else "None yet"
    memory    = _get_coach_memory(conn, uid)
    lat_stats = compute_latency_stats(conn, uid, 30)
    lat_note  = f" Average decision speed: {lat_stats['avg_time']}s (trend: {lat_stats['trend']})." if lat_stats["total_tracked"] >= 5 else ""

    today_summary = ""
    if today_test:
        pct = round((today_test["score"] or 0) / max(today_test["total"] or 1, 1) * 100, 1)
        today_summary = f" Today's test: {today_test['score']}/{today_test['total']} ({pct}%)."

    prompt = f"""You are ExamAI Coach giving a personalized daily insight for {u.get('name','Student')} 
({u.get('exam_type','General')} exam student).
Stats: {total_att} total questions | {acc}% accuracy | {streak}-day streak.{today_summary}
Weak chapters: {weak_list}.{lat_note}
{f'Memory: {memory[:300]}' if memory else ''}

Write a motivating, personalized 3-4 sentence daily insight. Include:
1. One observation about their performance
2. One specific tip for today
3. One encouraging note

Be specific, use their actual numbers, warm but concise."""

    try:
        insight = groq_chat(prompt, temperature=0.7, json_mode=False)
    except Exception as exc:
        logger.error("Daily insight failed: %s", exc)
        # ❌ BROKEN — the ! is inside the f-string expression
        # ✅ FIXED — extract the name first, then use it
        name = u.get('name', '')
        insight = f"Keep going, {name}! You've answered {total_att} questions with {acc}% accuracy. Focus on your weak areas today and stay consistent!"

    conn.close()
    return {
        "insight": insight,
        "stats": {
            "total_attempted": total_att, "accuracy": acc,
            "streak": streak, "today_done": today_test is not None,
        },
    }


# ── ANALYTICS ─────────────────────────────────────────────────────────────────

@app.get("/api/analytics")
def analytics(u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]

    # Overall stats
    total_att = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=?", (uid,)).fetchone()[0]
    total_cor = conn.execute("SELECT COUNT(*) FROM test_attempts WHERE user_id=? AND is_correct=1", (uid,)).fetchone()[0]
    acc = round(total_cor / max(total_att, 1) * 100, 1) if total_att else 0

    # Daily test history (last 14 days)
    daily_hist = conn.execute("""
        SELECT test_date, score, total, chapter_name,
               ROUND(CAST(score AS FLOAT)/MAX(total,1)*100, 1) as pct
        FROM daily_tests WHERE user_id=? AND completed=1
        ORDER BY test_date DESC LIMIT 14
    """, (uid,)).fetchall()

    # Chapter performance
    chapter_perf = conn.execute("""
        SELECT q.chapter,
               COUNT(ta.id)                                          AS total_attempts,
               SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END)     AS correct,
               ROUND(CAST(SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END) AS FLOAT)
                     / MAX(COUNT(ta.id),1)*100, 1)                  AS accuracy
        FROM test_attempts ta JOIN questions q ON ta.question_id=q.id
        WHERE ta.user_id=?
        GROUP BY q.chapter ORDER BY accuracy DESC
    """, (uid,)).fetchall()

    # Difficulty breakdown
    diff_perf = conn.execute("""
        SELECT q.difficulty,
               COUNT(ta.id) as total,
               SUM(CASE WHEN ta.is_correct=1 THEN 1 ELSE 0 END) as correct
        FROM test_attempts ta JOIN questions q ON ta.question_id=q.id
        WHERE ta.user_id=?
        GROUP BY q.difficulty
    """, (uid,)).fetchall()

    # Weekly trend (last 7 days activity)
    weekly = conn.execute("""
        SELECT date(attempted_at) as day,
               COUNT(*) as total,
               SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as correct
        FROM test_attempts WHERE user_id=?
        AND attempted_at >= date('now', '-7 days')
        GROUP BY day ORDER BY day ASC
    """, (uid,)).fetchall()

    # Top weak questions
    weak_qs = conn.execute("""
        SELECT q.id, q.question, q.chapter,
               SUM(CASE WHEN ta.is_correct=0 THEN 1 ELSE 0 END) as wrong_count
        FROM questions q JOIN test_attempts ta ON ta.question_id=q.id AND ta.user_id=?
        WHERE q.user_id=?
        GROUP BY q.id HAVING wrong_count >= 2
        ORDER BY wrong_count DESC LIMIT 5
    """, (uid, uid)).fetchall()

    streak    = get_streak(conn, uid)
    lat_stats = compute_latency_stats(conn, uid, 30)
    conn.close()

    return {
        "overview": {
            "total_attempted": total_att,
            "total_correct": total_cor,
            "overall_accuracy": acc,
            "study_streak": streak,
        },
        "daily_history": [dict(r) for r in daily_hist],
        "chapter_performance": [
            {"chapter": r["chapter"], "total_attempts": r["total_attempts"],
             "correct": r["correct"], "accuracy": r["accuracy"]}
            for r in chapter_perf
        ],
        "difficulty_breakdown": [
            {"difficulty": r["difficulty"], "total": r["total"], "correct": r["correct"],
             "accuracy": round(r["correct"] / max(r["total"], 1) * 100, 1)}
            for r in diff_perf
        ],
        "weekly_activity": [dict(r) for r in weekly],
        "top_weak_questions": [
            {"id": r["id"], "question": r["question"][:120] + "...",
             "chapter": r["chapter"], "wrong_count": r["wrong_count"]}
            for r in weak_qs
        ],
        "latency": lat_stats,
    }


# ── STUDY PLAN ────────────────────────────────────────────────────────────────

@app.post("/api/study-plan")
def generate_study_plan(req: StudyPlanReq, u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]

    chapters_row = conn.execute("SELECT chapters FROM users WHERE id=?", (uid,)).fetchone()
    chapters = json.loads(chapters_row["chapters"]) if chapters_row and chapters_row["chapters"] else []

    chapter_data = []
    for ch in chapters:
        att, cor, acc = chapter_stats(conn, uid, ch)
        subs_p, subs_t = get_sub_chapter_progress(conn, uid, ch)
        chapter_data.append({"name": ch, "attempted": att, "accuracy": acc,
                              "subs_practiced": subs_p, "subs_total": subs_t})

    weak_chapters = [c for c in chapter_data if c["accuracy"] < 60 and c["attempted"] > 0]
    untouched     = [c for c in chapter_data if c["attempted"] == 0]
    focus = req.focus_chapters or [c["name"] for c in (weak_chapters or untouched)[:3]]

    exam_date_note = f"Exam date: {req.exam_date}." if req.exam_date else "No exam date set."
    hours_note     = f"Daily study time available: {req.daily_hours} hours."

    prompt = f"""You are an expert study planner for {u.get('exam_type','General')}.
Student: {u.get('name','Student')}
{exam_date_note} {hours_note}
Chapters: {json.dumps([c['name'] for c in chapter_data])}
Weak chapters (< 60% accuracy): {[c['name'] for c in weak_chapters]}
Untouched chapters: {[c['name'] for c in untouched]}
Focus areas requested: {focus}

Create a practical weekly study plan. Structure it day by day (Monday–Sunday).
Each day: 1 main chapter focus, specific sub-topics, time allocation, goals.
Prioritize weak areas. Include rest days.
Return as plain text with clear Day headings. Be specific and actionable. Max 400 words."""

    try:
        plan_text = groq_chat(prompt, temperature=0.6, json_mode=False)
    except Exception as exc:
        logger.error("Study plan generation failed: %s", exc)
        plan_text = "Could not generate plan right now. Focus on your weakest chapter daily for 1 hour, then practice questions."

    conn.execute("INSERT OR REPLACE INTO study_plans (user_id, plan, updated_at) VALUES (?,?,?)",
                 (uid, plan_text, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"plan": plan_text, "exam_date": req.exam_date, "daily_hours": req.daily_hours,
            "focus_chapters": focus, "generated_at": datetime.now().isoformat()}


@app.get("/api/study-plan")
def get_study_plan(u=Depends(get_current_user)):
    conn = get_db()
    row  = conn.execute("SELECT plan, updated_at FROM study_plans WHERE user_id=?", (u["uid"],)).fetchone()
    conn.close()
    if not row:
        return {"plan": None, "updated_at": None, "message": "No study plan yet. Generate one!"}
    return {"plan": row["plan"], "updated_at": row["updated_at"]}


# ── BOOKMARKS ─────────────────────────────────────────────────────────────────

@app.post("/api/bookmarks")
def add_bookmark(req: BookmarkReq, u=Depends(get_current_user)):
    conn = get_db()
    q = conn.execute("SELECT id FROM questions WHERE id=? AND user_id=?",
                     (req.question_id, u["uid"])).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")
    try:
        conn.execute("INSERT INTO bookmarks (user_id, question_id, note) VALUES (?,?,?)",
                     (u["uid"], req.question_id, req.note[:500]))
        conn.commit()
        new_achs = check_and_award_achievements(conn, u["uid"], {"mode": "bookmark"})
        conn.close()
        return {"success": True, "message": "Bookmarked!", "new_achievements": new_achs}
    except sqlite3.IntegrityError:
        conn.execute("UPDATE bookmarks SET note=? WHERE user_id=? AND question_id=?",
                     (req.note[:500], u["uid"], req.question_id))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Bookmark updated"}


@app.delete("/api/bookmarks/{question_id}")
def remove_bookmark(question_id: int, u=Depends(get_current_user)):
    conn = get_db()
    conn.execute("DELETE FROM bookmarks WHERE user_id=? AND question_id=?", (u["uid"], question_id))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Bookmark removed"}


@app.get("/api/bookmarks")
def get_bookmarks(u=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("""
        SELECT b.question_id, b.note, b.created_at,
               q.question, q.options, q.correct_answer, q.explanation, q.chapter, q.difficulty
        FROM bookmarks b JOIN questions q ON b.question_id=q.id
        WHERE b.user_id=? ORDER BY b.created_at DESC
    """, (u["uid"],)).fetchall()
    conn.close()
    return {"bookmarks": [
        {"question_id": r["question_id"], "note": r["note"], "created_at": r["created_at"],
         "question": r["question"], "chapter": r["chapter"], "difficulty": r["difficulty"],
         "options": json.loads(r["options"]) if isinstance(r["options"], str) else r["options"],
         "correct_answer": r["correct_answer"], "explanation": r["explanation"]}
        for r in rows
    ], "total": len(rows)}


# ── REPORT QUESTION ───────────────────────────────────────────────────────────

@app.post("/api/questions/report")
def report_question(req: ReportQuestionReq, u=Depends(get_current_user)):
    conn = get_db()
    q = conn.execute("SELECT id FROM questions WHERE id=? AND user_id=?",
                     (req.question_id, u["uid"])).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")
    conn.execute("INSERT INTO question_reports (user_id, question_id, reason) VALUES (?,?,?)",
                 (u["uid"], req.question_id, req.reason[:500]))
    conn.execute("UPDATE questions SET report_count = report_count + 1 WHERE id=?", (req.question_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Report submitted. Thank you for the feedback!"}


# ── HINT / POWER-UPS ──────────────────────────────────────────────────────────

@app.post("/api/powerup/use")
def use_powerup(req: UseHintReq, u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]

    q = conn.execute("SELECT * FROM questions WHERE id=? AND user_id=?",
                     (req.question_id, uid)).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")
    q = dict(q)
    opts = json.loads(q["options"])

    if req.hint_type == "hint":
        cost = HINT_COST_COINS
        if not spend_coins(conn, uid, cost):
            conn.close()
            raise HTTPException(400, f"Not enough coins. Hint costs {cost} coins.")
        # Generate a hint via AI
        prompt = f"""For this MCQ question:
Q: {q['question']}
Options: {json.dumps(opts)}
Correct answer index: {q['correct_answer']}

Give a short, helpful hint (1-2 sentences) that guides the student WITHOUT revealing the answer.
Plain text only."""
        try:
            hint_text = groq_chat(prompt, temperature=0.5, json_mode=False)
        except Exception:
            hint_text = f"Think about the key concept related to {q['chapter']}."
        conn.execute("INSERT INTO powerup_uses (user_id, powerup_type, question_id, result) VALUES (?,?,?,?)",
                     (uid, "hint", req.question_id, hint_text))
        conn.commit()
        coins_row = conn.execute("SELECT coins FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        return {"type": "hint", "hint": hint_text, "cost": cost,
                "coins_remaining": coins_row["coins"] if coins_row else 0}

    elif req.hint_type == "fifty_fifty":
        cost = FIFTY_FIFTY_COST_COINS
        if not spend_coins(conn, uid, cost):
            conn.close()
            raise HTTPException(400, f"Not enough coins. 50/50 costs {cost} coins.")
        correct = q["correct_answer"]
        wrong_indices = [i for i in range(4) if i != correct]
        remove_two = random.sample(wrong_indices, min(2, len(wrong_indices)))
        remaining  = [i for i in range(4) if i not in remove_two]
        conn.execute("INSERT INTO powerup_uses (user_id, powerup_type, question_id, result) VALUES (?,?,?,?)",
                     (uid, "fifty_fifty", req.question_id, json.dumps(remaining)))
        conn.commit()
        coins_row = conn.execute("SELECT coins FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        return {"type": "fifty_fifty", "remaining_options": remaining,
                "eliminated_options": remove_two, "cost": cost,
                "coins_remaining": coins_row["coins"] if coins_row else 0}

    else:
        conn.close()
        raise HTTPException(400, "Invalid hint_type. Use 'hint' or 'fifty_fifty'.")


# ── DAILY CHALLENGE ───────────────────────────────────────────────────────────

@app.get("/api/challenge/daily")
def get_daily_challenge(u=Depends(get_current_user)):
    conn  = get_db()
    uid   = u["uid"]
    today = date.today().isoformat()

    existing = conn.execute(
        "SELECT * FROM daily_challenges WHERE user_id=? AND challenge_date=?",
        (uid, today)).fetchone()
    if existing:
        existing = dict(existing)
        conn.close()
        return {
            "challenge_date": existing["challenge_date"],
            "question": existing["question_text"],
            "options": json.loads(existing["options"]) if isinstance(existing["options"], str) else existing["options"],
            "chapter": existing["chapter"],
            "completed": bool(existing["completed"]),
            "user_answer": existing["user_answer"],
            "correct_answer": existing["correct_answer"] if existing["completed"] else None,
            "explanation": existing["explanation"] if existing["completed"] else None,
            "xp_reward": existing["xp_reward"],
        }

    # Generate a fresh challenge question
    exam_row  = conn.execute("SELECT exam_type, exam_language, chapters FROM users WHERE id=?", (uid,)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    chapters  = json.loads(exam_row["chapters"] or "[]") if exam_row and exam_row["chapters"] else []
    lang_instruction = get_language_instruction(lang_key)

    # Pick a random chapter to challenge on
    chapter = random.choice(chapters) if chapters else exam_type
    style_hints = get_exam_style_hints(exam_type)

    prompt = f"""You are an expert question setter for {exam_type}.{lang_instruction}
Generate 1 CHALLENGING daily challenge question about "{chapter}".
Make it harder than usual — test deep understanding, not just recall.

Return ONLY this JSON:
{{
  "question": "Question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": 2,
  "explanation": "Detailed explanation",
  "difficulty": "hard"
}}"""

    try:
        raw    = groq_chat(prompt, temperature=0.8, json_mode=True)
        q_data = parse_json(raw)
        vq     = validate_question_dict(q_data)
    except Exception as exc:
        logger.error("Daily challenge generation failed: %s", exc)
        vq = None

    if not vq:
        # Fallback: pick a random hard question from DB
        db_q = conn.execute(
            "SELECT * FROM questions WHERE user_id=? AND difficulty='hard' ORDER BY RANDOM() LIMIT 1",
            (uid,)).fetchone()
        if not db_q:
            db_q = conn.execute(
                "SELECT * FROM questions WHERE user_id=? ORDER BY RANDOM() LIMIT 1",
                (uid,)).fetchone()
        if db_q:
            db_q = dict(db_q)
            vq = {"question": db_q["question"],
                  "options": json.loads(db_q["options"]) if isinstance(db_q["options"], str) else db_q["options"],
                  "correct_answer": db_q["correct_answer"],
                  "explanation": db_q["explanation"], "difficulty": "hard"}
            chapter = db_q["chapter"]

    if not vq:
        conn.close()
        return {"challenge_date": today, "question": None,
                "message": "No challenge available today. Generate some questions first!"}

    conn.execute(
        "INSERT INTO daily_challenges (user_id,challenge_date,question_text,options,correct_answer,explanation,chapter,xp_reward) VALUES (?,?,?,?,?,?,?,?)",
        (uid, today, vq["question"], json.dumps(vq["options"]),
         vq["correct_answer"], vq["explanation"], chapter, DAILY_CHALLENGE_XP))
    conn.commit()
    conn.close()
    return {
        "challenge_date": today,
        "question": vq["question"],
        "options": vq["options"],
        "chapter": chapter,
        "completed": False,
        "user_answer": None,
        "correct_answer": None,
        "explanation": None,
        "xp_reward": DAILY_CHALLENGE_XP,
    }


@app.post("/api/challenge/daily/submit")
def submit_daily_challenge(req: DailyChallengeSubmitReq, u=Depends(get_current_user)):
    conn  = get_db()
    uid   = u["uid"]
    today = date.today().isoformat()

    challenge = conn.execute(
        "SELECT * FROM daily_challenges WHERE user_id=? AND challenge_date=?",
        (uid, today)).fetchone()
    if not challenge:
        conn.close()
        raise HTTPException(404, "No daily challenge found for today. Request one first.")
    challenge = dict(challenge)
    if challenge["completed"]:
        conn.close()
        return {
            "already_completed": True,
            "is_correct": req.user_answer == challenge["correct_answer"],
            "correct_answer": challenge["correct_answer"],
            "explanation": challenge["explanation"],
            "xp_reward": 0,
        }

    is_correct = req.user_answer == challenge["correct_answer"]
    conn.execute(
        "UPDATE daily_challenges SET completed=1, user_answer=? WHERE user_id=? AND challenge_date=?",
        (req.user_answer, uid, today))
    conn.commit()

    xp_earned = challenge["xp_reward"] if is_correct else 10
    award_xp(conn, uid, xp_earned, "daily_challenge")
    if is_correct:
        award_coins(conn, uid, COINS_PER_CORRECT * 3)  # Triple coins for daily challenge
    new_achs = check_and_award_achievements(conn, uid, {"mode": "challenge"})
    conn.close()
    return {
        "is_correct": is_correct,
        "correct_answer": challenge["correct_answer"],
        "explanation": challenge["explanation"],
        "xp_earned": xp_earned,
        "coins_earned": COINS_PER_CORRECT * 3 if is_correct else 0,
        "new_achievements": new_achs,
    }


# ── MOOD CHECK-IN ─────────────────────────────────────────────────────────────

@app.post("/api/mood/checkin")
def mood_checkin(req: MoodCheckinReq, u=Depends(get_current_user)):
    if not 1 <= req.mood <= 5:
        raise HTTPException(400, "mood must be 1–5")
    if not 1 <= req.energy <= 5:
        raise HTTPException(400, "energy must be 1–5")
    conn = get_db()
    conn.execute(
        "INSERT INTO mood_checkins (user_id, mood, energy, note) VALUES (?,?,?,?)",
        (u["uid"], req.mood, req.energy, req.note[:300]))
    conn.commit()
    new_achs = check_and_award_achievements(conn, u["uid"], {"mode": "mood"})
    conn.close()
    return {"success": True, "message": "Mood logged!", "new_achievements": new_achs}


@app.get("/api/mood/history")
def mood_history(limit: int = Query(14, ge=1, le=60), u=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT mood, energy, note, checkin_date FROM mood_checkins "
        "WHERE user_id=? ORDER BY checkin_date DESC LIMIT ?",
        (u["uid"], limit)).fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows], "total": len(rows)}


# ── TEACH ME MODE ─────────────────────────────────────────────────────────────

@app.post("/api/teach-me/start")
def teach_me_start(req: TeachMeReq, u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]

    # Resolve question data from DB or request
    if req.question_id:
        q = conn.execute("SELECT * FROM questions WHERE id=? AND user_id=?",
                         (req.question_id, uid)).fetchone()
        if not q:
            conn.close()
            raise HTTPException(404, "Question not found")
        q = dict(q)
        question_text   = q["question"]
        options         = json.loads(q["options"]) if isinstance(q["options"], str) else q["options"]
        correct_answer  = q["correct_answer"]
        explanation     = q["explanation"]
        chapter         = q["chapter"]
        user_answer     = req.user_answer
    else:
        question_text   = req.question_text or ""
        options         = req.options or []
        correct_answer  = req.correct_answer if req.correct_answer is not None else 0
        explanation     = req.explanation or ""
        chapter         = req.chapter or ""
        user_answer     = req.user_answer

    if not question_text:
        conn.close()
        raise HTTPException(400, "question_text or question_id is required")

    conn.execute(
        "INSERT INTO teach_sessions (user_id,question_id,question_text,correct_answer,user_answer,options,explanation,chapter) VALUES (?,?,?,?,?,?,?,?)",
        (uid, req.question_id, question_text, correct_answer, user_answer,
         json.dumps(options), explanation, chapter))
    conn.commit()
    session_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (uid,)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)

    opts_display = "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(options))
    correct_opt  = options[correct_answer] if 0 <= correct_answer < len(options) else "N/A"
    user_opt     = options[user_answer] if user_answer is not None and 0 <= user_answer < len(options) else "No answer given"

    prompt = f"""You are a patient, encouraging tutor for {exam_type}.{lang_instruction}

A student got this question WRONG and wants to learn from it:

QUESTION: {question_text}
OPTIONS:
{opts_display}

The student chose: {user_opt} ❌
Correct answer: {correct_opt} ✓

Official explanation: {explanation}

Your task — teach this concept in a way the student will REMEMBER it:
1. Start with WHY their answer was wrong (kindly)
2. Explain the correct concept clearly with a simple analogy or example
3. Give a memory trick to remember this
4. End with: "Does this make sense? Ask me anything about this!"

Keep it conversational (4-6 sentences), encouraging, and exam-focused."""

    try:
        teaching = groq_chat(prompt, temperature=0.65, json_mode=False)
    except Exception as exc:
        logger.error("Teach me start failed: %s", exc)
        teaching = f"Let me explain this. The correct answer is '{correct_opt}'. {explanation} Do you have any questions about this?"

    conn.execute("INSERT INTO coach_messages (user_id, message, msg_type) VALUES (?,?,'coach')", (uid, teaching))
    conn.commit()
    conn.close()
    return {
        "session_id": session_id,
        "teaching": teaching,
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "correct_option": correct_opt,
        "user_answer": user_answer,
        "user_option": user_opt,
        "chapter": chapter,
    }


@app.post("/api/teach-me/reply")
def teach_me_reply(req: TeachMeReplyReq, u=Depends(get_current_user)):
    conn = get_db()
    uid  = u["uid"]

    session = conn.execute("SELECT * FROM teach_sessions WHERE id=? AND user_id=?",
                           (req.session_id, uid)).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Teach session not found")
    session = dict(session)

    exam_row  = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (uid,)).fetchone()
    exam_type = exam_row["exam_type"] if exam_row else "General"
    lang_key  = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)

    opts = json.loads(session["options"]) if isinstance(session["options"], str) else (session["options"] or [])
    correct_opt = opts[session["correct_answer"]] if 0 <= session["correct_answer"] < len(opts) else "N/A"

    # Build conversation history for this teach session
    recent_msgs = conn.execute(
        "SELECT msg_type, message FROM coach_messages WHERE user_id=? ORDER BY id DESC LIMIT 10",
        (uid,)).fetchall()

    messages = [
        {"role": "system", "content": (
            f"You are a patient tutor for {exam_type}.{lang_instruction} "
            f"You are teaching the concept from: '{session['question_text']}'. "
            f"Correct answer: '{correct_opt}'. "
            f"Explanation: {session['explanation']}. "
            "Answer the student's follow-up questions clearly and concisely (2-4 sentences). "
            "Be warm, encouraging, and exam-focused."
        )}
    ]
    for m in reversed(recent_msgs):
        messages.append({"role": "user" if m["msg_type"] == "user" else "assistant",
                         "content": m["message"]})
    messages.append({"role": "user", "content": req.message})

    try:
        reply = groq_chat_with_history(messages, temperature=0.7)
    except Exception as exc:
        logger.error("Teach me reply failed: %s", exc)
        reply = "Great question! Keep exploring this topic and you'll master it."

    conn.execute("INSERT INTO coach_messages (user_id, message, msg_type) VALUES (?,?,'user')", (uid, req.message))
    conn.execute("INSERT INTO coach_messages (user_id, message, msg_type) VALUES (?,?,'coach')", (uid, reply))

    # Mark session complete after first reply
    if not session["completed"]:
        conn.execute("UPDATE teach_sessions SET completed=1 WHERE id=?", (req.session_id,))

    conn.commit()
    new_achs = check_and_award_achievements(conn, uid, {"mode": "teach"})
    conn.close()
    return {"reply": reply, "session_id": req.session_id, "new_achievements": new_achs}


# ── EXPLAIN QUESTION ──────────────────────────────────────────────────────────

@app.post("/api/questions/explain")
def explain_question(req: ExplainReq, u=Depends(get_current_user)):
    conn = get_db()
    q    = conn.execute("SELECT * FROM questions WHERE id=? AND user_id=?",
                        (req.question_id, u["uid"])).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")
    q = dict(q)
    opts        = json.loads(q["options"]) if isinstance(q["options"], str) else q["options"]
    correct_opt = opts[q["correct_answer"]] if 0 <= q["correct_answer"] < len(opts) else "N/A"
    user_opt    = opts[req.user_answer] if 0 <= req.user_answer < len(opts) else "No answer"
    exam_row    = conn.execute("SELECT exam_type, exam_language FROM users WHERE id=?", (u["uid"],)).fetchone()
    exam_type   = exam_row["exam_type"] if exam_row else "General"
    lang_key    = exam_row["exam_language"] if exam_row else None
    lang_instruction = get_language_instruction(lang_key)

    prompt = f"""Explain this {exam_type} MCQ for a student.{lang_instruction}

Q: {q['question']}
Student chose: {user_opt} ({'✓ CORRECT' if req.user_answer == q['correct_answer'] else '❌ WRONG'})
Correct answer: {correct_opt}

Explanation: {q['explanation']}

Write a 3-4 sentence explanation: 1) confirm/correct their choice, 2) explain the concept, 3) memory tip. Friendly tone."""

    try:
        explanation = groq_chat(prompt, temperature=0.6, json_mode=False)
    except Exception:
        explanation = q["explanation"]

    conn.close()
    return {
        "question_id": req.question_id,
        "question": q["question"],
        "user_answer": req.user_answer,
        "correct_answer": q["correct_answer"],
        "is_correct": req.user_answer == q["correct_answer"],
        "explanation": explanation,
        "chapter": q["chapter"],
    }


# ── LEADERBOARD ───────────────────────────────────────────────────────────────

@app.get("/api/leaderboard")
def leaderboard(u=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("""
        SELECT u.id, u.name, u.xp, u.level, u.avatar_color,
               COUNT(DISTINCT CASE WHEN ta.is_correct=1 THEN ta.id END) as correct_count,
               COUNT(ta.id) as total_count
        FROM users u
        LEFT JOIN test_attempts ta ON ta.user_id = u.id
        WHERE u.setup_done=1
        GROUP BY u.id
        ORDER BY u.xp DESC LIMIT 20
    """).fetchall()
    my_rank = next((i + 1 for i, r in enumerate(rows) if r["id"] == u["uid"]), None)
    conn.close()
    return {
        "leaderboard": [
            {"rank": i + 1, "name": r["name"], "xp": r["xp"], "level": r["level"],
             "avatar_color": r["avatar_color"],
             "accuracy": round(r["correct_count"] / max(r["total_count"], 1) * 100, 1),
             "total_attempted": r["total_count"],
             "is_me": r["id"] == u["uid"]}
            for i, r in enumerate(rows)
        ],
        "my_rank": my_rank,
    }


# ── ADMIN ENDPOINTS (Global Test Management) ──────────────────────────────────

def _require_admin(authorization: str = Header(None)):
    """Simple admin token check — set ADMIN_TOKEN in .env for security."""
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token:
        return True  # No token set → open admin (dev mode)
    if not authorization or authorization.strip() != f"Bearer {admin_token}":
        raise HTTPException(403, "Admin access required")
    return True


@app.post("/api/admin/global-tests/create")
def admin_create_global_test(
    title: str = Query(...),
    exam_type: str = Query(...),
    topic: str = Query(...),
    scheduled_at: str = Query(...),
    starts_at: str = Query(...),
    duration_minutes: int = Query(60),
    question_count: int = Query(20),
    _admin=Depends(_require_admin),
):
    """Admin: create a global test and auto-generate questions."""
    conn = get_db()
    conn.execute(
        "INSERT INTO global_tests (title, exam_type, topic, question_ids, scheduled_at, starts_at, duration_minutes, status) VALUES (?,?,?,?,?,?,?,?)",
        (title, exam_type, topic, "[]", scheduled_at, starts_at, duration_minutes, "generating"))
    conn.commit()
    test_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Generate questions
    lang_key  = detect_exam_language(exam_type)
    lang_instr = get_language_instruction(lang_key)
    web_ctx   = search_topic_context(topic, exam_type, deep=True)
    ctx_block = f'\nREFERENCE:\n"""\n{web_ctx[:3000]}\n"""\n' if web_ctx else ""
    style_hints = get_exam_style_hints(exam_type)
    prompt = build_question_prompt(topic, exam_type, question_count, style_hints,
                                    ctx_block, 0, lang_instr)
    try:
        raw    = groq_chat(prompt, temperature=0.75)
        data   = parse_json(raw)
        raw_qs = data.get("questions", []) if isinstance(data, dict) else data
    except Exception as exc:
        logger.error("Global test question generation failed: %s", exc)
        raw_qs = []

    q_ids = []
    for q in (raw_qs or []):
        vq = validate_question_dict(q)
        if not vq:
            continue
        conn.execute(
            "INSERT INTO global_questions (global_test_id, question, options, correct_answer, explanation, difficulty, topic_tag) VALUES (?,?,?,?,?,?,?)",
            (test_id, vq["question"], json.dumps(vq["options"]),
             vq["correct_answer"], vq["explanation"], vq["difficulty"], topic))
        conn.commit()
        qid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        q_ids.append(qid)

    status = "ready" if q_ids else "draft"
    conn.execute("UPDATE global_tests SET question_ids=?, status=? WHERE id=?",
                 (json.dumps(q_ids), status, test_id))
    conn.commit()
    conn.close()
    return {"test_id": test_id, "title": title, "status": status,
            "questions_generated": len(q_ids), "message": f"Global test created with {len(q_ids)} questions."}


@app.post("/api/admin/global-tests/{test_id}/set-status")
def admin_set_test_status(
    test_id: int,
    status: str = Query(...),
    _admin=Depends(_require_admin),
):
    """Admin: update test status (draft → ready → lobby → live → ended)."""
    VALID = {"draft", "ready", "lobby", "live", "ended"}
    if status not in VALID:
        raise HTTPException(400, f"status must be one of {VALID}")
    conn = get_db()
    row = conn.execute("SELECT id FROM global_tests WHERE id=?", (test_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Test not found")
    now_iso = datetime.now().isoformat()
    if status == "ended":
        # Compute rankings and winner
        parts = conn.execute(
            "SELECT user_id, score, time_taken_sec FROM global_participants "
            "WHERE global_test_id=? AND submitted_at IS NOT NULL "
            "ORDER BY score DESC, time_taken_sec ASC", (test_id,)).fetchall()
        for rank, p in enumerate(parts, 1):
            conn.execute("UPDATE global_participants SET rank=? WHERE global_test_id=? AND user_id=?",
                         (rank, test_id, p["user_id"]))
        if parts:
            winner = dict(parts[0])
            winner_name = conn.execute("SELECT name FROM users WHERE id=?", (winner["user_id"],)).fetchone()
            conn.execute(
                "UPDATE global_tests SET status=?, ended_at=?, winner_user_id=?, winner_name=?, winner_score=? WHERE id=?",
                (status, now_iso, winner["user_id"],
                 winner_name["name"] if winner_name else "Unknown",
                 winner["score"], test_id))
        else:
            conn.execute("UPDATE global_tests SET status=?, ended_at=? WHERE id=?",
                         (status, now_iso, test_id))
    else:
        conn.execute("UPDATE global_tests SET status=? WHERE id=?", (status, test_id))
    conn.commit()
    conn.close()
    return {"test_id": test_id, "status": status, "updated_at": now_iso}


@app.get("/api/admin/global-tests")
def admin_list_global_tests(_admin=Depends(_require_admin)):
    conn = get_db()
    rows = conn.execute(
        "SELECT gt.*, (SELECT COUNT(*) FROM global_participants WHERE global_test_id=gt.id) as participant_count, "
        "(SELECT COUNT(*) FROM global_questions WHERE global_test_id=gt.id) as question_count "
        "FROM global_tests gt ORDER BY gt.id DESC LIMIT 50").fetchall()
    conn.close()
    return {"tests": [dict(r) for r in rows]}


@app.delete("/api/admin/global-tests/{test_id}")
def admin_delete_global_test(test_id: int, _admin=Depends(_require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM global_questions WHERE global_test_id=?", (test_id,))
    conn.execute("DELETE FROM global_participants WHERE global_test_id=?", (test_id,))
    conn.execute("DELETE FROM global_tests WHERE id=?", (test_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Global test {test_id} deleted."}


# ── QUESTION POOL MANAGEMENT ──────────────────────────────────────────────────

@app.get("/api/questions")
def list_questions(
    chapter: Optional[str] = None,
    sub_chapter: Optional[str] = None,
    difficulty: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    u=Depends(get_current_user),
):
    conn  = get_db()
    uid   = u["uid"]
    where = ["q.user_id=?"]
    args  = [uid]
    if chapter:
        where.append("q.chapter=?"); args.append(chapter)
    if sub_chapter:
        where.append("q.sub_chapter=?"); args.append(sub_chapter)
    if difficulty:
        where.append("q.difficulty=?"); args.append(difficulty)
    where_clause = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM questions q WHERE {where_clause}", args).fetchone()[0]
    qs    = conn.execute(
        f"SELECT q.*, "
        f"(SELECT COUNT(*) FROM test_attempts ta WHERE ta.question_id=q.id AND ta.user_id=?) as attempt_count, "
        f"(SELECT SUM(ta.is_correct) FROM test_attempts ta WHERE ta.question_id=q.id AND ta.user_id=?) as correct_count, "
        f"(SELECT 1 FROM bookmarks b WHERE b.question_id=q.id AND b.user_id=?) as is_bookmarked "
        f"FROM questions q WHERE {where_clause} ORDER BY q.created_at DESC LIMIT ? OFFSET ?",
        [uid, uid, uid] + args + [limit, offset]).fetchall()
    conn.close()
    return {
        "questions": [
            {**fmt_q(q), "attempt_count": q["attempt_count"] or 0,
             "correct_count": q["correct_count"] or 0,
             "is_bookmarked": bool(q["is_bookmarked"])}
            for q in qs
        ],
        "total": total, "limit": limit, "offset": offset,
    }


@app.delete("/api/questions/{question_id}")
def delete_question(question_id: int, u=Depends(get_current_user)):
    conn = get_db()
    q = conn.execute("SELECT id FROM questions WHERE id=? AND user_id=?",
                     (question_id, u["uid"])).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")
    conn.execute("DELETE FROM test_attempts WHERE question_id=?", (question_id,))
    conn.execute("DELETE FROM bookmarks WHERE question_id=?", (question_id,))
    conn.execute("DELETE FROM questions WHERE id=?", (question_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Question deleted"}


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": "5.0",
        "groq_keys_configured": len(GROQ_KEYS),
        "openrouter_keys_configured": len(OPENROUTER_KEYS),
        "scheduler_running": HAS_SCHEDULER,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
def root():
    return {"message": "ExamAI API v5 is running!", "docs": "/docs"}


# ── STATIC FILES (serve frontend build) ───────────────────────────────────────
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Get the absolute path of the directory containing this script
BASE_DIR = Path(__file__).resolve().parent
frontend_path = BASE_DIR.parent / "frontend"

print(f"--> Looking for frontend at: {frontend_path}") # Debugging line

if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
else:
    print(f"--> WARNING: Frontend directory not found at {frontend_path}")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)