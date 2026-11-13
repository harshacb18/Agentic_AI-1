from __future__ import annotations

import json
import html
import os
import re
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException

from interview_coach.agents import (
    SYSTEM_PROMPTS,
    USER_PROMPTS,
    AnswerEvaluation,
    InterviewAgents,
)
from interview_coach.archive import InterviewArchive
from interview_coach.curriculum import (
    INTERVIEW_ROUNDS,
    QUESTIONS_PER_ROUND,
    SUGGESTED_COMPANIES,
    SUGGESTED_ROLES,
    build_role_round_csv,
)
from interview_coach.model import build_chat_model
from interview_coach.routing import Intent, detect_intent, should_retrieve

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def configured_model() -> tuple[str, str, str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    provider = os.getenv("AI_PROVIDER", "").strip()
    if not provider:
        provider = "OpenAI API" if api_key else "Ollama (local)"
    if provider == "OpenAI API":
        return (
            provider,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key,
            "",
        )
    if provider == "Ollama (local)":
        return (
            provider,
            os.getenv("OLLAMA_MODEL", "llama3.2"),
            "",
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    raise ValueError("AI_PROVIDER must be set to 'OpenAI API' or 'Ollama (local)' in .env.")


MODEL_PROVIDER, MODEL_NAME, MODEL_API_KEY, MODEL_BASE_URL = configured_model()

st.set_page_config(
    page_title="Interview Studio · Company-smart practice",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink:#24183f; --muted:#77708f; --purple:#7655f6; --pink:#ee71c4;
        --cyan:#78dcf4; --line:#e9e3f5; --surface:#fff; --deep:#211637;
    }
    html, body, [class*="css"] { font-family:'Segoe UI',sans-serif; color:var(--ink); }
    .stApp {
        background:
            radial-gradient(ellipse at 86% 4%, rgba(233,190,255,.32), transparent 30%),
            radial-gradient(ellipse at 5% 42%, rgba(185,222,255,.26), transparent 28%),
            #f8f6ff;
    }
    [data-testid="stAppViewContainer"] { background:transparent; }
    header[data-testid="stHeader"] { background:rgba(248,246,255,.7); }
    .block-container { max-width:1480px; padding:1.4rem 2.2rem 4rem; }
    h1,h2,h3 { font-family:'Segoe UI',sans-serif; letter-spacing:-.045em; color:var(--ink); }
    [data-testid="stSidebar"] {
        background:linear-gradient(165deg,#211637 0%,#302052 55%,#211637 100%);
        border-right:1px solid rgba(255,255,255,.11);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 { color:#fff; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color:#d5cdec !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color:#fff; background:#2b2045; border-color:#51416e; border-radius:12px;
    }
    [data-testid="stSidebar"] input { color:#fff !important; }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.13);
        border-radius:14px; padding:10px 14px;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#fff !important; }
    .brand { display:flex; align-items:center; gap:12px; margin:0 0 1.7rem; }
    .brand-mark {
        width:44px; height:44px; border-radius:15px;
        background:linear-gradient(145deg,#ff9bdc,#a78bff 60%,#79def2);
        color:#291841; display:grid; place-items:center; font-size:25px;
        box-shadow:0 8px 24px rgba(205,128,255,.32);
    }
    .brand-name { color:#fff; font:800 18px 'Segoe UI',sans-serif; letter-spacing:-.045em; }
    .brand-caption { color:#c8bce3; font-size:11px; margin-top:2px; }
    .hero {
        position:relative; isolation:isolate; overflow:hidden; min-height:270px;
        display:flex; flex-direction:column; justify-content:space-between; gap:25px;
        padding:25px 31px 21px; margin:0 0 23px;
        border:1px solid rgba(255,255,255,.24); border-radius:27px;
        color:#fff;
        background:
            radial-gradient(circle at 82% 28%,rgba(255,184,233,.38),transparent 25%),
            linear-gradient(112deg,#23163b 0%,#4a2c91 48%,#a14db7 100%);
        box-shadow:0 20px 55px rgba(74,43,122,.2);
    }
    .hero::before {
        content:""; position:absolute; z-index:-1; width:350px; height:350px;
        right:4%; top:-168px; border-radius:50%;
        border:1px solid rgba(255,255,255,.17);
        box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025);
    }
    .hero::after {
        content:"✦"; position:absolute; right:8%; top:34px;
        color:#ffe7a7; font-size:24px; text-shadow:0 0 22px #ffc5ef;
    }
    .hero-top,.hero-bottom { position:relative; z-index:1; display:flex; align-items:center; gap:12px; }
    .hero-top { justify-content:space-between; }
    .hero-badge,.live-pill {
        display:inline-flex; align-items:center; gap:8px; border-radius:999px;
        font-size:10px; font-weight:800; letter-spacing:.1em;
    }
    .hero-badge {
        padding:8px 12px; color:#fceaff;
        background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.15);
    }
    .hero-badge span { color:#ffd888; font-size:15px; }
    .live-pill {
        padding:8px 12px; color:#f4edff; background:rgba(24,16,43,.32);
        border:1px solid rgba(255,255,255,.19); white-space:nowrap;
    }
    .live-dot {
        width:7px; height:7px; border-radius:50%; background:#8ff0d1;
        box-shadow:0 0 0 4px rgba(143,240,209,.13),0 0 14px #8ff0d1;
    }
    .hero-copy { position:relative; z-index:1; max-width:770px; }
    .hero-kicker { color:#edcef9; font-size:10px; font-weight:800; letter-spacing:.16em; }
    .hero h1 {
        color:#fff; margin:7px 0 8px; font-size:clamp(2rem,4vw,3.35rem);
        line-height:1.03; font-weight:800; letter-spacing:-.065em;
    }
    .hero h1 span {
        color:#ffcaed; background:linear-gradient(90deg,#ffd0ef,#cfbcff 58%,#a6efff);
        -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
    }
    .hero-sub { max-width:620px; color:#e7dcf8; font-size:14px; line-height:1.55; }
    .hero-bottom {
        color:#ddd1f5; font-size:10px; font-weight:700; letter-spacing:.08em;
        border-top:1px solid rgba(255,255,255,.16); padding-top:13px;
    }
    .hero-bottom span { display:inline-flex; align-items:center; gap:7px; }
    .hero-art {
        position:absolute; z-index:0; right:12%; top:48px; font-size:clamp(5rem,10vw,8.5rem);
        line-height:1; opacity:.92; filter:drop-shadow(0 16px 26px rgba(16,7,36,.3));
        transform:rotate(7deg);
    }
    .metric {
        position:relative; min-height:112px; padding:16px 17px;
        background:rgba(255,255,255,.9); border:1px solid #e9e2f5;
        border-radius:18px; box-shadow:0 7px 24px rgba(70,49,111,.055);
        transition:transform .2s ease,box-shadow .2s ease;
    }
    .metric:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(90,63,143,.12); }
    .metric-top { display:flex; justify-content:space-between; align-items:center; }
    .metric-label { color:#817997; font-size:11px; font-weight:700; letter-spacing:.03em; }
    .metric-icon {
        width:29px; height:29px; display:grid; place-items:center; border-radius:10px;
        color:#6248bf; background:#f0eaff; font-size:15px;
    }
    .metric-value {
        margin-top:9px; color:#2b1d49; font:800 clamp(16px,1.5vw,22px) 'Segoe UI',sans-serif;
        letter-spacing:-.045em; overflow-wrap:anywhere;
    }
    .metric-hint { color:#918aa3; font-size:10px; margin-top:3px; }
    .panel-title { color:#281c43; font:800 17px 'Segoe UI',sans-serif; letter-spacing:-.035em; margin-bottom:4px; }
    .panel-hint { color:#807994; font-size:12px; margin-bottom:14px; }
    .evidence {
        border-left:3px solid #b68bff; background:linear-gradient(120deg,#fbf8ff,#f1f8ff);
        padding:12px 14px; border-radius:0 12px 12px 0; margin:8px 0;
        color:#35244f; font-size:13px;
    }
    .evidence-meta { color:#7457ac; font-size:11px; margin-bottom:5px; font-weight:700; }
    div.stButton > button {
        min-height:42px; border-radius:12px; font-weight:700; border:1px solid #e2d9f2;
        color:#514372; background:#fff; transition:transform .18s ease,box-shadow .18s ease;
    }
    div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 9px 22px rgba(102,74,164,.14); border-color:#b9a2f6; }
    div.stButton > button[kind="primary"] {
        color:#fff; border:0; background:linear-gradient(100deg,#7655f6,#bb5cda 65%,#ed75bb);
        box-shadow:0 10px 24px rgba(132,75,210,.25);
    }
    div.stButton > button[kind="primary"]:hover {
        color:#fff; background:linear-gradient(100deg,#6944eb,#ae4dce 65%,#e963ae);
        box-shadow:0 14px 28px rgba(132,75,210,.32);
    }
    [data-testid="stChatMessage"] {
        border:1px solid #e9e2f5; background:rgba(255,255,255,.92);
        border-radius:17px; box-shadow:0 6px 20px rgba(64,47,105,.045);
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color:#24183f !important; opacity:1 !important;
    }
    .current-question {
        position:relative; overflow:hidden; margin:13px 0 18px; padding:21px 24px 23px;
        border:1px solid #d9caff; border-left:5px solid #805cf5; border-radius:18px;
        background:
            radial-gradient(circle at 100% 0%,rgba(191,232,255,.42),transparent 34%),
            linear-gradient(125deg,#fff 0%,#f8f2ff 70%,#f2fbff 100%);
        box-shadow:0 12px 30px rgba(96,65,161,.12);
    }
    .question-label {
        display:flex; align-items:center; gap:9px; margin-bottom:11px;
        color:#7252c8; font-size:10px; font-weight:800; letter-spacing:.13em;
    }
    .question-label span {
        padding:4px 8px; border-radius:999px; color:#4c377f;
        background:#ece4ff; letter-spacing:.06em;
    }
    .question-text {
        color:#24183f; font-size:clamp(16px,1.25vw,20px); font-weight:700;
        line-height:1.55; overflow-wrap:anywhere; white-space:pre-wrap;
    }
    [data-testid="stChatInput"] {
        border:1px solid #ded5f0; border-radius:15px; background:#fff;
        box-shadow:0 8px 25px rgba(74,54,119,.08);
    }
    [data-testid="stChatInput"]:focus-within { border-color:#9c80f2; box-shadow:0 0 0 3px rgba(152,121,242,.12); }
    [data-testid="stProgressBar"] > div > div > div > div {
        background:linear-gradient(90deg,#7655f6,#d064cf,#73d7ee);
    }
    [data-testid="stAlert"] { border-radius:14px; }
    [data-testid="stExpander"] {
        border:1px solid #e8e1f4; border-radius:15px; background:rgba(255,255,255,.75);
    }
    hr { border-color:#e8e1f1; }
    @media(max-width:900px) { .hero-art { right:5%; opacity:.42; } }
    @media(max-width:650px) {
        .block-container { padding:1rem 1rem 3rem; }
        .hero { min-height:280px; padding:20px; border-radius:21px; }
        .hero-top { align-items:flex-start; flex-direction:column; }
        .hero-art { right:-4%; top:80px; opacity:.25; }
        .hero-bottom { flex-wrap:wrap; gap:8px 14px; }
    }
    @media(prefers-reduced-motion:reduce) { *,*::before,*::after { transition:none !important; animation:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_archive(sample_csv: bytes) -> InterviewArchive:
    archive = InterviewArchive(os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "data" / "chroma")))
    archive.ingest_csv(
        sample_csv,
        source_label="Illustrative practice dataset",
        dataset="illustrative sample — not institutional interview history",
    )
    return archive


def init_state() -> None:
    defaults = {
        "active": False,
        "company": "Acme Analytics",
        "role": "Data Analyst",
        "focus": "A balanced mix of behavioral and role-specific topics",
        "question": "",
        "last_answer": "",
        "last_evaluation": None,
        "evaluations": [],
        "messages": [],
        "asked_questions": [],
        "last_evidence": [],
        "last_report": "",
        "round_index": 0,
        "question_index": 0,
        "round_answers": [],
        "round_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.asked_questions:
        st.session_state.asked_questions = list(
            dict.fromkeys(
                message["content"].strip()
                for message in st.session_state.messages
                if message["role"] == "assistant"
                and message["content"].strip().endswith("?")
            )
        )


def show_error(error: Exception) -> None:
    st.error(f"{type(error).__name__}: {error}")
    st.caption("Your session is unchanged. Fix the issue described above and try again.")


def score_text(evaluation: dict) -> str:
    return (
        f"**Correctness:** {evaluation['correctness']}/5 · "
        f"**Depth:** {evaluation['depth']}/5 · "
        f"**Communication:** {evaluation['communication']}/5\n\n"
        f"{evaluation['summary']}\n\n"
        f"**What worked:** {'; '.join(evaluation['strengths'])}\n\n"
        f"**Improve next:** {'; '.join(evaluation['improvements'])}\n\n"
        f"**Next step:** {evaluation['next_step']}"
    )


def retrieve(
    archive: InterviewArchive,
    intent: Intent,
    *,
    query: str,
    kind: str | None,
    exclude_content: list[str] | None = None,
    round_name: str | None = None,
) -> list[dict]:
    if not should_retrieve(
        intent,
        company=st.session_state.company,
        role=st.session_state.role,
        has_evaluations=bool(st.session_state.evaluations),
    ):
        return []
    return archive.retrieve(
        company=st.session_state.company,
        role=st.session_state.role,
        query=query,
        kind=kind,
        exclude_content=exclude_content,
        round_name=round_name,
    )


def ask_round_question(
    archive: InterviewArchive,
    agents: InterviewAgents,
    previous_questions: list[str],
    *,
    round_index: int | None = None,
) -> tuple[str, list[dict]]:
    round_index = st.session_state.round_index if round_index is None else round_index
    interview_round = INTERVIEW_ROUNDS[round_index]
    evidence = retrieve(
        archive,
        Intent.INTERVIEW,
        query=(
            f"{st.session_state.role} {interview_round.name} "
            f"{interview_round.focus} {st.session_state.focus}"
        ),
        kind="question",
        exclude_content=previous_questions,
        round_name=interview_round.name,
    )
    question = agents.ask_question(
        company=st.session_state.company,
        role=st.session_state.role,
        focus=f"{interview_round.name}: {interview_round.focus}. {st.session_state.focus}",
        records=evidence,
        previous_questions=previous_questions,
    )
    return question, evidence


def queue_round_question(
    archive: InterviewArchive,
    agents: InterviewAgents,
    *,
    round_index: int,
    question_index: int,
) -> None:
    question, evidence = ask_round_question(
        archive,
        agents,
        st.session_state.asked_questions,
        round_index=round_index,
    )
    st.session_state.round_index = round_index
    st.session_state.question_index = question_index
    st.session_state.round_answers = []
    st.session_state.round_complete = False
    st.session_state.question = question
    st.session_state.asked_questions = list(
        dict.fromkeys(st.session_state.asked_questions + [question])
    )
    st.session_state.last_evidence = evidence
    st.session_state.messages.append({"role": "assistant", "content": question})


def render_evidence(records: list[dict]) -> None:
    if not records:
        st.info("No matching archive evidence was retrieved. The response is not presented as company-specific.")
        return
    for item in records:
        metadata = item["metadata"]
        match = "exact role match" if item["exact_role"] else "company-only match"
        st.markdown(
            f"<div class='evidence'><div class='evidence-meta'>"
            f"{html.escape(metadata['company'])} · {html.escape(metadata['role'])} · "
            f"{html.escape(metadata['kind'])} · {match}"
            f"</div>{html.escape(item['content'])}<br><small>Source: "
            f"{html.escape(metadata['source'])} · {html.escape(metadata['dataset'])}</small></div>",
            unsafe_allow_html=True,
        )


init_state()
try:
    archive = get_archive((ROOT / "data" / "sample_interview_archive.csv").read_bytes())
except (ValueError, OSError, RuntimeError) as error:
    st.error(f"Could not open the interview archive: {error}")
    st.stop()

with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='brand-mark'>🦄</div><div>"
        "<div class='brand-name'>Interview Studio</div>"
        "<div class='brand-caption'>Your career glow-up starts here</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Your session")
    company_input = st.text_input(
        "Target company",
        value=st.session_state.company,
        help="Try: " + ", ".join(SUGGESTED_COMPANIES),
        disabled=st.session_state.active,
    )
    role_input = st.text_input(
        "Target role",
        value=st.session_state.role,
        help="Round packs: " + ", ".join(SUGGESTED_ROLES) + ". Other roles are supported too.",
        disabled=st.session_state.active,
    )
    st.caption("Popular companies: Google · Microsoft · Amazon · NVIDIA · Siemens")
    st.caption("Role packs: Data Analyst · AI/ML · DevOps · Mechanical · Software · Product")
    st.selectbox(
        "Interview focus",
        [
            "A balanced mix of behavioral and role-specific topics",
            "Technical depth",
            "Behavioral and leadership",
            "Problem solving",
            "Product and strategy",
        ],
        key="focus",
    )
    st.divider()
    st.markdown("#### Interview archive")
    st.metric("Indexed records", archive.count)
    st.caption(
        "The included records are clearly labeled illustrative examples, not real institutional history."
    )
    upload = st.file_uploader("Add an archive CSV", type=["csv"])
    if upload is not None:
        if st.button("Index uploaded archive", use_container_width=True):
            try:
                added = archive.ingest_csv(
                    upload.getvalue(),
                    source_label=upload.name,
                    dataset=f"uploaded archive · {upload.name}",
                )
                st.success(f"Indexed {added} new record(s). Existing records were left unchanged.")
                st.rerun()
            except (UnicodeDecodeError, ValueError) as error:
                show_error(error)
    st.caption("CSV columns: company, role, type, content; source and round are optional.")
    if st.button("Reset interview session", use_container_width=True):
        st.session_state.active = False
        st.session_state.question = ""
        st.session_state.last_answer = ""
        st.session_state.last_evaluation = None
        st.session_state.evaluations = []
        st.session_state.messages = []
        st.session_state.asked_questions = []
        st.session_state.round_index = 0
        st.session_state.question_index = 0
        st.session_state.round_answers = []
        st.session_state.round_complete = False
        st.session_state.last_evidence = []
        st.session_state.last_report = ""
        st.rerun()


st.markdown(
    "<div class='hero'>"
    "<div class='hero-top'>"
    "<div class='hero-badge'><span>✦</span> THE INTERVIEW GLOW-UP</div>"
    "<div class='live-pill'><span class='live-dot'></span> GROUNDED PRACTICE · MADE FOR YOUR NEXT WIN</div>"
    "</div>"
    "<div class='hero-copy'>"
    "<div class='hero-kicker'>YOUR NEXT CHAPTER STARTS HERE</div>"
    "<h1>Big interview energy.<br><span>Meet your moment.</span></h1>"
    "<div class='hero-sub'>Real company questions. Personal coaching. A little more confidence "
    "every round.</div>"
    "</div>"
    "<div class='hero-bottom'><span>✦ REAL QUESTIONS</span><span>✦ SMART FEEDBACK</span>"
    "<span>✦ MAIN-CHARACTER CONFIDENCE</span></div>"
    "<div class='hero-art' aria-hidden='true'>🦄</div>"
    "</div>",
    unsafe_allow_html=True,
)

evaluation_count = len(st.session_state.evaluations)
average_score = 0.0
if evaluation_count:
    score_values = [
        (item["correctness"] + item["depth"] + item["communication"]) / 3
        for item in st.session_state.evaluations
    ]
    average_score = sum(score_values) / len(score_values)

metric_columns = st.columns(4)
for index, (column, label, value, hint) in enumerate(
    zip(
        metric_columns,
        ("Target", "Role", "Rounds evaluated", "Average score"),
        (
            st.session_state.company,
            st.session_state.role,
            str(evaluation_count),
            f"{average_score:.1f} / 5" if evaluation_count else "—",
        ),
        (
            "Company-specific practice",
            "Tailored question retrieval",
            "One evaluation per completed round",
            "Across correctness, depth & clarity",
        ),
    )
):
    metric_icon = ("✦", "◈", "↗", "★")[index]
    column.markdown(
        f"<div class='metric'><div class='metric-top'><div class='metric-label'>{label}</div>"
        f"<div class='metric-icon'>{metric_icon}</div></div>"
        f"<div class='metric-value'>{html.escape(value)}</div>"
        f"<div class='metric-hint'>{html.escape(hint)}</div></div>",
        unsafe_allow_html=True,
    )

st.write("")
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.markdown("<div class='panel-title'>✦ &nbsp; The interview room</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-hint'>Answer naturally. Each round is scored against both responses "
        "and relevant archive evidence. Complete two questions for one round evaluation.</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.active:
        current_round = INTERVIEW_ROUNDS[st.session_state.round_index]
        progress = (
            (st.session_state.round_index + 1) / len(INTERVIEW_ROUNDS)
            if st.session_state.round_complete
            else (
                st.session_state.round_index
                + st.session_state.question_index / QUESTIONS_PER_ROUND
            )
            / len(INTERVIEW_ROUNDS)
        )
        st.progress(progress)
        if st.session_state.round_complete:
            st.caption(
                f"ROUND {st.session_state.round_index + 1}/{len(INTERVIEW_ROUNDS)} "
                f"COMPLETE · {current_round.name}"
            )
        else:
            st.caption(
                f"ROUND {st.session_state.round_index + 1}/{len(INTERVIEW_ROUNDS)} · "
                f"{current_round.name} · QUESTION "
                f"{st.session_state.question_index + 1}/{QUESTIONS_PER_ROUND}"
            )
    if not st.session_state.active:
        st.info("Choose a company and role in the sidebar, then start your mock interview.")
        if st.button(
            f"Start {len(INTERVIEW_ROUNDS)}-round interview · "
            f"{len(INTERVIEW_ROUNDS) * QUESTIONS_PER_ROUND} questions",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.company = company_input.strip()
            st.session_state.role = role_input.strip()
            if not st.session_state.company or not st.session_state.role:
                st.error("Enter both a target company and role before starting.")
            else:
                try:
                    role_pack = build_role_round_csv(
                        st.session_state.company, st.session_state.role
                    )
                    if role_pack:
                        archive.ingest_csv(
                            role_pack,
                            source_label="Illustrative role-round practice pack",
                            dataset="illustrative practice questions — not institutional history",
                        )
                    agents = InterviewAgents(
                        build_chat_model(
                            provider=MODEL_PROVIDER,
                            model_name=MODEL_NAME,
                            api_key=MODEL_API_KEY,
                            base_url=MODEL_BASE_URL,
                        )
                    )
                    st.session_state.asked_questions = []
                    st.session_state.round_answers = []
                    st.session_state.round_index = 0
                    st.session_state.question_index = 0
                    st.session_state.round_complete = False
                    st.session_state.evaluations = []
                    st.session_state.last_evaluation = None
                    st.session_state.last_answer = ""
                    st.session_state.last_report = ""
                    st.session_state.messages = []
                    queue_round_question(
                        archive,
                        agents,
                        round_index=0,
                        question_index=0,
                    )
                    st.session_state.active = True
                    st.rerun()
                except (httpx.HTTPError, OutputParserException, RuntimeError, ValueError) as error:
                    show_error(error)
    else:
        for index, message in enumerate(st.session_state.messages):
            is_current_question = (
                index == len(st.session_state.messages) - 1
                and message["role"] == "assistant"
                and bool(st.session_state.question)
                and message["content"] == st.session_state.question
            )
            if is_current_question:
                st.markdown(
                    "<div class='current-question'>"
                    "<div class='question-label'>✦ CURRENT INTERVIEW QUESTION "
                    "<span>YOUR TURN</span></div>"
                    f"<div class='question-text'>{html.escape(message['content'])}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        if st.session_state.question:
            st.caption("Submit your answer to unlock the next question in this round.")
        chat_input = st.chat_input(
            "Type your answer to the current round question…",
            disabled=st.session_state.round_complete,
        )
        button_columns = st.columns(3)
        next_round_index = st.session_state.round_index + 1
        if next_round_index < len(INTERVIEW_ROUNDS):
            next_round_label = (
                f"Start Round {next_round_index + 1}"
                if st.session_state.round_complete
                else "Finish this round to continue"
            )
        else:
            next_round_label = (
                "Interview complete"
                if st.session_state.round_complete
                else "Finish this round to continue"
            )
        next_round_button = button_columns[0].button(
            next_round_label,
            type="primary" if st.session_state.round_complete and next_round_index < len(INTERVIEW_ROUNDS) else "secondary",
            use_container_width=True,
            disabled=not st.session_state.round_complete or next_round_index >= len(INTERVIEW_ROUNDS),
        )
        coach_button = button_columns[1].button(
            "Coach my last round",
            use_container_width=True,
            disabled=not bool(st.session_state.last_answer) or not st.session_state.round_complete,
        )
        report_button = button_columns[2].button(
            "Session report", use_container_width=True, disabled=not bool(st.session_state.evaluations)
        )

        request: tuple[Intent, str] | None = None
        if chat_input:
            request = (
                detect_intent(chat_input, has_active_question=bool(st.session_state.question)),
                chat_input,
            )
        elif next_round_button:
            request = (Intent.INTERVIEW, "__start_next_round__")
        elif coach_button:
            request = (Intent.COACH, "coach my last answer")
        elif report_button:
            request = (Intent.REPORT, "generate a session report")

        if request:
            intent, user_text = request
            st.info(f"Intent router → **{intent.value}**")
            try:
                agents = InterviewAgents(
                    build_chat_model(
                        provider=MODEL_PROVIDER,
                        model_name=MODEL_NAME,
                        api_key=MODEL_API_KEY,
                        base_url=MODEL_BASE_URL,
                    )
                )
                if intent is Intent.INTERVIEW:
                    if user_text != "__start_next_round__":
                        raise ValueError(
                            "Answer the current question to complete this round before moving on."
                        )
                    next_round_index = st.session_state.round_index + 1
                    if (
                        not st.session_state.round_complete
                        or next_round_index >= len(INTERVIEW_ROUNDS)
                    ):
                        raise ValueError("Complete the current round before starting another.")
                    queue_round_question(
                        archive,
                        agents,
                        round_index=next_round_index,
                        question_index=0,
                    )
                elif intent is Intent.EVALUATE:
                    question = st.session_state.question
                    answer = user_text.strip()
                    command_answer = re.match(
                        r"^(?:evaluate|score|assess|rate|review my answer)\s*(?:this|my answer)?\s*:\s*(.+)$",
                        answer,
                        re.I | re.S,
                    )
                    if command_answer:
                        answer = command_answer.group(1).strip()
                    if not question:
                        raise ValueError("Ask the interviewer for a question before submitting an answer.")
                    if not answer or answer.lower() in {"evaluate", "score", "assess", "rate"}:
                        raise ValueError("Include the answer you would like evaluated.")
                    completed_answers = st.session_state.round_answers + [
                        {"question": question, "answer": answer}
                    ]
                    if st.session_state.question_index + 1 < QUESTIONS_PER_ROUND:
                        next_question_index = st.session_state.question_index + 1
                        next_question, evidence = ask_round_question(
                            archive,
                            agents,
                            st.session_state.asked_questions,
                        )
                        st.session_state.round_answers = completed_answers
                        st.session_state.question_index = next_question_index
                        st.session_state.question = next_question
                        st.session_state.asked_questions = list(
                            dict.fromkeys(
                                st.session_state.asked_questions + [next_question]
                            )
                        )
                        st.session_state.last_answer = answer
                        st.session_state.last_evidence = evidence
                        st.session_state.messages.extend(
                            [
                                {"role": "user", "content": answer},
                                {"role": "assistant", "content": next_question},
                            ]
                        )
                    else:
                        round_questions = "\n\n".join(
                            f"Question {index + 1}: {item['question']}"
                            for index, item in enumerate(completed_answers)
                        )
                        round_answers_text = "\n\n".join(
                            f"Answer {index + 1}: {item['answer']}"
                            for index, item in enumerate(completed_answers)
                        )
                        evidence_by_content: dict[str, dict] = {}
                        for item in completed_answers:
                            for record in retrieve(
                                archive,
                                intent,
                                query=f"{item['question']}\nCandidate answer: {item['answer']}",
                                kind=None,
                            ):
                                evidence_by_content.setdefault(record["content"], record)
                        evidence = list(evidence_by_content.values())
                        result: AnswerEvaluation = agents.evaluate(
                            company=st.session_state.company,
                            role=st.session_state.role,
                            question=(
                                f"Round: {INTERVIEW_ROUNDS[st.session_state.round_index].name}\n"
                                f"{round_questions}"
                            ),
                            answer=round_answers_text,
                            records=evidence,
                        )
                        evaluation = result.model_dump()
                        st.session_state.last_answer = round_answers_text
                        st.session_state.last_evaluation = evaluation
                        st.session_state.evaluations.append(
                            {
                                "round": INTERVIEW_ROUNDS[
                                    st.session_state.round_index
                                ].name,
                                "question_count": len(completed_answers),
                                "question": round_questions,
                                "answer": round_answers_text,
                                **evaluation,
                            }
                        )
                        st.session_state.round_answers = completed_answers
                        st.session_state.round_complete = True
                        st.session_state.question = ""
                        st.session_state.last_evidence = evidence
                        round_label = (
                            f"Round {st.session_state.round_index + 1} · "
                            f"{INTERVIEW_ROUNDS[st.session_state.round_index].name}"
                        )
                        st.session_state.messages.extend(
                            [
                                {"role": "user", "content": answer},
                                {
                                    "role": "assistant",
                                    "content": f"**{round_label} evaluation**\n\n"
                                    f"{score_text(evaluation)}",
                                },
                            ]
                        )
                elif intent is Intent.COACH:
                    if not st.session_state.last_answer:
                        raise ValueError("Answer an interview question first so your coaching can be personalised.")
                    evaluation_text = (
                        json.dumps(st.session_state.last_evaluation, ensure_ascii=False)
                        if st.session_state.last_evaluation
                        else "No structured evaluation is available yet."
                    )
                    evidence = retrieve(
                        archive,
                        intent,
                        query=st.session_state.last_answer,
                        kind="feedback",
                    )
                    response = agents.coach_answer(
                        company=st.session_state.company,
                        role=st.session_state.role,
                        answer=st.session_state.last_answer,
                        evaluation=evaluation_text,
                        records=evidence,
                    )
                    st.session_state.last_evidence = evidence
                    st.session_state.messages.extend(
                        [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": response},
                        ]
                    )
                else:
                    if not st.session_state.evaluations:
                        raise ValueError("Complete at least one evaluated answer before requesting a report.")
                    evaluation_text = json.dumps(
                        st.session_state.evaluations, ensure_ascii=False, indent=2
                    )
                    evidence = retrieve(
                        archive,
                        intent,
                        query=f"{st.session_state.role} interview feedback improvement",
                        kind="feedback",
                    )
                    response = agents.write_report(
                        company=st.session_state.company,
                        role=st.session_state.role,
                        evaluations=evaluation_text,
                        records=evidence,
                    )
                    st.session_state.last_report = response
                    st.session_state.last_evidence = evidence
                    st.session_state.messages.extend(
                        [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": response},
                        ]
                    )
                st.rerun()
            except (
                httpx.HTTPError,
                OutputParserException,
                RuntimeError,
                ValueError,
            ) as error:
                show_error(error)

with right:
    st.markdown("<div class='panel-title'>✧ &nbsp; Your feedback</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-hint'>A clear view of what is working and what to practice next.</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.last_evaluation:
        latest = st.session_state.last_evaluation
        for label, key in (
            ("Correctness", "correctness"),
            ("Depth", "depth"),
            ("Communication", "communication"),
        ):
            st.markdown(f"**{label}** · {latest[key]} / 5")
            st.progress(latest[key] / 5)
        st.markdown("**Strengths**")
        for strength in latest["strengths"]:
            st.markdown(f"- {strength}")
        st.markdown("**Practice next**")
        for gap in latest["improvements"]:
            st.markdown(f"- {gap}")
        st.caption(latest["next_step"])
    else:
        st.markdown(
            "<div class='panel' style='background:#f7faf7;color:#718079;font-size:13px'>"
            "Your evaluation appears here after your first answer.</div>",
            unsafe_allow_html=True,
        )
    if st.session_state.last_evidence:
        match_type = (
            "exact company + role"
            if any(item["exact_role"] for item in st.session_state.last_evidence)
            else "company-only"
        )
        st.caption(f"Archive match used: {match_type}")
        with st.expander("Archive evidence used this turn", expanded=False):
            render_evidence(st.session_state.last_evidence)
    if st.session_state.last_report:
        st.download_button(
            "Download session report",
            data=st.session_state.last_report,
            file_name="interview-session-report.txt",
            mime="text/plain",
        )

st.write("")
with st.expander("How it works · agents, intent routing & retrieval"):
    st.markdown(
        """
        **Intent routing:** explicit words such as “coach”, “score”, “report”, and “next question”
        route to the matching agent. While a question is open, an unlabelled message is treated
        as the candidate's answer. The router and retrieval policy are implemented in
        `interview_coach/routing.py`.

        **Agentic RAG:** the selected intent, company, role, and session state determine whether
        ChromaDB is queried and which kind of record is useful. Retrieval prefers exact company
        + role matches, then falls back to company-only evidence; an empty result is explicitly
        shown rather than presented as historical fact. ChromaDB's local embedding function
        indexes the included illustrative CSV and any uploaded archive.

        **LangChain / LCEL:** each agent connects a `ChatPromptTemplate`, selected chat model,
        and output parser with the `prompt | model | parser` LCEL pattern. The evaluator parses
        its response into a validated Pydantic score object.

        **Privacy:** prompts, candidate answers, and retrieved snippets are processed by the
        service configured in `.env`. Use the local option for sensitive archives, and remove
        identifying details before uploading historical data.
        """
    )
    st.markdown("**System prompts used**")
    for agent_name, system_prompt in SYSTEM_PROMPTS.items():
        with st.expander(agent_name.title()):
            st.code(system_prompt, language="text")
            st.markdown("**User template**")
            st.code(USER_PROMPTS[agent_name], language="text")
    st.markdown("**Prompting techniques:** zero-shot instructions, role/persona prompting, explicit "
                "grounding constraints, structured output validation, and conditional retrieval.")
