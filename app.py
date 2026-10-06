"""GlobeTrot concierge: Streamlit entry point and page router.  Run with:  streamlit run app.py

Two pages share this frame: chat.py (describe the trip in a message) and plan_form.py (fill in a form).
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import logs
import render

load_dotenv()
logs.configure()

ROOT = Path(__file__).parent
SITE_URL = os.getenv("SITE_URL", "http://localhost:5173")  # the GlobeTrot landing page

st.set_page_config(page_title="GlobeTrot — Concierge", page_icon="✈️", layout="centered", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(ROOT / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
st.markdown(render.nav_html(SITE_URL), unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("messages", [])   # what the chat page shows
ss.setdefault("session", None)  # a chat session from llm.new_session(), which holds the model history

pages = st.navigation(
    [
        st.Page("chat.py", title="Chat", url_path="chat", default=True),
        st.Page("plan_form.py", title="Plan with a form", url_path="form"),
    ],
    position="hidden",
)
pages.run()
