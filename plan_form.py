"""Form page: collect the trip details, turn them into a message, and hand it to the chat page."""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

import render

INTERESTS = ["Beaches", "Food", "History & culture", "Nightlife", "Nature", "Adventure", "Shopping", "Relaxing"]

ss = st.session_state


def trip_message() -> str:
    """The form answers as the same kind of sentence a user would type."""
    days, travellers = int(ss.form_days), int(ss.form_travelers)
    text = (
        f"{days} day{'s' if days != 1 else ''} in {ss.form_destination.strip()}, ₹{int(ss.form_budget):,}, "
        f"starting {ss.form_start:%A %d %B %Y}, for {travellers} traveller{'s' if travellers != 1 else ''}"
    )
    if ss.form_interests:
        text += f". Interests: {', '.join(ss.form_interests).lower()}"
    if ss.form_notes.strip():
        text += f". {ss.form_notes.strip()}"
    return text


st.markdown(
    '<section class="st-hero"><p class="label">Concierge</p>'
    '<h1 class="display">Plan with a<br><em>form.</em></h1>'
    '<p class="st-hero__lede">Fill in the details and we will plan the trip. '
    'Prefer to type it out? Use the chat instead.</p></section>',
    unsafe_allow_html=True,
)

with st.container(key="planner"):
    with st.form("trip_form", border=False):
        st.text_input("Where to?", key="form_destination", placeholder="Goa, Jaipur, Paris…")
        c1, c2 = st.columns(2)
        c1.number_input("Days", min_value=1, max_value=30, value=4, key="form_days")
        c2.number_input("Travellers", min_value=1, max_value=20, value=2, key="form_travelers")
        c3, c4 = st.columns(2)
        c3.number_input("Budget (₹, total)", min_value=500, step=500, value=20000, key="form_budget")
        c4.date_input("Starting on", value=date.today() + timedelta(days=7), min_value=date.today(), key="form_start")
        st.multiselect("What do you enjoy?", INTERESTS, key="form_interests")
        st.text_area("Anything else? (optional)", key="form_notes", height=80, placeholder="Vegetarian food, travelling with kids…")
        submitted = st.form_submit_button("Plan my trip")

    if submitted:
        if ss.form_destination.strip():
            ss.pending = trip_message()
            st.switch_page("chat.py")
        else:
            st.markdown(render.notice_html("Tell us where you would like to go."), unsafe_allow_html=True)

    st.page_link("chat.py", label="Back to chat", icon=":material/arrow_back:")
