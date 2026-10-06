"""Chat page: describe the trip in a message and the concierge plans it."""
from __future__ import annotations

import streamlit as st

import content
import export
import llm
import render

AVATARS = {"user": ":material/person:", "assistant": ":material/flight_takeoff:"}
EXAMPLES = [
    "4 days in Goa, ₹20,000, starting this Friday",
    "3 days in Jaipur, ₹15,000, starting Saturday",
    "Paris for a week, ₹5,000",
]

ss = st.session_state


def reset_chat() -> None:
    ss.messages = []
    ss.session = None
    ss.pop("pending", None)


def queue_prompt(text: str) -> None:
    ss.pending = text


def downloads(itinerary, index: int) -> None:
    stem = export.file_stem(itinerary)
    with st.container(key=f"downloads_{index}", horizontal=True):
        st.download_button("Download .txt", export.itinerary_text(itinerary), f"{stem}.txt", "text/plain", key=f"dl_txt_{index}", on_click="ignore")
        st.download_button("Download .json", export.itinerary_json(itinerary), f"{stem}.json", "application/json", key=f"dl_json_{index}", on_click="ignore")


def show(message: dict, index: int) -> None:
    """Draw one stored message (the one at `index` in ss.messages) inside the current chat_message container."""
    if message["role"] == "user":
        st.markdown(render.user_html(message["text"]), unsafe_allow_html=True)
    elif message.get("error"):
        st.markdown(render.notice_html(message["text"]), unsafe_allow_html=True)
    elif message.get("itinerary"):
        if message.get("log"):
            with st.expander("How this plan was made"):
                for line in message["log"]:
                    st.markdown(f"- {line}")
        st.markdown(render.itinerary_html(message["itinerary"]), unsafe_allow_html=True)
        downloads(message["itinerary"], index)
    else:
        st.markdown(message["text"])


st.button("Reset chat", key="reset", on_click=reset_chat)

# Arriving from a journey page ("?journey=goa") starts the chat with that trip, once.
journey_slug = st.query_params.get("journey")
if journey_slug:
    st.query_params.pop("journey")
    opening = content.journey_prompt(journey_slug)
    if opening and not ss.messages:
        ss.pending = opening

typed = st.chat_input("Tell us about your trip…")
prompt = typed or ss.pop("pending", None)

if not ss.messages and not prompt:
    st.markdown(render.hero_html(), unsafe_allow_html=True)
    if not llm.has_api_key():
        st.markdown(render.notice_html(llm.missing_key_message()), unsafe_allow_html=True)
    with st.container(key="examples"):
        st.markdown('<p class="label">Try one</p>', unsafe_allow_html=True)
        for i, example in enumerate(EXAMPLES):
            st.button(example, key=f"example_{i}", on_click=queue_prompt, args=(example,), use_container_width=True)
        st.page_link("plan_form.py", label="Or plan with a form", icon=":material/edit_note:")

for i, message in enumerate(ss.messages):
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        show(message, i)

if prompt:
    ss.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        show(ss.messages[-1], len(ss.messages) - 1)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        holder = st.empty()
        log: list = []
        try:
            if ss.session is None:
                ss.session = llm.new_session()
            with holder.container():
                status = st.status("Thinking…", expanded=True)

            def on_event(kind: str, text: str) -> None:
                log.append(text)
                status.write(text)

            result = ss.session.send(prompt, on_event)
            if result.tool_calls:
                status.update(label="Planned with live data", state="complete", expanded=False)
            else:
                holder.empty()
            reply = {"role": "assistant", "text": result.text, "itinerary": result.itinerary, "log": log}
        except llm.ChatError as exc:
            holder.empty()
            reply = {"role": "assistant", "text": str(exc), "error": True}

        ss.messages.append(reply)
        show(reply, len(ss.messages) - 1)
