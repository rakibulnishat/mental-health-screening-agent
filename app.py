import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agent.langgraph_agent import build_agent, fresh_agent_state
from database.db import save_session, get_user_history, build_trend_message, detect_worsening
from utils.report import generate_pdf_report

st.set_page_config(page_title='Mental Health Screening Assistant', page_icon='🧠', layout='centered')
st.title('🧠 Mental Health Screening Assistant')
st.caption('A screening and support tool — not a diagnostic system. All outputs are for informational purposes only.')
st.divider()

with st.sidebar:
    st.header('Settings')
    user_id = st.text_input('Your User ID', value='user_001')
    st.divider()
    st.subheader('Session History')
    records = get_user_history(user_id)
    if records:
        st.write(f'{len(records)} past session(s) found.')
        trend = build_trend_message(records)
        if trend:
            if detect_worsening(records):
                st.warning(trend)
            else:
                st.success(trend)
        if st.button('📄 Download PDF Report'):
            pdf_path = generate_pdf_report(user_id, records, output_dir='.')
            with open(pdf_path, 'rb') as f:
                st.download_button(label='Click to download', data=f, file_name=os.path.basename(pdf_path), mime='application/pdf')
    else:
        st.info('No past sessions found for this user ID.')
    st.divider()
    st.caption('🇧🇩 Crisis support: Kaan Pete Roi — 01779-554391')

if 'agent_state' not in st.session_state or st.session_state.get('current_user') != user_id:
    st.session_state.agent_state = fresh_agent_state(user_id)
    st.session_state.current_user = user_id
    st.session_state.agent = build_agent()
    st.session_state.session_saved = False
    st.session_state.agent_state = st.session_state.agent.invoke(st.session_state.agent_state)

for msg in st.session_state.agent_state['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

state = st.session_state.agent_state

if state['session_complete'] and not st.session_state.session_saved:
    phq9 = state['screening'].get('phq9_total', 0)
    gad7 = state['screening'].get('gad7_total', 0)
    d_risk = state['fused_risk'].get('depression_risk', 0.0)
    a_risk = state['fused_risk'].get('anxiety_risk', 0.0)
    esc = state['escalation_level'] or 'low'
    save_session(user_id, phq9, gad7, d_risk, a_risk, esc)
    st.session_state.session_saved = True
    st.success('✅ Session complete and saved.')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🔄 Start New Session'):
            st.session_state.agent_state = fresh_agent_state(user_id)
            st.session_state.session_saved = False
            st.session_state.agent_state = st.session_state.agent.invoke(st.session_state.agent_state)
            st.rerun()
    with col2:
        all_records = get_user_history(user_id)
        pdf_path = generate_pdf_report(user_id, all_records, output_dir='.')
        with open(pdf_path, 'rb') as f:
            st.download_button(label='📄 Download Report', data=f, file_name=os.path.basename(pdf_path), mime='application/pdf')

if not state['session_complete']:
    user_input = st.chat_input('Type your answer here (0, 1, 2, or 3)...')
    if user_input:
        st.session_state.agent_state['messages'].append({'role': 'user', 'content': user_input})
        st.session_state.agent_state = st.session_state.agent.invoke(st.session_state.agent_state)
        st.rerun()
