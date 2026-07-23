"""
FFCell 로켓 조립 품질 대시보드 (드릴다운형)
데이터: build_cycle.py 산출물 (cycles.csv / patterns.csv / feature_importance.csv / metrics.json / drilldown.csv)
실행: streamlit run app.py
"""
import os, json
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = "."
BG, ACCENT, DEFECT, INK = "#F5F2ED", "#4A7C59", "#E24B4A", "#2B2B2B"
CLASS_COLORS = {"Normal": ACCENT, "NoNose": "#E8A13C",
                "NoNose,NoBody2": "#C77DFF", "NoNose,NoBody2,NoBody1": DEFECT}
ROBOT_KOR = {"R01": "R01 (베이스/바디1)", "R02": "R02 (그리퍼-제외피처)",
             "R03": "R03 (조립력·핵심신호)", "R04": "R04 (노즈)"}

st.set_page_config(page_title="FFCell 품질 대시보드", layout="wide", page_icon="🚀")
st.markdown(f"""<style>
 .stApp {{background:{BG};}} h1,h2,h3,h4 {{color:{INK};}}
 .kpi{{background:#fff;border:1px solid #e6e0d6;border-radius:14px;padding:14px 16px;
   box-shadow:0 1px 3px rgba(0,0,0,.05);}}
 .kpi .lbl{{font-size:12px;color:#8a8577;margin-bottom:5px;}}
 .kpi .val{{font-size:27px;font-weight:700;color:{INK};line-height:1;}}
 .kpi .sub{{font-size:11px;color:#a7a294;margin-top:5px;}}
 [data-testid="stSidebar"]{{background:#efeae1;}}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load(name):
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p): return None
    return json.load(open(p, encoding="utf-8")) if name.endswith(".json") else pd.read_csv(p)

@st.cache_data
def drill_for_cycle(cid):
    d = load("drilldown.csv")
    if d is None: return None
    return d[d["id"] == cid].sort_values("t")

def kpi(col, lbl, val, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

cyc = load("cycles.csv"); met = load("metrics.json")
fi = load("feature_importance.csv"); patt = load("patterns.csv")
if cyc is None or met is None:
    st.error("cycles.csv / metrics.json 이 없어요. build_cycle.py 를 먼저 실행하세요."); st.stop()

st.sidebar.title("🚀 FFCell 품질 대시보드")
st.sidebar.caption("로켓 조립 라인 · 규칙기반 계층 분류")
page = st.sidebar.radio("페이지", ["① 종합 현황", "② 센서·로봇 움직임",
                                   "③ 사이클 드릴다운", "④ 이미지 CV (개발 중)"])
st.sidebar.caption(f"유효 사이클 {len(cyc)}개 (327→317, v73)")

tot = len(cyc); succ = int((~cyc["defect"]).sum()); fail = int(cyc["defect"].sum())

# ───────── ① 종합 현황 ─────────
if page.startswith("①"):
    st.title("종합 현황")
    st.caption("이 라인에서 지금까지 무엇이, 얼마나, 어떻게 조립됐나")

    st.markdown("#### 조립 카운트")
    c = st.columns(4)
    kpi(c[0], "총 조립 시도", f"{tot}")
    kpi(c[1], "조립 성공 (정상)", f"{succ}", f"{succ/tot*100:.1f}%")
    kpi(c[2], "조립 실패 (결함)", f"{fail}", f"{fail/tot*100:.1f}%")
    kpi(c[3], "실패율", f"{fail/tot*100:.1f}%")

    st.markdown("#### 분류 성능 (규칙기반 계층 분류 · 5-fold 교차검증)")
    c = st.columns(3)
    kpi(c[0], "정확도", f"{met['accuracy']*100:.1f}%")
    kpi(c[1], "탐지 균형 (macro recall)", f"{met['macro_recall']*100:.1f}%", "클래스 불균형 보정")
    kpi(c[2], "결함 탐지율", f"{met['defect_detection']*100:.1f}%", "실제 결함 중 탐지")

    a, b = st.columns([1, 1])
    with a:
        st.markdown("##### 정상 vs 결함")
        d = pd.DataFrame({"구분": ["정상", "결함"], "수": [succ, fail]})
        f = px.pie(d, names="구분", values="수", hole=.55,
                   color="구분", color_discrete_map={"정상": ACCENT, "결함": DEFECT})
        f.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(f, use_container_width=True)
    with b:
        st.markdown("##### 클래스 분포")
        dd = cyc["label"].value_counts().reset_index(); dd.columns = ["label", "n"]
        f = px.bar(dd, x="label", y="n", color="label", text="n",
                   color_discrete_map=CLASS_COLORS)
        f.update_layout(showlegend=False, height=300, plot_bgcolor="white",
                        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(f, use_container_width=True)

    st.markdown("#### 피처 중요도 · 클래스별 recall
