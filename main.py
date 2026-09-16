import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 페이지 기본 설정 (제목, 아이콘, 브라우저 탭 제목)
# ---------------------------------------------------------
st.set_page_config(page_title="선수 유형 나누기", page_icon="⚽", layout="wide")

st.title("⚽ 선수 유형 나누기")

# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/eafc25_top100.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------------
# 능력치 한글 이름 매핑
# ---------------------------------------------------------
stat_label_map = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움",
}
stat_cols = list(stat_label_map.keys())

# ---------------------------------------------------------
# 포지션 대분류 만들기 (positions 열의 맨 앞 포지션 기준)
# 묶는 데는 사용하지 않고, 나중에 교차표에서만 사용
# ---------------------------------------------------------
FORWARD_POS = {"ST", "CF", "LW", "RW"}
MID_POS = {"CAM", "CM", "CDM", "LM", "RM"}
DEFEND_POS = {"CB", "LB", "RB", "LWB", "RWB"}

def get_position_group(positions_str):
    first_pos = str(positions_str).split(",")[0].strip()
    if first_pos in FORWARD_POS:
        return "공격수"
    elif first_pos in MID_POS:
        return "미드필더"
    elif first_pos in DEFEND_POS:
        return "수비수"
    else:
        return "기타"

df["position_group"] = df["positions"].apply(get_position_group)

# ---------------------------------------------------------
# 능력치 선택 (2개 이상, 기본은 6개 전체)
# ---------------------------------------------------------
st.subheader("1. 묶는 데 사용할 능력치 고르기")

selected_stats = st.multiselect(
    "군집화에 사용할 능력치를 골라주세요 (최소 2개)",
    options=stat_cols,
    default=stat_cols,
    format_func=lambda x: stat_label_map[x],
)

if len(selected_stats) < 2:
    st.warning("능력치를 2개 이상 선택해주세요.")
    st.stop()

# ---------------------------------------------------------
# 표준화 (선택한 능력치 기준)
# ---------------------------------------------------------
X = df[selected_stats].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 묶음 수 선택 (2~6, 기본 3)
# ---------------------------------------------------------
st.subheader("2. 묶음 수 정하기")

n_clusters = st.slider("묶음 수를 선택하세요", min_value=2, max_value=6, value=3)

# ---------------------------------------------------------
# 엘보우 방법: 묶음 수 1~7에 대한 이너셔(관성) 계산
# 이너셔 = 각 점이 자기 묶음 중심에서 떨어진 거리의 제곱을 모두 더한 값
# ---------------------------------------------------------
st.subheader("3. 묶음 수에 따른 이너셔(엘보우 방법)")

k_range = list(range(1, 8))
inertia_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(
        x=k_range,
        y=inertia_list,
        mode="lines+markers",
        name="이너셔",
    )
)
# 현재 선택한 묶음 수 위치에 세로선 긋기
fig_elbow.add_vline(
    x=n_clusters,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택: {n_clusters}",
    annotation_position="top",
)
fig_elbow.update_layout(
    title="묶음 수에 따른 이너셔 변화",
    xaxis_title="묶음 수",
    yaxis_title="이너셔 (거리 제곱의 합)",
    xaxis=dict(tickmode="linear", dtick=1),
)
st.plotly_chart(fig_elbow, use_container_width=True)

# ---------------------------------------------------------
# 묶음 수별 이너셔와 바로 앞 값과의 감소량 표
# ---------------------------------------------------------
elbow_rows = []
prev_inertia = None
for k, inertia in zip(k_range, inertia_list):
    if prev_inertia is None:
        decrease = None
    else:
        decrease = round(prev_inertia - inertia, 2)
    elbow_rows.append(
        {
            "묶음 수": k,
            "이너셔": round(inertia, 2),
            "바로 앞 값에서 감소량": decrease if decrease is not None else "",
        }
    )
    prev_inertia = inertia

elbow_df = pd.DataFrame(elbow_rows)
st.dataframe(elbow_df, use_container_width=True)

# ---------------------------------------------------------
# 실루엣 점수: 묶음 수 2~7
# ---------------------------------------------------------
st.subheader("4. 묶음 수에 따른 실루엣 점수")

sil_k_range = list(range(2, 8))
sil_scores = []

for k in sil_k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_tmp)
    sil_scores.append(score)

fig_sil = go.Figure()
fig_sil.add_trace(
    go.Scatter(
        x=sil_k_range,
        y=sil_scores,
        mode="lines+markers",
        name="실루엣 점수",
        line=dict(color="green"),
    )
)
fig_sil.add_vline(
    x=n_clusters,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택: {n_clusters}",
    annotation_position="top",
)
fig_sil.update_layout(
    title="묶음 수에 따른 실루엣 점수 변화",
    xaxis_title="묶음 수",
    yaxis_title="실루엣 점수",
    xaxis=dict(tickmode="linear", dtick=1),
)
st.plotly_chart(fig_sil, use_container_width=True)

sil_df = pd.DataFrame(
    {
        "묶음 수": sil_k_range,
        "실루엣 점수": [round(s, 4) for s in sil_scores],
    }
)
st.dataframe(sil_df, use_container_width=True)

# ---------------------------------------------------------
# 실제 군집화 (현재 선택한 묶음 수로 K-평균, 난수 고정)
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
df["cluster_raw"] = raw_labels

# ---------------------------------------------------------
# 묶음 번호를 슈팅 평균이 큰 순서로 ㉮, ㉯, ㉰... 재매핑
# ---------------------------------------------------------
label_names_all = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]
label_names = label_names_all[:n_clusters]

cluster_order = (
    df.groupby("cluster_raw")["shooting"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_label_map = {raw_id: label_names[i] for i, raw_id in enumerate(cluster_order)}
df["cluster"] = df["cluster_raw"].map(cluster_label_map)

# ---------------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------------
st.subheader("5. 2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "가로축 능력치",
        options=selected_stats,
        index=0,
        format_func=lambda x: stat_label_map[x],
        key="x_2d",
    )
with col2:
    y_axis_2d = st.selectbox(
        "세로축 능력치",
        options=selected_stats,
        index=min(1, len(selected_stats) - 1),
        format_func=lambda x: stat_label_map[x],
        key="y_2d",
    )

fig_2d = px.scatter(
    df,
    x=x_axis_2d,
    y=y_axis_2d,
    color="cluster",
    hover_name="name_ko",
    labels={
        x_axis_2d: stat_label_map[x_axis_2d],
        y_axis_2d: stat_label_map[y_axis_2d],
        "cluster": "묶음",
    },
    category_orders={"cluster": label_names},
    title="2차원 산점도",
)
st.plotly_chart(fig_2d, use_container_width=True)

# ---------------------------------------------------------
# 3차원 산점도 (선택 능력치가 3개 이상일 때만)
# ---------------------------------------------------------
st.subheader("6. 3차원 산점도")

if len(selected_stats) < 3:
    st.info("3차원 산점도를 보려면 능력치를 3개 이상 선택해주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "X축 능력치",
            options=selected_stats,
            index=0,
            format_func=lambda x: stat_label_map[x],
            key="x_3d",
        )
    with col4:
        y_axis_3d = st.selectbox(
            "Y축 능력치",
            options=selected_stats,
            index=min(1, len(selected_stats) - 1),
            format_func=lambda x: stat_label_map[x],
            key="y_3d",
        )
    with col5:
        z_axis_3d = st.selectbox(
            "Z축 능력치",
            options=selected_stats,
            index=min(2, len(selected_stats) - 1),
            format_func=lambda x: stat_label_map[x],
            key="z_3d",
        )

    fig_3d = px.scatter_3d(
        df,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color="cluster",
        hover_name="name_ko",
        labels={
            x_axis_3d: stat_label_map[x_axis_3d],
            y_axis_3d: stat_label_map[y_axis_3d],
            z_axis_3d: stat_label_map[z_axis_3d],
            "cluster": "묶음",
        },
        category_orders={"cluster": label_names},
        title="3차원 산점도",
    )
    # 점 크기를 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# 묶음별 인원과 여섯 능력치 평균
# ---------------------------------------------------------
st.subheader("7. 묶음별 인원과 능력치 평균")

summary_rows = []
for label in label_names:
    sub = df[df["cluster"] == label]
    if len(sub) == 0:
        continue
    row = {"묶음": label, "인원": len(sub)}
    for stat in stat_cols:
        row[stat_label_map[stat]] = round(sub[stat].mean(), 1)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------------
# 묶음별 종합 능력치(overall) 상위 5명 (한글 이름)
# ---------------------------------------------------------
st.subheader("8. 묶음별 종합 능력치 상위 5명")

for label in label_names:
    sub = df[df["cluster"] == label].sort_values("overall", ascending=False)
    if len(sub) == 0:
        continue
    top5 = sub.head(5)["name_ko"].tolist()
    st.write(f"**{label} 묶음 상위 5명:** " + ", ".join(top5))

# ---------------------------------------------------------
# 묶음과 포지션 교차표
# ---------------------------------------------------------
st.subheader("9. 묶음과 포지션 교차표")

cross_tab = pd.crosstab(df["cluster"], df["position_group"])
# 묶음 순서를 label_names 순서대로 정렬
cross_tab = cross_tab.reindex(label_names)

st.dataframe(cross_tab, use_container_width=True)
