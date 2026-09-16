import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# 묶음 수 선택 (2~6, 기본 3)
# ---------------------------------------------------------
n_clusters = st.slider("묶음 수를 선택하세요", min_value=2, max_value=6, value=3)

# ---------------------------------------------------------
# 표준화 후 K-평균 군집화 (난수 고정)
# ---------------------------------------------------------
X = df[selected_stats].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
st.subheader("2. 2차원 산점도")

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
st.subheader("3. 3차원 산점도")

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
st.subheader("4. 묶음별 인원과 능력치 평균")

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
st.subheader("5. 묶음별 종합 능력치 상위 5명")

for label in label_names:
    sub = df[df["cluster"] == label].sort_values("overall", ascending=False)
    if len(sub) == 0:
        continue
    top5 = sub.head(5)["name_ko"].tolist()
    st.write(f"**{label} 묶음 상위 5명:** " + ", ".join(top5))

# ---------------------------------------------------------
# 묶음과 포지션 교차표
# ---------------------------------------------------------
st.subheader("6. 묶음과 포지션 교차표")

cross_tab = pd.crosstab(df["cluster"], df["position_group"])
# 묶음 순서를 label_names 순서대로 정렬
cross_tab = cross_tab.reindex(label_names)

st.dataframe(cross_tab, use_container_width=True)
