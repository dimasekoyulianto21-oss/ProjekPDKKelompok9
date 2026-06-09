import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.stats import pearsonr, spearmanr
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Dashboard Kelompok 9",
    layout="wide"
)

url_sst = "https://drive.google.com/uc?id=16Q5ICRrXeqbAJEbdOE370oh_-p0TO7w3"
url_chl = "https://drive.google.com/uc?id=1NUGuWdAbZypRuugfLz2zpNBEf7IJnOUA"

@st.cache_data
def load_data():
    sst = pd.read_csv(url_sst)
    chl = pd.read_csv(url_chl)
    return sst, chl

sst_df, chl_df = load_data()

sst_df["time"] = pd.to_datetime(sst_df["time"])
chl_df["time"] = pd.to_datetime(chl_df["time"])

sst_df["year"] = sst_df["time"].dt.year
chl_df["year"] = chl_df["time"].dt.year

sst_df["month"] = sst_df["time"].dt.month
chl_df["month"] = chl_df["time"].dt.month

sst_df["yearmonth"] = sst_df["time"].dt.to_period("M")
chl_df["yearmonth"] = chl_df["time"].dt.to_period("M")

sst_min = sst_df["thetao"].min()
sst_max = sst_df["thetao"].max()

chl_min = chl_df["chl"].min()
chl_max = chl_df["chl"].max()

st.sidebar.title("Dashboard Kelompok 9")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Home Dashboard",
        "Analisis SST dan Klorofil",
        "Unduh Data"
    ]
)

def interpolate(df, value_col):

    lon = df["longitude"].values
    lat = df["latitude"].values
    val = df[value_col].values

    grid_lon = np.linspace(lon.min(), lon.max(), 250)
    grid_lat = np.linspace(lat.min(), lat.max(), 250)

    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    grid_val = griddata(
        (lon, lat),
        val,
        (grid_lon, grid_lat),
        method="linear"
    )

    return grid_lon, grid_lat, grid_val

if menu == "Home Dashboard":

    st.title("Pengolahan Data Kelautan Kelompok 9")

    st.subheader("Ringkasan SST")

    col1,col2, col5 = st.columns(3)

    with col1:
        st.metric("SST Min", f"{sst_df['thetao'].min():.2f} °C")

    with col2:
        st.metric("SST Max", f"{sst_df['thetao'].max():.2f} °C")

    with col5:
        st.metric("SST Mean", f"{sst_df['thetao'].mean():.2f} °C")

    st.divider()

    st.subheader("Ringkasan Klorofil-a")

    col3,col4, col6 = st.columns(3)

    with col3:
        st.metric("Chl Min", f"{chl_df['chl'].min():.3f} mg/l")

    with col4:
        st.metric("Chl Max", f"{chl_df['chl'].max():.3f} mg/l")

    with col6:
        st.metric("Chl Mean", f"{chl_df['chl'].mean():.3f} mg/l")

    st.divider()

    fig1 = go.Figure()

    fig1.update_layout(
        mapbox=dict(
            style="carto-positron", 
            center=dict(lat=-5.5, lon=129),
            zoom=5
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Study Area Laut Banda"
    )

    fig1.add_trace(
        go.Scattermapbox(
            lat=[
                -6, -6,
                -5.0, -5.0,
                -6
            ],
            lon=[
                125.75, 129.25,
                129.25, 125.75,
                125.75
            ],
            mode="lines",
            line=dict(color="red", width=3),
            name="Study Area"
        )
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    sample_pts = sst_df.sample(
        min(1000, len(sst_df))
    )

    fig1.add_trace(
        go.Scattermapbox(
            lat=sample_pts["latitude"],
            lon=sample_pts["longitude"],
            mode="markers",
            marker=dict(
                size=3,
                color="blue"
            ),
            name="Data"
        )
    )

    year = st.selectbox(
        "Pilih Tahun Analisis",
        sorted(list(set(sst_df["year"].unique()) & set(chl_df["year"].unique())))
    )

    kolom1, kolom2 = st.columns(2)

    with kolom1:  
            
            df_year = ( sst_df[sst_df["year"] == year] .groupby(["latitude","longitude"], as_index=False)["thetao"] .mean() )

            grid_lon, grid_lat, grid_sst = interpolate(df_year, "thetao")

            fig = go.Figure()

            fig.add_trace(
                go.Contour(
                    z=grid_sst,
                    x=grid_lon[0],
                    y=grid_lat[:,0],
                    colorscale="balance",
                    contours=dict(coloring="heatmap"),
                    colorbar=dict(title="°C")
                )
            )

            fig.update_layout(
                title=f"Mean SST Laut Banda - Tahun {year}",
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)


    with kolom2:

        df_year1 = (
            chl_df[chl_df["year"] == year]
            .groupby(
                ["latitude", "longitude"],
                as_index=False
            )["chl"].mean()
        )

        grid_lon, grid_lat, grid_chl = interpolate(df_year1, "chl")

        fig1 = go.Figure()

        fig1.add_trace(
            go.Contour(
                z=grid_chl,
                x=grid_lon[0],
                y=grid_lat[:, 0],
                colorscale="YlGn",
                contours=dict(coloring="heatmap"),
                colorbar=dict(title="mg/m³")
            )
        )

        fig1.update_layout(
            title=f"Mean Klorofil-a Laut Banda - Tahun {year}",
            height=700
        )

        st.plotly_chart(fig1, use_container_width=True)



elif menu == "Analisis SST dan Klorofil":

    st.title("Analisis Hubungan SST dan Klorofil-a")

    sst_df["yearmonth"] = sst_df["time"].dt.to_period("M")
    chl_df["yearmonth"] = chl_df["time"].dt.to_period("M")

    sst_month = (
        sst_df
        .groupby("yearmonth")["thetao"]
        .mean()
        .reset_index()
    )

    chl_month = (
        chl_df
        .groupby("yearmonth")["chl"]
        .mean()
        .reset_index()
    )

    sst_month["yearmonth"] = sst_month["yearmonth"].dt.to_timestamp()
    chl_month["yearmonth"] = chl_month["yearmonth"].dt.to_timestamp()

    df_ts = pd.merge(
        sst_month,
        chl_month,
        on="yearmonth"
    )

    df_ts = df_ts.dropna()

    # tambah tahun
    df_ts["year"] = df_ts["yearmonth"].dt.year

    pearson_corr, _ = pearsonr(df_ts["thetao"], df_ts["chl"])
    spearman_corr, _ = spearmanr(df_ts["thetao"], df_ts["chl"])

    def interpret_corr(r):
        r_abs = abs(r)
        if r_abs >= 0.7:
            return "Kuat"
        elif r_abs >= 0.4:
            return "Sedang"
        else:
            return "Lemah"

    kategori_pearson = interpret_corr(pearson_corr)
    kategori_spearman = interpret_corr(spearman_corr)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Pearson Correlation", f"{pearson_corr:.3f}")

    with c2:
        st.metric("Spearman Correlation", f"{spearman_corr:.3f}")

    st.markdown(
        f"""
        ### Interpretasi Korelasi
        """
    )

    if pearson_corr < 0:

        st.info(
            """
            **Interpretasi Oseanografi:**

            Terdapat hubungan negatif antara SST dan klorofil-a.  
            Hal ini menunjukkan bahwa ketika suhu permukaan laut menurun, 
            konsentrasi klorofil-a cenderung meningkat.

            Fenomena ini umum terjadi akibat:
            - Upwelling (pengangkatan massa air dingin kaya nutrien)
            - Peningkatan produktivitas fitoplankton
            - Pengaruh monsun di Laut Banda
            """
        )

    else:

        st.info(
            """
            **Interpretasi Oseanografi:**

            Hubungan positif menunjukkan SST meningkat diikuti peningkatan klorofil-a, 
            yang dapat disebabkan oleh:
            - Stratifikasi kolom air
            - Variabilitas lokal
            - Kondisi perairan dangkal
            """
        )

    st.success(
        f"""
        Kesimpulan:
        Terdapat hubungan negatif antara SST dan klorofil-a 
        """
    )

    st.divider()

    st.subheader("Time Series SST dan Klorofil-a")

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    fig_ts.add_trace(
        go.Scatter(
            x=df_ts["yearmonth"],
            y=df_ts["thetao"],
            mode="lines+markers",
            name="SST",
            line=dict(color="#ff4d4d", width=2),
            marker=dict(size=4)
        ),
        secondary_y=False
    )

    fig_ts.add_trace(
        go.Scatter(
            x=df_ts["yearmonth"],
            y=df_ts["chl"],
            mode="lines+markers",
            name="Klorofil-a",
            line=dict(color="#00cc88", width=2),
            marker=dict(size=4)
        ),
        secondary_y=True
    )

    fig_ts.update_layout(
        title="Variabilitas Bulanan SST dan Klorofil-a Laut Banda",
        hovermode="x unified",
        height=650,
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="black")
    )

    fig_ts.update_xaxes(
        tickformat="%b\n%Y",
        dtick="M6",
        rangeslider_visible=True
    )

    fig_ts.update_yaxes(
        title_text="SST (°C)",
        secondary_y=False
    )

    fig_ts.update_yaxes(
        title_text="Klorofil-a (mg/m³)",
        secondary_y=True
    )

    st.plotly_chart(fig_ts, use_container_width=True)

    st.divider()

    st.subheader("Hubungan SST vs Klorofil-a")

    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scatter(
            x=df_ts["thetao"],
            y=df_ts["chl"],
            mode="markers",
            marker=dict(
                size=7,
                color=df_ts["thetao"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="SST"),
                opacity=1
            ),
            text=df_ts["yearmonth"]
        )
    )

    fig_scatter.update_layout(
        height=600,
        xaxis_title="SST (°C)",
        yaxis_title="Klorofil-a (mg/m³)",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="black")
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

elif menu == "Unduh Data":

    tahun = st.selectbox(
        "Pilih Tahun",
        sorted(
            sst_df["year"].unique()
        )
    )

    sst_filter = (
        sst_df[
            sst_df["year"] == tahun
        ]
    )

    chl_filter = (
        chl_df[
            chl_df["year"] == tahun
        ]
    )

    st.subheader("SST")

    st.dataframe(
        sst_filter,
        use_container_width=True
    )

    st.download_button(
        "Download SST",
        sst_filter.to_csv(index=False),
        file_name="sst.csv"
    )

    st.divider()

    st.subheader("Klorofil-a")

    st.dataframe(
        chl_filter,
        use_container_width=True
    )

    st.download_button(
        "Download Chlorophyll",
        chl_filter.to_csv(index=False),
        file_name="chlorophyll.csv"
    )

st.caption("PDK K9")
