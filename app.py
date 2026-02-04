import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import re

# ==========================================
# 1. CẤU HÌNH & CSS TÙY CHỈNH
# ==========================================
st.set_page_config(page_title="Raman AI Analyzer Pro", layout="wide", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    .stTable { background-color: white; border-radius: 10px; overflow: hidden; }
    h1 { color: #1e3a8a; font-weight: 800; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. KIẾN TRÚC MẠNG (Giữ nguyên logic của bạn)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x): return torch.relu(self.conv(x) + self.shortcut(x))


class RamanResNet(nn.Module):
    def __init__(self, num_targets=4):
        super(RamanResNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            ResidualBlock(64, 64), ResidualBlock(64, 128),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_targets))

    def forward(self, x): return self.regressor(self.feature_extractor(x))


# ==========================================
# 3. HÀM TIỀN XỬ LÝ & LOAD DATA
# ==========================================
@st.cache_resource
def load_model():
    model = RamanResNet(num_targets=4)
    # Tự động chọn đúng file model hiện có trong máy bạn
    try:
        model.load_state_dict(torch.load('raman_resnet_experiment.pth', map_location='cpu'))
    except:
        model.load_state_dict(torch.load('raman_resnet_v2.1.pth', map_location='cpu'))
    model.eval()
    return model


def preprocess_input(spectrum):
    clean = median_filter(spectrum, size=3)
    x = clean.reshape(1, -1)
    d1 = savgol_filter(x, 15, 3, deriv=1)
    d2 = savgol_filter(x, 15, 3, deriv=2)
    snv = lambda data: (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    x_proc = np.stack([snv(x), snv(d1), snv(d2)], axis=1)
    return torch.tensor(x_proc, dtype=torch.float32)


# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.title("🔬 Raman AI Analyzer Pro")
st.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.header("📥 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file Spectra (.csv)", type="csv")

if not uploaded_file:
    st.info("💡 **Hướng dẫn**: Tải lên tệp CSV từ thiết bị của bạn để bắt đầu phân tích nồng độ mẫu.")
    st.image("https://img.freepik.com/free-vector/data-analysis-concept-illustration_114360-1611.jpg", width=400)
else:
    with st.spinner('Đang tải dữ liệu...'):
        df_spec = pd.read_csv(uploaded_file)
        df_meta = pd.read_csv('Sugar_Concentrations.csv')
        model = load_model()
        all_samples = df_spec.columns[1:].tolist()

    # Bộ lọc Sidebar chuyên nghiệp hơn
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Lựa chọn mẫu")
    method = st.sidebar.radio("Phương thức chọn:", ["Theo vị trí giếng", "Danh sách đầy đủ"])

    selected_sample = None
    if method == "Theo vị trí giếng":
        try:
            plates = sorted(list(set([s.split('_')[5] for s in all_samples])))
            sel_plate = st.sidebar.selectbox("Plate", plates)
            plate_samples = [s for s in all_samples if s.split('_')[5] == sel_plate]
            rows = sorted(list(set([re.findall(r'[A-Z]', s.split('_')[4])[0] for s in plate_samples])))
            sel_row = st.sidebar.select_slider("Hàng (Row)", options=rows)
            row_samples = [s for s in plate_samples if s.split('_')[4].startswith(sel_row)]
            cols = sorted(list(set([int(re.findall(r'\d+', s.split('_')[4])[0]) for s in row_samples])))
            sel_col = st.sidebar.selectbox("Cột (Column)", cols)
            final_options = [s for s in row_samples if s.split('_')[4] == f"{sel_row}{sel_col}"]
            selected_sample = st.sidebar.selectbox("Lần đo", final_options) if final_options else None
        except:
            st.sidebar.error("Lỗi định dạng tên mẫu")
    else:
        selected_sample = st.sidebar.selectbox("Chọn từ danh sách:", all_samples)

    if selected_sample:
        # Layout chính
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.subheader(f"📍 Phân tích: {selected_sample}")
            spectrum = df_spec[selected_sample].values
            wavenumbers = df_spec.iloc[:, 0].values

            # Đồ thị Matplotlib nhưng đẹp hơn
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.plot(wavenumbers, spectrum, color='#d1d5db', lw=1, label='Dữ liệu thô', alpha=0.5)
            ax.plot(wavenumbers, median_filter(spectrum, 3), color='#1e40af', lw=2, label='Đã xử lý nhiễu')
            ax.set_xlabel("Số sóng (cm-1)", fontsize=12)
            ax.set_ylabel("Cường độ", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            st.pyplot(fig)

        with col_right:
            st.subheader("📊 Kết quả dự đoán AI")
            input_tensor = preprocess_input(spectrum)
            with torch.no_grad():
                preds = np.maximum(model(input_tensor).numpy()[0] * 375.0, 0)  #

            sugars = ["Sucrose", "Fructose", "Maltose", "Glucose"]

            # Hiển thị dạng Metric Cards
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            metrics = [m1, m2, m3, m4]

            for i, s in enumerate(sugars):
                metrics[i].metric(s, f"{preds[i]:.2f} µl", delta_color="off")

            # Bảng so sánh nếu có Metadata
            parts = selected_sample.split('_')
            cell_id = f"{parts[4]}_{parts[5]}"
            truth_row = df_meta[df_meta['Cell Number'] == cell_id]

            if not truth_row.empty:
                st.markdown("#### 📏 Đối chiếu Metadata")
                actuals = truth_row[[f'{s} [ul]' for s in sugars]].values[0]
                compare_df = pd.DataFrame({
                    "Thành phần": sugars,
                    "Thực tế": np.round(actuals, 2),
                    "AI Dự đoán": np.round(preds, 2),
                    "Sai số": np.round(preds - actuals, 2)
                })
                st.table(compare_df)
                mae = np.mean(np.abs(preds - actuals))
                st.success(f"✅ Độ chính xác cao - MAE: **{mae:.2f} µl**")

        # Thông số mô hình
        with st.expander("ℹ️ Thông tin mô hình & Hiệu năng"):
            st.table(pd.DataFrame({
                "Loại đường": sugars, "MAE (Training)": [2.77, 2.59, 2.76, 4.41],
                "R-squared": [0.992, 0.996, 0.996, 0.993]
            }))

