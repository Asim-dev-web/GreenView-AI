import streamlit as st
import leafmap.foliumap as leafmap
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
from folium.plugins import LocateControl, Geocoder
import zipfile
import io
import streamlit.components.v1 as components
import base64

st.set_page_config(
    page_title="GreenView AI", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🌲"
)

def get_logo_url():
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none">
      <circle cx="50" cy="50" r="48" fill="#020617" stroke="#10B981" stroke-width="2"/>
      <path d="M50 25C50 25 20 50 50 75C80 50 50 25 50 25Z" fill="#065f46" stroke="#34D399" stroke-width="2"/>
      <path d="M50 35C50 35 35 50 50 65C65 50 50 35 50 35Z" fill="#10B981"/>
      <circle cx="50" cy="50" r="8" fill="#0f172a"/>
    </svg>
    """
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

st.logo(get_logo_url(), icon_image=get_logo_url())

components.html(
    """
    <script>
        function forceTop() {
            var main = window.parent.document.querySelector(".main");
            if (main) { main.scrollTop = 0; }
        }
        var interval = setInterval(forceTop, 50);
        setTimeout(function() { clearInterval(interval); }, 1000);
    </script>
    """,
    height=0,
)

@st.cache_resource
def load_engine():
    model = smp.UnetPlusPlus("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load("summit_final_model.pth", map_location="cpu"))
    model.eval()
    return model

if 'lat' not in st.session_state: 
    st.session_state.update({'lat': 28.6327, 'lon': 77.2195, 'zoom': 17})

download_icon_svg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMyAxNnYyYTIgMiAwIDAgMCAyIDJoMTRhMiAyIDAgMCAwIDItMnYtMiIvPjxwb2x5bGluZSBwb2ludHM9IjggMTIgMTIgMTYgMTYgMTIiLz48bGluZSB4MT0iMTIiIHkxPSIxNiIgeDI9IjEyIiB5Mj0iMyIvPjwvc3ZnPg=="
help_icon_svg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMTBCOTgxIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik05LjA5IDlhMyAzIDAgMCAxIDUuODMgMS41YzAgMi0zIDMtMyAzIi8+PGxpbmUgeDE9IjEyIiB5MT0iMTciIHgyPSIxMi4wMSIgeTI9IjE3Ii8+PC9zdmc+"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Outfit:wght@300;600&display=swap');
    
    .block-container {{ padding-top: 1.5rem !important; padding-bottom: 2rem !important; }}
    
    .stApp {{ 
        background: radial-gradient(circle at top right, #0f172a, #020617) !important; 
        color: #F8FAFC !important; 
    }}

    .help-container {{
        position: fixed; bottom: 30px; right: 30px; z-index: 9999; font-family: 'Outfit', sans-serif;
    }}
    .help-icon {{
        width: 50px; height: 50px; background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981;
        border-radius: 50%; backdrop-filter: blur(10px); display: flex; align-items: center;
        justify-content: center; cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.3); transition: all 0.3s ease;
    }}
    .help-icon:hover {{ background: rgba(16, 185, 129, 0.2); box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); transform: scale(1.1); }}
    .help-icon img {{ width: 24px; height: 24px; }}
    
    .tooltip-text {{
        visibility: hidden; width: 260px; background-color: #0f172a; color: #fff;
        text-align: left; border-radius: 12px; padding: 15px; position: absolute; z-index: 1;
        bottom: 125%; right: 0; margin-left: -60px; opacity: 0; transition: opacity 0.3s;
        border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        font-size: 14px; line-height: 1.5;
    }}
    .tooltip-text strong {{ color: #34D399; }}
    .help-container:hover .tooltip-text {{ visibility: visible; opacity: 1; }}
    .tooltip-text::after {{
        content: ""; position: absolute; top: 100%; right: 20px; margin-left: -5px;
        border-width: 5px; border-style: solid; border-color: #10B981 transparent transparent transparent;
    }}

    .brand-title {{ 
        position: relative; font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -2px;
        line-height: 1.1; margin-bottom: 5px; z-index: 1;
        background: linear-gradient(90deg, transparent 10%, rgba(52, 211, 153, 0.6) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; -webkit-text-stroke: 1px #34D399;
    }}
    .brand-title::before {{
        content: attr(data-text); position: absolute; left: 0; top: 0; width: 100%; height: 100%;
        z-index: -1; color: transparent; -webkit-text-stroke: 0; text-shadow: 0 0 20px rgba(52, 211, 153, 0.6); 
        -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 100%); mask-image: linear-gradient(90deg, transparent 0%, black 100%);
    }}
    .brand-subtitle {{ font-family: 'Outfit', sans-serif; color: #94A3B8; margin-bottom: 30px; }}
    .brand-title {{ font-size: 64px; text-align: left; }}
    .brand-subtitle {{ font-size: 18px; text-align: left; }}
    
    @media (max-width: 768px) {{
        .brand-title {{ font-size: 24px !important; text-align: center !important; white-space: nowrap !important; }}
        .brand-subtitle {{ font-size: 12px !important; text-align: center !important; margin-bottom: 15px !important; }}
    }}

    div[data-testid="stButton"] button {{
        background: linear-gradient(90deg, #065f46, #0d9488) !important;
        box-shadow: none !important; border: none !important; color: white !important;
        font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
        font-size: 16px !important; letter-spacing: 2px !important; text-transform: uppercase !important;
        border-radius: 12px !important; padding: 16px 24px !important;
        width: 100% !important; transition: all 0.3s ease !important;
    }}
    div[data-testid="stButton"] button:hover {{
        background: linear-gradient(90deg, #047857, #14b8a6) !important;
        box-shadow: 0 4px 20px rgba(13, 148, 136, 0.4) !important;
        transform: translateY(-2px) !important; letter-spacing: 3px !important;
    }}
    div[data-testid="stButton"] button:active {{ transform: translateY(1px) !important; box-shadow: none !important; }}

    iframe {{
        border-radius: 20px !important; border: 2px solid rgba(52, 211, 153, 0.3) !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2), 0 0 40px rgba(99, 102, 241, 0.15) !important;
        transition: all 0.3s ease-in-out;
    }}
    iframe:hover {{ border-color: rgba(52, 211, 153, 0.6) !important; box-shadow: 0 0 25px rgba(16, 185, 129, 0.4), 0 0 60px rgba(99, 102, 241, 0.3) !important; }}

    [data-testid="stDownloadButton"] {{ display: flex; justify-content: flex-end; }}
    [data-testid="stDownloadButton"] button {{
        background: linear-gradient(135deg, #10B981, #047857) !important;
        width: 50px !important; height: 50px !important; border-radius: 50% !important;
        border: 1px solid rgba(255,255,255,0.1) !important; z-index: 999;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        transition: transform 0.1s ease-in-out, box-shadow 0.2s ease !important;
        color: transparent !important;
    }}
    
    @media (min-width: 769px) {{
        [data-testid="stDownloadButton"] button {{ transform: translateY(85px); margin-right: -150px !important; }}
        [data-testid="stDownloadButton"] button:hover {{ transform: translateY(82px) scale(1.05) !important; box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important; }}
        [data-testid="stDownloadButton"] button:active {{ transform: translateY(85px) scale(0.95) !important; }}
    }}
    @media (max-width: 768px) {{
        [data-testid="stDownloadButton"] button {{ transform: translateY(10px); margin-right: 0px !important; }}
        [data-testid="stDownloadButton"] button:active {{ transform: translateY(10px) scale(0.95) !important; }}
    }}
    
    [data-testid="stDownloadButton"] button::after {{
        content: ""; display: block; width: 24px; height: 24px;
        background-image: url('{download_icon_svg}'); background-repeat: no-repeat;
        background-position: center center; background-size: contain;
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    }}
    [data-testid="stMetricValue"] {{ color: #34D399 !important; font-family: 'Syne', sans-serif; font-size: 42px !important; }}
    #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div class="help-container">
        <div class="tooltip-text">
            <strong>Optimal Accuracy Tip</strong><br><br>
            Zoom in until the map scale bar (bottom right) reads <strong>50m</strong> or <strong>100ft</strong>.<br><br>
            This level of detail is required for the AI to correctly identify individual trees.
        </div>
        <div class="help-icon">
            <img src="{help_icon_svg}">
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="brand-title" data-text="GREENVIEW AI">GREENVIEW AI</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Autonomous urban canopy intelligence mapping.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.7, 1], gap="large")

with col_left:
    st.markdown("##### 1. Locate Area")
    m = leafmap.Map(center=[st.session_state.lat, st.session_state.lon], zoom=st.session_state.zoom)
    m.add_tile_layer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', name='Satellite', attribution='Google')
    LocateControl().add_to(m) 
    Geocoder().add_to(m) 
    
    map_data = m.to_streamlit(height=480, responsive=True, key="geo_force_view")
    
    if isinstance(map_data, dict) and 'center' in map_data:
        st.session_state.lat = map_data['center']['lat']
        st.session_state.lon = map_data['center']['lng']

with col_right:
    st.markdown("##### 2. Run Analysis")
    file = st.file_uploader("Upload Map Screenshot", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if st.button("GENERATE MASK"):
        if file:
            model = load_engine() 
            with st.spinner("Analyzing vegetation density..."):
                img = Image.open(file).convert("RGB")
                img = np.array(img)
                
                h, w, c = img.shape
                pad_h = (32 - h % 32) % 32
                pad_w = (32 - w % 32) % 32
                
                img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                
                x_tensor = torch.tensor(img_padded.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    pred = torch.sigmoid(model(x_tensor)).cpu().squeeze().numpy()
                
                pred = pred[:h, :w]
                
                heatmap = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                
                st.session_state.gv_res = {
                    "overlay": overlay, "mask": heatmap, "input": img,
                    "score": np.mean(pred > 0.45) * 100
                }
                st.rerun() 
        else:
            st.error("Please upload a file first.")

if "gv_res" in st.session_state:
    st.write("---")
    
    res_head, res_dl = st.columns([5, 1])
    
    with res_head: 
        st.markdown("### Result Analysis")
    
    with res_dl:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "x") as csv_zip:
            for name, data in [("source.png", st.session_state.gv_res["input"]), 
                               ("heatmap.png", st.session_state.gv_res["mask"]), 
                               ("combined.png", st.session_state.gv_res["overlay"])]:
                img_pil = Image.fromarray(data)
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format='PNG')
                csv_zip.writestr(name, img_byte_arr.getvalue())
        
        st.download_button(label=" ", data=buf.getvalue(), file_name="greenview_analysis.zip", mime="application/zip")

    st.metric("Canopy Coverage", f"{st.session_state.gv_res['score']:.1f}%")
    
    r1, r2, r3 = st.columns(3)
    r1.image(st.session_state.gv_res["input"], caption="Source Image")
    r2.image(st.session_state.gv_res["mask"], caption="Plantation Predicted Heatmap")
    r3.image(st.session_state.gv_res["overlay"], caption="Combined Overlay")