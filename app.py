import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from dlclive import DLCLive, Processor
from collections import deque
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Desactiva GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Elimina avisos innecesarios
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Monitor Cairo AI - UNAP", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .subheader-text {
        font-size: 18px;
        color: #e0e0e0;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    
    .estado-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    
    .control-panel {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    
    .divider-custom {
        border: 1px solid #e0e0e0;
        margin: 20px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 14px;
        margin-top: 10px;
    }
    
    .pose-info {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-top: 10px;
    }
    
    h1 {
        color: #2c3e50;
    }
    
    h3 {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER MEJORADO ---
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 36px;">🐕 Sistema de Monitoreo Veterinario</h1>
    <p class="subheader-text">Análisis Biomecánico, Etológico y Cinético - Proyecto Cairo v2.5</p>
</div>
""", unsafe_allow_html=True)

# RUTA DEL MODELO DLC
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODELO = os.path.join(BASE_DIR, "exported-models", "DLC_DAPS_Protocol_efficientnet-b0_iteration-1_shuffle-1")

# --- CLASE PARA FILTRADO KALMAN ---
class KalmanPointTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q *= 0.01
        
    def update(self, x, y):
        self.kf.predict()
        self.kf.update(np.array([x, y]))
        return float(self.kf.x[0]), float(self.kf.x[1])

# --- CLASE PRINCIPAL DE ANÁLISIS ---
class AnalizadorBiomecanicoCairo:
    def __init__(self, confidence_threshold=0.6, buffer_size=30):
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.pose_history = deque(maxlen=buffer_size)
        self.kalman_filters = {}
        self.animal_size_ref = None
        self.size_calibrated = False
        self.estado_anterior = None
        self.frames_en_estado = 0
        
    def filtrar_confianza(self, pose_raw):
        pose_filtrada = []
        for idx, punto in enumerate(pose_raw):
            x, y, conf = float(punto[0]), float(punto[1]), float(punto[2])
            if conf < self.confidence_threshold and len(self.pose_history) > 0:
                x, y = self.pose_history[-1][idx][0], self.pose_history[-1][idx][1]
            if idx not in self.kalman_filters:
                self.kalman_filters[idx] = KalmanPointTracker()
            xf, yf = self.kalman_filters[idx].update(x, y)
            pose_filtrada.append((xf, yf, conf))
        return pose_filtrada

    def calcular_angulo(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]]).flatten()
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]]).flatten()
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-4 or norm2 < 1e-4: return 0.0
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def analizar_estado(self, pose, u):
        u_suelo, u_cojera, u_reposo, u_sentado, u_juego, min_frames = u
        self.pose_history.append(pose)
        
        if len(self.pose_history) < 5:
            return "CALIBRANDO...", (128, 128, 128), {}

        nariz, cruz, cadera = pose[0], pose[2], pose[3]
        r_izq, pie_izq = pose[11], pose[13]
        r_der, pie_der = pose[14], pose[16]

        if not self.size_calibrated:
            self.animal_size_ref = np.linalg.norm(np.array([cruz[0]-cadera[0], cruz[1]-cadera[1]]))
            if self.animal_size_ref > 20: self.size_calibrated = True

        mov = np.std([p[2][0] for p in self.pose_history]) + np.std([p[2][1] for p in self.pose_history])
        
        aceleracion = 0
        if len(self.pose_history) > 2:
            v1 = np.linalg.norm(np.array(pose[2][:2]) - np.array(self.pose_history[-1][2][:2]))
            v2 = np.linalg.norm(np.array(self.pose_history[-1][2][:2]) - np.array(self.pose_history[-2][2][:2]))
            aceleracion = abs(v1 - v2)

        ang_izq = self.calcular_angulo(cadera, r_izq, pie_izq)
        ang_der = self.calcular_angulo(cadera, r_der, pie_der)
        asim = abs(ang_izq - ang_der)

        estado_nuevo, color = "MOVIMIENTO NORMAL", (0, 255, 0)

        if mov < u_reposo:
            estado_nuevo, color = "REPOSO / DURMIENDO", (255, 0, 0)
        elif mov > u_juego and aceleracion > (u_juego * 0.5):
            estado_nuevo, color = "✨ JUGANDO / CORRIENDO", (255, 105, 180)
        elif abs(cadera[1] - pie_der[1]) < (self.animal_size_ref * u_sentado if self.animal_size_ref else 60):
            estado_nuevo, color = "SENTADO / EN ESPERA", (255, 255, 0)
        elif mov > (u_reposo * 1.5) and asim > u_cojera:
            estado_nuevo, color = "⚠️ ALERTA: COJERA", (0, 0, 255)
        elif nariz[1] > u_suelo:
            estado_nuevo, color = "COMIENDO / OLFATEANDO", (0, 255, 255)

        if self.estado_anterior == estado_nuevo: self.frames_en_estado += 1
        else:
            if self.frames_en_estado < min_frames: estado_nuevo = self.estado_anterior if self.estado_anterior else estado_nuevo
            else: self.frames_en_estado = 0; self.estado_anterior = estado_nuevo

        met = {"Asimetría": f"{asim:.1f}°", "Energía": f"{mov:.1f}", "Acel": f"{aceleracion:.1f}"}
        return estado_nuevo, color, met

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(PATH_MODELO): 
        st.error(f"⚠️ Ruta no encontrada: {PATH_MODELO}")
        return None
    try:
        dlc_proc = Processor()
        dlc_live = DLCLive(PATH_MODELO, processor=dlc_proc)
        dlc_live.init_inference()
        return dlc_live
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

# --- SIDEBAR MEJORADO ---
st.sidebar.markdown("### 🕹️ Panel de Control Cairo")
st.sidebar.markdown("---")
u_suelo = st.sidebar.slider("📏 Umbral Suelo (Y)", 300, 900, 600)
u_cojera = st.sidebar.slider("🔍 Sensibilidad Cojera", 10, 50, 25)
u_reposo = st.sidebar.slider("😴 Filtro Reposo", 0.5, 10.0, 3.0)
u_sentado = st.sidebar.slider("🪑 Factor Sentado", 0.1, 1.5, 0.7)
u_juego = st.sidebar.slider("⚡ Energía de Juego", 5.0, 50.0, 25.0)
conf_dlc = st.sidebar.slider("📊 Confianza DLC", 0.1, 0.9, 0.5)

modelo_ai = cargar_modelo()

# --- GESTIÓN DE ENTRADA ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📹 Fuente de Entrada")
modo = st.sidebar.radio("Selecciona Fuente:", ("Cámara Live", "Subir Video"))

# Variables compartidas para WebRTC
if 'estado_actual' not in st.session_state:
    st.session_state.estado_actual = "ESPERANDO..."
if 'metricas_actuales' not in st.session_state:
    st.session_state.metricas_actuales = {"Asimetría": "0°", "Energía": "0", "Acel": "0"}
if 'pose_actual' not in st.session_state:
    st.session_state.pose_actual = None

# --- CLASE VIDEO PROCESSOR PARA WEBRTC ---
class CairoVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.modelo = modelo_ai
        self.analizador = None
        self.esqueleto = [(0,2),(2,3),(3,4),(2,5),(5,6),(6,7),(2,8),(8,9),(9,10),(3,11),(11,12),(12,13),(3,14),(14,15),(15,16)]
        self.lock = threading.Lock()
        self.frame_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inicializar analizador en el primer frame
        if self.analizador is None:
            self.analizador = AnalizadorBiomecanicoCairo(confidence_threshold=conf_dlc)
        
        if self.modelo is None:
            cv2.putText(img, "Modelo no cargado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        self.frame_count += 1
        
        try:
            # Convertir imagen a RGB para DLC
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Inferencia
            raw = self.modelo.get_pose(img_rgb)
            
            # Verificar que raw tenga datos válidos
            if raw is None or len(raw) == 0:
                cv2.putText(img, "Sin deteccion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            pose = self.analizador.filtrar_confianza(raw)
            estado, color, met = self.analizador.analizar_estado(
                pose, 
                (u_suelo, u_cojera, u_reposo, u_sentado, u_juego, 6)
            )
            
            # Actualizar estado global
            with self.lock:
                st.session_state.estado_actual = estado
                st.session_state.metricas_actuales = met
                st.session_state.pose_actual = pose
            
            # Renderizado
            cv2.rectangle(img, (10,10), (550,90), (0,0,0), -1)
            cv2.putText(img, estado, (25,55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Dibujar esqueleto
            for c in self.esqueleto:
                if c[0] < len(pose) and c[1] < len(pose):
                    p1, p2 = pose[c[0]], pose[c[1]]
                    if p1[2] > conf_dlc and p2[2] > conf_dlc:  # Solo dibujar si ambos puntos tienen buena confianza
                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255,255,255), 2)
            
            # Dibujar puntos
            for idx, p in enumerate(pose):
                if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:  # Verificar que esté dentro de la imagen
                    color_punto = (0,255,0) if p[2]>conf_dlc else (0,165,255)
                    cv2.circle(img, (int(p[0]), int(p[1])), 5, color_punto, -1)
            
            # Mostrar FPS
            cv2.putText(img, f"Frame: {self.frame_count}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
        except Exception as e:
            error_msg = str(e)[:50]
            cv2.putText(img, f"Error: {error_msg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            print(f"Error en procesamiento: {e}")  # Para debug en consola
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- FUNCIÓN DE PROCESAMIENTO VIDEO ---
def run_video(path):
    if modelo_ai is None:
        st.error("Modelo no cargado.")
        return

    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        st.error("No se pudo acceder al video.")
        return

    col_video, col_control = st.columns([1.5, 1])
    
    analizador = AnalizadorBiomecanicoCairo(confidence_threshold=conf_dlc)
    esqueleto = [(0,2),(2,3),(3,4),(2,5),(5,6),(6,7),(2,8),(8,9),(9,10),(3,11),(11,12),(12,13),(3,14),(14,15),(15,16)]

    col_btn_left, col_btn_right = st.columns([1.5, 1])
    with col_btn_left:
        stop = st.button("⏹️ Detener Procesamiento", use_container_width=True, key="btn_stop")

    with col_video:
        st.markdown("### 🎥 Transmisión en Vivo")
        st_frame = st.empty()
    
    with col_control:
        st_estado = st.empty()
        st_met = st.empty()
        st_info = st.empty()

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret: break
        
        raw = modelo_ai.get_pose(frame)
        pose = analizador.filtrar_confianza(raw)
        estado, color, met = analizador.analizar_estado(pose, (u_suelo, u_cojera, u_reposo, u_sentado, u_juego, 6))
        
        cv2.rectangle(frame, (10,10), (550,90), (0,0,0), -1)
        cv2.putText(frame, estado, (25,55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        for c in esqueleto:
            p1, p2 = pose[c[0]], pose[c[1]]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255,255,255), 2)
        for p in pose:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0,255,0) if p[2]>conf_dlc else (0,165,255), -1)

        with col_video:
            st_frame.image(frame, channels="BGR", use_container_width=True)
        
        with col_control:
            st_estado.markdown(f"""
            <div class="estado-card">
                {estado}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            
            with st_met.container():
                st.markdown("#### 🔍 Métricas")
                st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 22px; margin-bottom: 10px;">📐</div>
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{met.get("Asimetría", "0°")}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">Asimetría</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin: 12px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 22px; margin-bottom: 10px;">⚡</div>
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{met.get("Energía", "0")}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">Energía Cinética</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin: 12px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 22px; margin-bottom: 10px;">🚀</div>
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{met.get("Acel", "0")}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">Aceleración</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            
            with st_info.container():
                st.markdown("#### 🦴 Posición")
                st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="pose-info">
                    <div style="margin-bottom: 10px;"><strong>👃 Nariz:</strong> ({pose[0][0]:.0f}, {pose[0][1]:.0f})</div>
                    <div style="margin-bottom: 10px;"><strong>💪 Pecho:</strong> ({pose[2][0]:.0f}, {pose[2][1]:.0f})</div>
                    <div><strong>🦴 Cadera:</strong> ({pose[3][0]:.0f}, {pose[3][1]:.0f})</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown("#### ⚙️ Configuración")
                st.markdown('<div style="margin: 8px 0;"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-box">
                    <div style="margin-bottom: 6px;">✅ <strong>Confianza:</strong> {conf_dlc:.2f}</div>
                    <div>⚡ <strong>Energía:</strong> {u_juego:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

    cap.release()
    if path:
        os.unlink(path)

# --- INTERFAZ PRINCIPAL ---
if modo == "Cámara Live":
    # Configuración RTC para mejor compatibilidad
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    col_video, col_control = st.columns([1.5, 1])
    
    with col_video:
        st.markdown("### 🎥 Transmisión en Vivo")
        webrtc_ctx = webrtc_streamer(
            key="cairo-monitor",
            video_processor_factory=CairoVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col_control:
        st_estado = st.empty()
        st_met = st.empty()
        st_info = st.empty()
        
        # Actualizar interfaz cada segundo
        if webrtc_ctx.state.playing:
            estado = st.session_state.estado_actual
            met = st.session_state.metricas_actuales
            pose = st.session_state.pose_actual
            
            st_estado.markdown(f"""
            <div class="estado-card">
                {estado}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            
            with st_met.container():
                st.markdown("#### 🔍 Métricas")
                st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 22px; margin-bottom: 10px;">📐</div>
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{met.get("Asimetría", "0°")}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">Asimetría</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin: 12px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 22px; margin-bottom: 10px;">⚡</div>
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{met.get("Energía", "0")}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">Energía Cinética</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin: 12px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 22px; margin-bottom: 10px;">🚀</div>
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{met.get("Acel", "0")}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">Aceleración</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            
            with st_info.container():
                st.markdown("#### 🦴 Posición")
                st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
                if pose:
                    st.markdown(f"""
                    <div class="pose-info">
                        <div style="margin-bottom: 10px;"><strong>👃 Nariz:</strong> ({pose[0][0]:.0f}, {pose[0][1]:.0f})</div>
                        <div style="margin-bottom: 10px;"><strong>💪 Pecho:</strong> ({pose[2][0]:.0f}, {pose[2][1]:.0f})</div>
                        <div><strong>🦴 Cadera:</strong> ({pose[3][0]:.0f}, {pose[3][1]:.0f})</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
                
                st.markdown("#### ⚙️ Configuración")
                st.markdown('<div style="margin: 8px 0;"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-box">
                    <div style="margin-bottom: 6px;">✅ <strong>Confianza:</strong> {conf_dlc:.2f}</div>
                    <div>⚡ <strong>Energía:</strong> {u_juego:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    f = st.file_uploader("Sube el video de Cairo", type=["mp4", "mov", "avi"])
    if f:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(f.read())
        video_path = tfile.name
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🚀 INICIAR PROCESAMIENTO", use_container_width=True, key="btn_start"):
                run_video(video_path)
    else:
        st.info("👆 Sube un video para comenzar el análisis")
