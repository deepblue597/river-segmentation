import streamlit as st
from datetime import datetime
from PIL import Image
import io
import requests
import time
import os 

# Configure page with professional styling
st.set_page_config(
    page_title="River Segmentation AI Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .status-success { border-left-color: #28a745; background: #f8fff9; }
    .status-error { border-left-color: #dc3545; background: #fff8f8; }
    .status-warning { border-left-color: #ffc107; background: #fffdf7; }
    .status-info { border-left-color: #17a2b8; background: #f7fdff; }
    
    .processing-steps {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration

BACKEND_IP = os.environ.get("BACKEND_IP", "localhost")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 8000))
BACKEND_URL = f"http://{BACKEND_IP}:{BACKEND_PORT}"

# Initialize session state
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []
if 'current_processing' not in st.session_state:
    st.session_state.current_processing = None

def upload_image_to_backend(file):
    """Upload image to FastAPI backend with better error handling"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{BACKEND_URL}/upload_image/", files=files, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            return False, error_detail
    except requests.exceptions.Timeout:
        return False, "Upload timeout - please try again"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to server - please check if backend is running"
    except Exception as e:
        return False, f"Upload error: {str(e)}"

def get_image_from_backend(object_name, timeout=30):
    """Get processed image from backend with retries"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/get_image/{object_name}", 
            params={"timeout": timeout},
            timeout=timeout + 5
        )
        
        if response.status_code == 200:
            return response.content
        elif response.status_code == 404:
            return None
        else:
            st.error(f"Error fetching {object_name}: Status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching {object_name}: {str(e)}")
        return None

# def check_backend_health():
#     """Check if backend is accessible"""
#     try:
#         response = requests.get(f"{BACKEND_URL}/health", timeout=5)
#         return response.status_code == 200
#     except:
#         return False

def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌊 River Segmentation AI Platform</h1>
        <p>Advanced water body detection using deep learning</p>
    </div>
    """, unsafe_allow_html=True)

# def render_sidebar():
#     """Render professional sidebar with system info"""
#     with st.sidebar:
#         st.header("🎛️ System Status")
        
#         # Backend health check
#         if check_backend_health():
#             st.success("🟢 Backend Online")
#         else:
#             st.error("🔴 Backend Offline")
#             st.warning("Please start the backend server")
        
#         st.header("⚙️ Settings")
        
#         max_file_size = st.slider(
#             "Max File Size (MB)", 
#             min_value=1, 
#             max_value=50, 
#             value=25,
#             help="Maximum allowed file size for upload"
#         )
        
#         timeout_setting = st.slider(
#             "Processing Timeout (sec)", 
#             min_value=30, 
#             max_value=300, 
#             value=60,
#             help="How long to wait for processing results"
#         )
        
#         st.header("📋 Supported Formats")
#         st.markdown("""
#         - **JPEG** (.jpg, .jpeg)
#         - **PNG** (.png)
#         - **TIFF** (.tiff)
#         - **BMP** (.bmp)
#         """)
        
#         if st.session_state.upload_history:
#             st.header("🗑️ Data Management")
#             if st.button("Clear History", type="secondary"):
#                 st.session_state.upload_history = []
#                 st.rerun()
        
#         return max_file_size, timeout_setting

def render_upload_section(max_file_size):
    """Render professional upload section"""
    st.header("📤 Upload Image")
    
    # Upload area with drag & drop styling
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        accept_multiple_files=False,
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help=f"Maximum file size: {max_file_size}MB"
    )
    
    if uploaded_file:
        # File validation
        file_size_mb = uploaded_file.size / (1024 * 1024)
        is_valid = file_size_mb <= max_file_size
        
        # Check if already uploaded
        already_uploaded = any(
            entry['filename'] == uploaded_file.name and entry['success'] 
            for entry in st.session_state.upload_history
        )
        
        # File info card
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**📁 {uploaded_file.name}**")
                if already_uploaded:
                    st.markdown("🔄 *Previously uploaded*")
            
            with col2:
                st.markdown(f"**{file_size_mb:.2f} MB**")
            
            with col3:
                if not is_valid:
                    st.error("❌ Too large")
                elif already_uploaded:
                    st.warning("⚠️ Already Uploaded")
                else:
                    st.success("✅ Ready")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if is_valid and not already_uploaded:
                if st.button("🚀 Process Image", type="primary", use_container_width=True):
                    process_image(uploaded_file)
            elif already_uploaded:
                if st.button("🔄 Process Again", type="secondary", use_container_width=True):
                    process_image(uploaded_file)
            else:
                st.button("❌ File Too Large", disabled=True, use_container_width=True)
        
        with col2:
            if uploaded_file:
                # Preview image
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Preview", use_container_width=True)
                except Exception as e:
                    st.error(f"Cannot preview: {e}")

def process_image(file):
    """Process image with professional progress tracking"""
    st.session_state.current_processing = file.name
    
    # Processing steps
    steps = [
        ("📤 Uploading to server", 0.2),
        ("🤖 AI model processing", 0.6),
        ("🎨 Generating visualizations", 0.8),
        ("✅ Finalizing results", 1.0)
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Upload
        status_text.info("📤 Uploading image to server...")
        file.seek(0)
        success, result = upload_image_to_backend(file)
        progress_bar.progress(0.2)
        
        if success:
            # Add to history
            st.session_state.upload_history.append({
                'filename': file.name,
                'success': True,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'size': f"{file.size / (1024*1024):.2f} MB",
                'file_size': file.size,
                'error': None
            })
            
            # Simulate processing steps
            for step_name, progress in steps[1:]:
                status_text.info(step_name)
                time.sleep(0.5)  # Simulate processing time
                progress_bar.progress(progress)
            
            status_text.success("🎉 Processing completed successfully!")
            st.balloons()
            
        else:
            # Add failed upload to history
            st.session_state.upload_history.append({
                'filename': file.name,
                'success': False,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'size': f"{file.size / (1024*1024):.2f} MB",
                'file_size': file.size,
                'error': str(result)
            })
            status_text.error(f"❌ Upload failed: {result}")
    
    except Exception as e:
        status_text.error(f"❌ Processing error: {str(e)}")
    
    finally:
        st.session_state.current_processing = None
        time.sleep(1)
        st.rerun()

def render_history_section():
    """Render professional upload history"""
    st.header("📊 Processing History")
    
    if not st.session_state.upload_history:
        st.info("No uploads yet. Upload an image to get started!")
        return
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_uploads = len(st.session_state.upload_history)
    successful_uploads = sum(1 for entry in st.session_state.upload_history if entry['success'])
    failed_uploads = total_uploads - successful_uploads
    total_size = sum(entry.get('file_size', 0) for entry in st.session_state.upload_history)
    
    with col1:
        st.metric("Total Uploads", total_uploads)
    with col2:
        st.metric("Successful", successful_uploads, delta=f"{successful_uploads/total_uploads*100:.0f}%")
    with col3:
        st.metric("Failed", failed_uploads)
    with col4:
        st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
    
    # Recent uploads
    st.subheader("Recent Uploads")

    
    for entry in reversed(st.session_state.upload_history[-5:]):  # Last 5 entries
        status_class = "status-success" if entry['success'] else "status-error"
        status_icon = "✅" if entry['success'] else "❌"
        
        st.markdown(f"""
        <div class="status-card {status_class}">
            <strong>{status_icon} {entry['filename']}</strong><br>
            <small>{entry['timestamp']} • {entry['size']}</small>
            {f"<br><em>Error: {entry['error']}</em>" if not entry['success'] else ""}
        </div>
        """, unsafe_allow_html=True)



def render_results_section(timeout_setting = 30 ):
    """Render professional results section"""
    if not st.session_state.upload_history:
        return
    
    successful_uploads = [entry for entry in st.session_state.upload_history if entry['success']]
    if not successful_uploads:
        return
    
    st.header("🔍 Analysis Results")
    
    # Get latest successful upload
    latest = successful_uploads[-1]
    
    st.markdown(f"""
    <div class="processing-steps">
        <h4>📋 Processing Pipeline</h4>
        <p><strong>Image:</strong> {latest['filename']}</p>
        <p><strong>Size:</strong> {latest['size']}</p>
        <p><strong>Processed:</strong> {latest['timestamp']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Overlay Visualization")
        st.info("💡 Green areas indicate detected water bodies")
        
        with st.spinner("Loading overlay..."):
            overlay_bytes = get_image_from_backend(f'overlay/{latest["filename"]}', timeout_setting)
        
        if overlay_bytes:
            try:
                image = Image.open(io.BytesIO(overlay_bytes))
                st.image(image, caption=f"Water Detection Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Display error: {e}")
        else:
            st.warning("⏳ Overlay not ready yet - processing may still be in progress")
    
    with col2:
        st.subheader("🎯 Binary Mask")
        st.info("💡 White = Water, Black = Land")
        
        with st.spinner("Loading prediction mask..."):
            mask_bytes = get_image_from_backend(f'predictions/{latest["filename"]}', timeout_setting)
        
        if mask_bytes:
            try:
                image = Image.open(io.BytesIO(mask_bytes))
                st.image(image, caption=f"Water Detection Mask", use_container_width=True)
            except Exception as e:
                st.error(f"Display error: {e}")
        else:
            st.warning("⏳ Mask not ready yet - processing may still be in progress")

def main():
    """Main application"""
    render_header()
    
    # Sidebar
    #max_file_size, timeout_setting = render_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_upload_section(50)
        # render_results_section()
    
    with col2:
        render_history_section()
    
        # Full-width analysis results section
    render_results_section()

if __name__ == "__main__":
    main()