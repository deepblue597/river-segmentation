import streamlit as st

from datetime import datetime
from PIL import Image
import io

import requests



# Configure page
st.set_page_config(
    page_title="River Segmentation Image Upload",
    page_icon="🌊",
    layout="wide"
)

# Backend API configuration
BACKEND_URL = "http://localhost:8000"  # Update this to your backend URL


# Initialize session state
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []

# if 'upload_images' not in st.session_state:
#     st.session_state.upload_images = [] 

def upload_image_to_backend(file):
    """Upload image to FastAPI backend"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{BACKEND_URL}/upload_image/", files=files)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Upload failed")
    except Exception as e:
        return False, str(e)

def get_image_from_backend(object_name, timeout=30):
    """Get processed image from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/get_image/{object_name}", params={"timeout": timeout})
        
        if response.status_code == 200:
            return response.content
        elif response.status_code == 404:
            print(f"Image not found: {object_name}")
            return None
        else:
            print(f"Error fetching {object_name}: Status {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception fetching {object_name}: {e}")
        return None    


def main():
    st.title("🌊 River Segmentation Image Upload")
    st.markdown("Upload images to be processed by the river segmentation pipeline")
    
    # # Sidebar configuration
    # st.sidebar.header("⚙️ Configuration")
        
    # # Kafka settings
    # bootstrap_server = st.sidebar.text_input(
    #     "Kafka Bootstrap Server", 
    #     value="localhost:29092",
    #     help="Kafka broker address"
    # )
    
    # topic_name = st.sidebar.text_input(
    #     "Kafka Topic", 
    #     value="River",
    #     help="Topic to send images to"
    # )
    
    # # Upload settings
    # max_file_size = st.sidebar.slider(
    #     "Max File Size (MB)", 
    #     min_value=1, 
    #     max_value=100, 
    #     value=10
    # )
    
    
    max_file_size = 50  # Set max file size to 50MB
    # Main upload area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Images")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            accept_multiple_files=False,
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            label_visibility="visible",
            help=f"Maximum file size: {max_file_size}MB per file"
        )
        
        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} files")
            
            # Show file details
            with st.expander("📋 File Details", expanded=True):
                for i, file in enumerate(uploaded_files):
                    file_size_mb = file.size / (1024 * 1024)
                    status = "✅" if file_size_mb <= max_file_size else "❌ Too large"
                    
                    col_name, col_size, col_status = st.columns([3, 1, 1])
                    col_name.write(f"{i+1}. {file.name}")
                    col_size.write(f"{file_size_mb:.2f} MB")
                    col_status.write(status)
            
            # Filter valid files
            valid_files = [
                f for f in uploaded_files 
                if f.size / (1024 * 1024) <= max_file_size
            ]
            
            if valid_files:
                st.info(f"Ready to upload {len(valid_files)} valid files")
                
                # Upload button
                if st.button("🚀 Send to Backend", type="primary", use_container_width=True):
                    upload_images_to_backend(valid_files)
            else:
                st.error("No valid files to upload. Check file sizes.")
            
            
    
    with col2:
        st.header("📊 Upload History")
        
        if st.session_state.upload_history:
            for entry in reversed(st.session_state.upload_history[-10:]):  # Last 10 entries
                with st.container():
                    status_icon = "✅" if entry['success'] else "❌"
                    st.write(f"{status_icon} {entry['filename']}")
                    st.caption(f"{entry['timestamp']} | {entry['size']}")
                    if not entry['success']:
                        st.error(f"Error: {entry['error']}")
                    st.divider()
        else:
            st.info("No uploads yet")
    
    # Statistics
    if st.session_state.upload_history:
        st.header("📈 Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        total_uploads = len(st.session_state.upload_history)
        successful_uploads = sum(1 for entry in st.session_state.upload_history if entry['success'])
        total_size = sum(entry.get('file_size', 0) for entry in st.session_state.upload_history)
        
        col1.metric("Total Uploads", total_uploads)
        col2.metric("Successful", successful_uploads)
        col3.metric("Total Size", f"{total_size / (1024*1024):.2f} MB")
        
        # Show results for last uploaded image
        if successful_uploads > 0:
            st.header("🔍 Latest Results")
            last_successful = None
            for entry in reversed(st.session_state.upload_history):
                if entry['success']:
                    last_successful = entry
                    break
            
            if last_successful:
                col1, col2 = st.columns(2)
                
                with st.spinner("Waiting for model to process image..."):
                    with col1:
                        st.subheader("🎨 Overlay")
                        overlay_bytes = get_image_from_backend(f'overlay/{last_successful["filename"]}')
                        if overlay_bytes:
                            try:
                                image = Image.open(io.BytesIO(overlay_bytes))
                                st.image(image, caption=f"Overlay: {last_successful['filename']}")
                            except Exception as e:
                                st.error(f"Could not display overlay: {e}")
                        else:
                            st.warning("Overlay not ready yet")
                    
                    with col2:
                        st.subheader("🎯 Prediction Mask")
                        mask_bytes = get_image_from_backend(f'predictions/{last_successful["filename"]}')
                        if mask_bytes:
                            try:
                                image = Image.open(io.BytesIO(mask_bytes))
                                st.image(image, caption=f"Mask: {last_successful['filename']}")
                            except Exception as e:
                                st.error(f"Could not display mask: {e}")
                        else:
                            st.warning("Prediction mask not ready yet")

def upload_images_to_backend(files):
    """Upload images to backend with progress tracking"""
    
    # Upload progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    
    for i, file in enumerate(files):
        # Update progress
        progress = (i + 1) / len(files)
        progress_bar.progress(progress)
        status_text.text(f"Uploading {file.name}... ({i+1}/{len(files)})")
        
        try:
            # Reset file pointer
            file.seek(0)
            
            # Upload to backend
            success, result = upload_image_to_backend(file)
            
            if success:
                success_count += 1
                st.session_state.upload_history.append({
                    'filename': file.name,
                    'success': True,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'size': f"{file.size / (1024*1024):.2f} MB",
                    'file_size': file.size,
                    'error': None
                })
            else:
                st.session_state.upload_history.append({
                    'filename': file.name,
                    'success': False,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'size': f"{file.size / (1024*1024):.2f} MB",
                    'file_size': file.size,
                    'error': str(result)
                })
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    # Final status
    progress_bar.progress(1.0)
    status_text.text("Upload completed!")
    
    if success_count == len(files):
        st.success(f"🎉 Successfully uploaded all {success_count} images!")
    else:
        st.warning(f"⚠️ Uploaded {success_count}/{len(files)} images successfully")
    
    # Auto-refresh to show updated history
    st.rerun()

if __name__ == "__main__":
    main()