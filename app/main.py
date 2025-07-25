import streamlit as st
import json
import base64
import os
from datetime import datetime
import tempfile
from PIL import Image
import io
import sys

sys.path.append(os.path.dirname('../kafka'))

from kafka.producer import create_kafka_producer, delivery_callback

# Configure page
st.set_page_config(
    page_title="River Segmentation Image Upload",
    page_icon="🌊",
    layout="wide"
)

# Initialize session state
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []

def create_message(image_file, filename):
    """Create Kafka message from uploaded image file"""
    # Read image data
    image_data = image_file.read()
    image_file.seek(0)  # Reset file pointer
    
    # Encode to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Create message
    message = {
        'filename': filename,
        'image': image_base64,
        'date': datetime.now().isoformat(),
        'file_size': len(image_data),
        'upload_source': 'streamlit_app'
    }
    
    return message

def send_to_kafka(producer, message, topic_name, message_id):
    """Send message to Kafka and return success status"""
    try:
        producer.produce(
            topic_name,
            value=json.dumps(message),
            key=str(message_id),
            callback=delivery_callback
        )
        producer.poll(0)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    st.title("🌊 River Segmentation Image Upload")
    st.markdown("Upload images to be processed by the river segmentation pipeline")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
        
    # Kafka settings
    bootstrap_server = st.sidebar.text_input(
        "Kafka Bootstrap Server", 
        value="localhost:29092",
        help="Kafka broker address"
    )
    
    topic_name = st.sidebar.text_input(
        "Kafka Topic", 
        value="River",
        help="Topic to send images to"
    )
    
    # Upload settings
    max_file_size = st.sidebar.slider(
        "Max File Size (MB)", 
        min_value=1, 
        max_value=100, 
        value=10
    )
    
    # Main upload area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Images")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
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
                if st.button("🚀 Send to Kafka", type="primary", use_container_width=True):
                    upload_images(valid_files, bootstrap_server, topic_name)
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

def upload_images(files, bootstrap_server, topic_name):
    """Upload images to Kafka with progress tracking"""
    
    try:
        # Create Kafka producer
        with st.spinner("Connecting to Kafka..."):
            producer = create_kafka_producer(
                bootstrap_server=bootstrap_server,
                acks='all',
                compression_type='snappy'
            )
        
        st.success("✅ Connected to Kafka")
        
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
                # Create message
                message = create_message(file, file.name)
                
                # Send to Kafka
                success, error = send_to_kafka(producer, message, topic_name, i)
                
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
                        'error': error
                    })
                    
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        # Ensure all messages are sent
        producer.flush()
        
        # Final status
        progress_bar.progress(1.0)
        status_text.text("Upload completed!")
        
        if success_count == len(files):
            st.success(f"🎉 Successfully uploaded all {success_count} images!")
        else:
            st.warning(f"⚠️ Uploaded {success_count}/{len(files)} images successfully")
        
        # Auto-refresh to show updated history
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Kafka: {str(e)}")
        st.info("Make sure Kafka is running and the bootstrap server address is correct")

if __name__ == "__main__":
    main()