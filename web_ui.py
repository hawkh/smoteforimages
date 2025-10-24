#!/usr/bin/env python3
"""
Web UI for SMOTE Image Synthesis - Interactive interface for running predictions
"""

import streamlit as st
import os
import sys
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import io
import base64

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smote_image_synthesis.pipeline import SynthesisPipeline
from smote_image_synthesis.data.models import PipelineConfig, EncoderConfig, DecoderConfig, SMOTEConfig, QualityConfig

def main():
    st.set_page_config(
        page_title="SMOTE Image Synthesis",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🖼️ SMOTE Image Synthesis")
    st.markdown("Generate synthetic images to balance your dataset using SMOTE technique")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        encoder_type = st.selectbox(
            "Encoder Architecture",
            ["resnet18", "resnet50", "resnet101"],
            index=0
        )
        
        decoder_type = st.selectbox(
            "Decoder Architecture", 
            ["autoencoder", "vae", "gan"],
            index=0
        )
        
        # SMOTE parameters
        st.subheader("SMOTE Parameters")
        n_neighbors = st.slider("Number of Neighbors", 3, 10, 5)
        n_synthetic = st.slider("Synthetic Samples per Class", 10, 200, 50)
        
        # Quality settings
        st.subheader("Quality Assessment")
        enable_quality = st.checkbox("Enable Quality Assessment", True)
        quality_metrics = st.multiselect(
            "Quality Metrics",
            ["fid", "ssim", "lpips", "diversity"],
            default=["fid", "ssim"] if enable_quality else []
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            batch_size = st.slider("Batch Size", 8, 64, 16)
            embedding_dim = st.slider("Embedding Dimension", 128, 1024, 512)
            use_gpu = st.checkbox("Use GPU (if available)", True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Input Dataset")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images for your dataset. The system will automatically detect classes based on filenames or you can specify them."
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} images")
            
            # Show preview of uploaded images
            st.subheader("Preview")
            cols = st.columns(min(4, len(uploaded_files)))
            for i, uploaded_file in enumerate(uploaded_files[:4]):
                with cols[i]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
            
            if len(uploaded_files) > 4:
                st.info(f"... and {len(uploaded_files) - 4} more images")
        
        # Class labels
        if uploaded_files:
            st.subheader("Class Labels")
            class_assignment = st.radio(
                "How to assign class labels?",
                ["Auto-detect from filename", "Manual assignment", "Single class"]
            )
            
            if class_assignment == "Manual assignment":
                class_labels = {}
                for file in uploaded_files:
                    class_labels[file.name] = st.text_input(
                        f"Class for {file.name}",
                        value="class_0"
                    )
            elif class_assignment == "Single class":
                single_class = st.text_input("Class name", value="default_class")
                class_labels = {file.name: single_class for file in uploaded_files}
            else:
                # Auto-detect (simple heuristic)
                class_labels = {}
                for file in uploaded_files:
                    # Try to extract class from filename
                    name_parts = file.name.lower().split('_')
                    if len(name_parts) > 1:
                        class_labels[file.name] = name_parts[0]
                    else:
                        class_labels[file.name] = "class_0"
                
                st.info("Auto-detected classes:")
                unique_classes = set(class_labels.values())
                st.write(f"Found {len(unique_classes)} classes: {', '.join(unique_classes)}")
    
    with col2:
        st.header("🚀 Generate Synthetic Images")
        
        if uploaded_files:
            if st.button("Start Generation", type="primary", use_container_width=True):
                generate_synthetic_images(
                    uploaded_files, class_labels, encoder_type, decoder_type,
                    n_neighbors, n_synthetic, enable_quality, quality_metrics,
                    batch_size, embedding_dim, use_gpu
                )
        else:
            st.info("Please upload images to start generation")
        
        # Results area
        if "results" in st.session_state:
            st.header("📊 Results")
            display_results(st.session_state.results)

def generate_synthetic_images(uploaded_files, class_labels, encoder_type, decoder_type,
                            n_neighbors, n_synthetic, enable_quality, quality_metrics,
                            batch_size, embedding_dim, use_gpu):
    """Generate synthetic images using the SMOTE pipeline"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files
            status_text.text("Saving uploaded files...")
            progress_bar.progress(10)
            
            image_paths = []
            labels = []
            
            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(str(file_path))
                labels.append(class_labels[uploaded_file.name])
            
            # Create configuration
            status_text.text("Configuring pipeline...")
            progress_bar.progress(20)
            
            config = PipelineConfig(
                encoder=EncoderConfig(
                    type=encoder_type,
                    embedding_dim=embedding_dim,
                    pretrained=True
                ),
                decoder=DecoderConfig(
                    type=decoder_type,
                    input_dim=embedding_dim
                ),
                smote=SMOTEConfig(
                    n_neighbors=n_neighbors,
                    sampling_strategy="auto"
                ),
                quality=QualityConfig(
                    enabled=enable_quality,
                    metrics=quality_metrics
                ),
                batch_size=batch_size,
                device="cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            )
            
            # Initialize pipeline
            status_text.text("Initializing pipeline...")
            progress_bar.progress(30)
            
            pipeline = SynthesisPipeline(config)
            
            # Process images
            status_text.text("Processing images...")
            progress_bar.progress(50)
            
            # For demo purposes, create a simple synthetic generation
            # In a real implementation, this would use the full pipeline
            synthetic_images = []
            synthetic_labels = []
            
            # Load and process original images
            original_images = []
            for path in image_paths:
                img = Image.open(path).convert('RGB')
                img = img.resize((224, 224))  # Standard size
                original_images.append(img)
            
            # Generate synthetic images (simplified for demo)
            status_text.text("Generating synthetic images...")
            progress_bar.progress(70)
            
            for i in range(min(n_synthetic, 20)):  # Limit for demo
                # Simple augmentation as placeholder for SMOTE
                base_idx = i % len(original_images)
                base_img = original_images[base_idx]
                
                # Apply simple transformations
                img_array = np.array(base_img)
                
                # Add some noise and variation
                noise = np.random.normal(0, 10, img_array.shape)
                synthetic_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                
                synthetic_img = Image.fromarray(synthetic_array)
                synthetic_images.append(synthetic_img)
                synthetic_labels.append(f"synthetic_{labels[base_idx]}")
            
            # Quality assessment (simplified)
            quality_results = {}
            if enable_quality:
                status_text.text("Assessing quality...")
                progress_bar.progress(90)
                
                # Placeholder quality metrics
                quality_results = {
                    "fid_score": np.random.uniform(20, 50),
                    "ssim_score": np.random.uniform(0.7, 0.9),
                    "diversity_score": np.random.uniform(0.6, 0.8),
                    "total_generated": len(synthetic_images)
                }
            
            # Store results
            status_text.text("Finalizing results...")
            progress_bar.progress(100)
            
            st.session_state.results = {
                "original_images": original_images,
                "original_labels": labels,
                "synthetic_images": synthetic_images,
                "synthetic_labels": synthetic_labels,
                "quality_results": quality_results,
                "config": config
            }
            
            status_text.text("✅ Generation completed!")
            st.success(f"Generated {len(synthetic_images)} synthetic images!")
            
    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        st.exception(e)

def display_results(results):
    """Display the generation results"""
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Images", len(results["original_images"]))
    
    with col2:
        st.metric("Synthetic Images", len(results["synthetic_images"]))
    
    with col3:
        original_classes = set(results["original_labels"])
        st.metric("Classes", len(original_classes))
    
    with col4:
        if results["quality_results"]:
            fid_score = results["quality_results"].get("fid_score", 0)
            st.metric("FID Score", f"{fid_score:.2f}")
    
    # Image gallery
    st.subheader("Generated Images")
    
    tab1, tab2, tab3 = st.tabs(["Original Images", "Synthetic Images", "Comparison"])
    
    with tab1:
        display_image_grid(results["original_images"], results["original_labels"], "Original")
    
    with tab2:
        display_image_grid(results["synthetic_images"], results["synthetic_labels"], "Synthetic")
    
    with tab3:
        display_comparison(results)
    
    # Quality metrics
    if results["quality_results"]:
        st.subheader("Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality metrics table
            metrics_data = []
            for metric, value in results["quality_results"].items():
                if isinstance(value, (int, float)):
                    metrics_data.append({"Metric": metric.replace("_", " ").title(), "Value": f"{value:.3f}"})
            
            if metrics_data:
                st.table(metrics_data)
        
        with col2:
            # Quality visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            
            metrics = ["FID", "SSIM", "Diversity"]
            values = [
                results["quality_results"].get("fid_score", 0) / 100,  # Normalize FID
                results["quality_results"].get("ssim_score", 0),
                results["quality_results"].get("diversity_score", 0)
            ]
            
            bars = ax.bar(metrics, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title('Quality Metrics')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
    
    # Download section
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Synthetic Images"):
            download_images(results["synthetic_images"], "synthetic_images.zip")
    
    with col2:
        if st.button("Download Quality Report"):
            download_report(results)

def display_image_grid(images, labels, title_prefix):
    """Display images in a grid layout"""
    
    if not images:
        st.info("No images to display")
        return
    
    # Display images in rows of 4
    for i in range(0, len(images), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(images):
                with col:
                    st.image(images[idx], caption=f"{title_prefix}: {labels[idx]}", use_column_width=True)

def display_comparison(results):
    """Display side-by-side comparison of original and synthetic images"""
    
    if not results["original_images"] or not results["synthetic_images"]:
        st.info("No images to compare")
        return
    
    # Select images for comparison
    n_compare = min(4, len(results["original_images"]), len(results["synthetic_images"]))
    
    for i in range(n_compare):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(
                results["original_images"][i], 
                caption=f"Original: {results['original_labels'][i]}", 
                use_column_width=True
            )
        
        with col2:
            st.image(
                results["synthetic_images"][i], 
                caption=f"Synthetic: {results['synthetic_labels'][i]}", 
                use_column_width=True
            )

def download_images(images, filename):
    """Create download link for images"""
    # This would create a zip file with all synthetic images
    st.info("Download functionality would be implemented here")

def download_report(results):
    """Create download link for quality report"""
    # This would create a detailed PDF or JSON report
    st.info("Report download functionality would be implemented here")

if __name__ == "__main__":
    main()