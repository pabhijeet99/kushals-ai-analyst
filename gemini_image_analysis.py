import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import json
import base64
from urllib.parse import quote_plus

# --- CONFIGURATION & CONSTANTS ---

# App metadata
APP_PAGE_TITLE = "Kushal's AI Fashion Jewellery Analyst"
APP_PAGE_ICON = "💍"

# Logo URL (For Kushal's logo in the sidebar)
KUSHALS_LOGO_URL = "https://us.kushals.com/cdn/shop/files/Kushals_New_Logo.svg?v=1762258850&width=310"

# DIVERSE_STYLES for Section 5 (Lifestyle Images)
DIVERSE_STYLES = {
    "Classic Traditional": "A lavish, indoor setting with rich Indian fabrics (silk, brocade) and warm, ambient lighting, suitable for a wedding or festive occasion. Full-length portrait, sharp focus.",
    "Modern Casual": "An elegant, contemporary café or outdoor patio setting. The model is wearing fusion or modern Western attire. Bright, soft natural light. Mid-torso crop, slightly candid look.",
    "Corporate/Party Glam": "A sophisticated nighttime look, such as a sleek office party or a cocktail event. Dark, dramatic background with focused spot lighting to highlight the jewelry. Close-up on the model's face and jewelry.",
    "High-Key Studio Shot": "Pure, stark white background. Focus entirely on the product and the model's area wearing it. Crisp, high-key studio lighting. Close crop."
}

# Platform-specific image generation requirements (Prompts **Reinforced for Accurate Ring Sizing**)
PLATFORM_STANDARDS = {
    "Amazon": {
        "Resolution": "1000px minimum on the longest side (ideally 2000px for zoom). Square 1:1 aspect ratio.",
        "Prompt": "A highly professional, isolated product image of the attached jewelry on a pure, seamless, high-key white background (RGB 255, 255, 255). The jewelry must occupy 85% of the frame. Use crisp studio lighting. **CRITICAL: If the product is a ring, it must be worn realistically on a model's hand with a natural, accurately proportioned fit, avoiding oversized appearance.** The image must be square.",
        "Mime_Type": "image/jpeg"
    },
    "Myntra": {
        "Resolution": "Min 1500x2000px. Aspect ratio 3:4 (Portrait preferred).",
        "Prompt": "A high-quality, professional image showing the attached jewelry on an attractive Indian model in a clean, high-fashion, subtle lifestyle context. **CRITICAL: If the product is a ring, the model must be wearing it with a realistic, comfortable, and accurately sized fit.** The image should be portrait (3:4 aspect ratio) with soft, natural lighting, and NO visible brand logos or watermarks.",
        "Mime_Type": "image/jpeg"
    },
    "Flipkart": {
        "Resolution": "Min 500x500px (ideally 1000x1000px). Square 1:1 aspect ratio.",
        "Prompt": "A sharp, clear image of the attached jewelry on a pure, solid white or light-grey background. Must be product-only or on a mannequin/bust or a hand model. **CRITICAL: If the product is a ring, ensure realistic fit and accurate scale on the finger model.** No text, logos, or props. Use high-key studio lighting. The image must be square.",
        "Mime_Type": "image/jpeg"
    }
}

# Gemini API Schema for Structured Output (Unchanged)
COMPLIANCE_SUB_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={'Feedback': types.Schema(type=types.Type.STRING, description='A single, 2-3 line sentence advising the necessary action.')},
    required=['Feedback']
)

OUTPUT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        'Attitude': types.Schema(type=types.Type.STRING, description='A short, descriptive phrase.'),
        'SEO_Keywords': types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description='10 relevant keywords and long-tail phrases.'),
        'Product_Description': types.Schema(type=types.Type.STRING, description='A 3-sentence description optimized for online listings.'),
        'Inventory_Tags': types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description='5 key classification tags/categories.'),
        'Compliance_Feedback': types.Schema(
            type=types.Type.OBJECT,
            properties={'Amazon': COMPLIANCE_SUB_SCHEMA, 'Myntra': COMPLIANCE_SUB_SCHEMA, 'Flipkart': COMPLIANCE_SUB_SCHEMA},
            required=['Amazon', 'Myntra', 'Flipkart']
        ),
    },
    required=['Attitude', 'SEO_Keywords', 'Product_Description', 'Inventory_Tags', 'Compliance_Feedback']
)

# --- UTILITY FUNCTIONS ---

def set_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(
        page_title=APP_PAGE_TITLE,
        layout="wide",
        page_icon=APP_PAGE_ICON
    )

def apply_custom_css():
    """Applies custom CSS for better layout control."""
    st.markdown(
        """
        <style>
        /* Reduce default Streamlit padding at the top */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Style for resolution text */
        .resolution-info {
            font-size: 0.85em;
            color: #6c757d; 
            margin-top: -10px; 
            margin-bottom: 5px;
        }
        
        /* Ensure button text does not wrap onto two lines */
        div.stButton > button {
            white-space: nowrap; 
        }

        /* Remove default margins for the H4 to make section 4 cleaner */
        h4 {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        /* Adjust the sidebar width */
        section[data-testid="stSidebar"] {
            width: 250px !important; 
        }
        </style>
        """, unsafe_allow_html=True)

def toggle_image_display():
    """Toggles the visibility of the uploaded image in session state."""
    st.session_state.show_uploaded_image = not st.session_state.show_uploaded_image

def get_image_download_link(img, filename, text):
    """Generates an HTML link to download a PIL Image as a JPEG."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    safe_filename = quote_plus(filename)
    return f"""
    <a href="data:image/jpeg;base64,{img_str}" download="{safe_filename}">
        <button style="
            background-color: #f0f2f6; 
            color: #31333f; 
            border: 1px solid #d3d3d3; 
            padding: 0.25rem 0.5rem; 
            border-radius: 0.25rem; 
            cursor: pointer;
            margin-top: 10px;
            width: 100%;
            text-align: center;
        ">{text}</button>
    </a>
    """

def get_dummy_analysis():
    """Provides a fixed, placeholder structured output."""
    return {
        'Attitude': 'Placeholder - Traditional Festive Wear',
        'SEO_Keywords': [
            'placeholder gold plated necklace',
            'placeholder kundan jewelry',
            'placeholder wedding set',
            'placeholder festive jewellery',
            'placeholder traditional indian earrings'
        ],
        'Product_Description': 'This is a **placeholder description** because the Gemini API key was not successfully initialized. Please enter a valid key in the sidebar to get a real, AI-generated product analysis based on your uploaded image.',
        'Inventory_Tags': ['Placeholder', 'Jewelry Set', 'Necklace', 'Brass Base', 'Kundan'],
        'Compliance_Feedback': {
            'Amazon': {'Feedback': 'Placeholder: Ensure the image is a professional studio shot on a pure white background, with the jewellery occupying at least 85% of the frame. If the background is not white, please edit it.'},
            'Myntra': {'Feedback': 'Placeholder: Myntra typically prefers a subtle lifestyle context. Verify that no visible text or logos appear on the image.'},
            'Flipkart': {'Feedback': 'Placeholder: Flipkart main images must be free of models, text, or props. Zoom in to show the product clearly without any extra elements.'},
        },
    }

# --- CORE GEMINI API FUNCTIONS ---

@st.cache_resource(show_spinner=False)
def initialize_client(api_key):
    """Initializes and returns the Gemini client."""
    if api_key:
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing client: {e}")
            return None
    return None

def get_product_analysis(client, uploaded_file):
    """Calls Gemini to analyze the image and return structured content."""
    try:
        image_part = types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=uploaded_file.type)

        analysis_prompt = """
        You are an expert E-commerce Analyst specializing in Indian Fashion Jewellery. Analyze the attached product image.
        Provide the analysis by filling the required keys in the exact JSON response schema provided.
        For the Compliance_Feedback, provide a single, concise sentence (2-3 lines max) for each platform summarizing its compliance status and the one key action needed (if any) for the *uploaded product image* itself.
        """

        with st.spinner("Analyzing image and generating structured content..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image_part, analysis_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=OUTPUT_SCHEMA),
            )

        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text.lstrip("```json").rstrip("```").strip()
        json_response = json.loads(json_text)
        return json_response

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during API call: {e}")
        return None

def generate_platform_images(client, uploaded_file, description):
    """Generates images compliant with e-commerce platform standards."""
    platform_images_map = {}
    image_part = types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=uploaded_file.type)
    platforms_to_generate = PLATFORM_STANDARDS 

    progress_text_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i, (platform, standard) in enumerate(platforms_to_generate.items()):
        try:
            # Use the reinforced prompt for better image realism
            image_prompt = f"""
            Generate a high-quality, professional e-commerce product image for the platform {platform}.
            The attached jewelry product should be the focus.
            **Platform Requirement:** {standard['Prompt']}
            The product is generally described as: '{description}'.
            """

            progress_text = f"Generating compliant image for **{platform}** ({i + 1}/{len(platforms_to_generate)})..."
            progress_text_placeholder.markdown(progress_text)
            progress_bar.progress((i + 1) / len(platforms_to_generate))

            response = client.models.generate_content(
                model='gemini-2.5-flash-image',
                contents=[image_part, image_prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )

            for part in response.parts:
                if part.inline_data:
                    platform_images_map[platform] = Image.open(BytesIO(part.inline_data.data))
                    break

        except Exception as e:
            if "429 RESOURCE_EXHAUSTED" in str(e):
                st.error("🚨 QUOTA EXCEEDED ERROR 🚨: Image generation quota limit reached.")
                progress_bar.empty()
                progress_text_placeholder.empty()
                return platform_images_map
            else:
                st.error(f"Error generating image for {platform}: {e}")

    progress_bar.empty()
    progress_text_placeholder.empty()
    return platform_images_map

def generate_lifestyle_images(client, uploaded_file, description, styles):
    """Generates diverse lifestyle images using different styles."""
    generated_images_map = {}
    image_part = types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=uploaded_file.type)
    
    progress_text_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i, (style_key, style_description) in enumerate(styles.items()):
        try:
            # Use the reinforced prompt for better image realism
            image_prompt = f"""
            Generate a high-quality, professional, photorealistic e-commerce lifestyle image for Kushal's Fashion Jewellery.
            **Industrial Standards:** The image must be square (e.g., 1024x1024), sharp, in-focus, and have professional lighting. **CRITICAL: If the product is a ring, ensure it is worn realistically on the model's finger with accurate sizing and fit.**
            **Instructions:** 1. **Integrate the attached jewelry product** onto an attractive model suitable for high-end Indian fashion. 2. The scene should strictly follow the **Style/Context:** '{style_description}'. 3. The product is generally described as: '{description}'.
            Output only the final image.
            """
            progress_text = f"Generating style **{i + 1}/{len(styles)}**: {style_key}..."
            progress_text_placeholder.markdown(progress_text)
            progress_bar.progress((i + 1) / len(styles))

            response = client.models.generate_content(
                model='gemini-2.5-flash-image',
                contents=[image_part, image_prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )

            for part in response.parts:
                if part.inline_data:
                    generated_images_map[style_key] = Image.open(BytesIO(part.inline_data.data))
                    break

        except Exception as e:
            if "429 RESOURCE_EXHAUSTED" in str(e):
                st.error("🚨 QUOTA EXCEEDED ERROR 🚨: You have run out of the free quota for image generation.")
                progress_bar.empty()
                progress_text_placeholder.empty()
                return generated_images_map
            else:
                st.error(f"An error occurred during image generation for style '{style_key}': {e}")

    progress_bar.empty()
    progress_text_placeholder.empty()
    return generated_images_map

# --- INITIALIZATION AND UI SETUP ---

# 1. Set Page Config and Apply CSS
set_page_config()
apply_custom_css()

# 2. Initialize session state (Unchanged)
if 'show_uploaded_image' not in st.session_state:
    st.session_state.show_uploaded_image = False
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Real"
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'description' not in st.session_state:
    st.session_state.description = None
if 'generated_images_map' not in st.session_state:
    st.session_state.generated_images_map = {}
if 'platform_images_map' not in st.session_state:
    st.session_state.platform_images_map = {}


# 3. Sidebar for API Key Input & Client Initialization
client = None
with st.sidebar:
    # Kushal's Logo
    st.image(KUSHALS_LOGO_URL, width=150)
    
    # Configuration Header (Moved below logo as requested)
    st.markdown("## ⚙️ Configuration")
    
    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Get your key from Google AI Studio"
    )

    client = initialize_client(api_key)

    if client:
        st.success("API Client Initialized!")
    else:
        st.warning("Please enter your API Key for full functionality (especially image generation).")


# 4. Main Content Header
st.title(APP_PAGE_TITLE)
st.subheader("Analyze an image, generate marketing/SEO content, and AI lifestyle images using Gemini")

# --- MAIN APPLICATION LOGIC ---

uploaded_file = st.file_uploader(
    "Upload a Product Image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    uploaded_file.seek(0)
    
    try:
        image_to_display = Image.open(BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.error(f"Could not open image file: {e}")
        image_to_display = None

    if image_to_display:
        # 1. Image Toggle Button AND Analyze Button 
        col_show, col_analyze = st.columns([2, 5])
        
        with col_show:
            if st.session_state.show_uploaded_image:
                button_label = "🖼️ Hide Uploaded Image"
            else:
                button_label = "🖼️ Show Uploaded Image"
                
            st.button(button_label, on_click=toggle_image_display, help="Click to see the image you uploaded.")

        with col_analyze:
            # 2. Analysis Trigger Button 
            if st.button("🚀 Analyze & Generate Kushal's E-commerce Content", type="primary"):

                st.session_state.generated_images_map = {}
                st.session_state.platform_images_map = {}
                st.session_state.analysis_data = None

                # Run analysis
                if client:
                    st.session_state.analysis_mode = "Real"
                    st.session_state.analysis_data = get_product_analysis(client, uploaded_file)
                else:
                    st.session_state.analysis_mode = "Placeholder"
                    st.warning("Using **placeholder analysis** because the API client is not initialized.")
                    st.session_state.analysis_data = get_dummy_analysis()

                # Proceed only if analysis data is available
                if st.session_state.analysis_data:
                    st.session_state.description = st.session_state.analysis_data.get('Product_Description', 'N/A')

                    # --- Auto-Trigger Image Generation ---
                    if st.session_state.analysis_mode == "Real" and client:
                        
                        # 4. Generate ALL Platform-Specific Images first
                        st.session_state.platform_images_map = generate_platform_images(
                            client,
                            uploaded_file,
                            st.session_state.description
                        )

                        # 5. Generate Lifestyle Images
                        st.session_state.generated_images_map = generate_lifestyle_images(
                            client,
                            uploaded_file,
                            st.session_state.description,
                            DIVERSE_STYLES
                        )
                        st.balloons()
        
        # Conditionally display the uploaded image below the buttons if toggled
        if st.session_state.show_uploaded_image:
            st.image(image_to_display, caption="Uploaded Product", width=250)


        st.markdown("---")


        # 3. Display Results 
        if st.session_state.analysis_data:
            analysis_result = st.session_state.analysis_data

            st.header(f"💍 ✅ Structured E-commerce Content Analysis ({st.session_state.analysis_mode} Data)")

            # 1. Inferred Style
            st.subheader("1. 🎨 Inferred Style & Attitude")
            attitude = analysis_result.get('Attitude', 'N/A')
            st.info(f"**Product Style/Attitude:** `{attitude}`")

            # 2. Product Descriptions and Tags
            st.subheader("2. 📝 Product Description & Inventory Tags")
            description = analysis_result.get('Product_Description', 'N/A')
            st.markdown(f"**Description:** *{description}*")
            tags = analysis_result.get('Inventory_Tags', [])
            st.markdown(f"**Inventory Tags:** **`{', '.join(tags)}`**")

            # 3. SEO Keywords
            st.subheader("3. 🔍 SEO Keywords")
            st.caption("Use these for product listing optimization and search ranking.")
            st.code(", ".join(analysis_result.get('SEO_Keywords', [])))

            # --- MERGED SECTION 4: Image Compliance and Generated Images (Logos Removed) ---
            st.markdown("---")
            st.header("4. 🛡️ E-commerce Platform-Specific Images & Compliance")
            st.caption("AI-generated images tailored to meet platform visual standards, along with compliance feedback for your original upload.")

            compliance = analysis_result.get('Compliance_Feedback', {})
            platform_images_map = st.session_state.platform_images_map
            
            platforms = list(PLATFORM_STANDARDS.keys())
            compliance_cols = st.columns(len(platforms))

            for i, platform in enumerate(platforms):
                with compliance_cols[i]:
                    img = platform_images_map.get(platform)
                    standard = PLATFORM_STANDARDS.get(platform, {})
                    feedback = compliance.get(platform, {}).get('Feedback', 'N/A')

                    # **FIX: Display the platform name as a header (Logos Removed)**
                    st.subheader(platform)
                    
                    if img:
                        # 1. Display Image
                        st.image(img, caption=f"AI-Generated {platform} Image", use_container_width=True)

                        # 2. Display resolution requirement
                        st.markdown(f'<p class="resolution-info">Req. Resolution: {standard.get("Resolution", "N/A")}</p>', unsafe_allow_html=True)

                        # 3. Add Download Button
                        img_filename = f"Kushals_Product_Compliant_{platform}.jpg"
                        st.markdown(
                            get_image_download_link(img, img_filename, f"⬇️ Download {platform} Image (JPG)"),
                            unsafe_allow_html=True
                        )
                    elif st.session_state.analysis_mode == "Real":
                        st.warning(f"Image for {platform} not generated (API issue/quota).")
                        
                    # 4. Display Compliance Feedback for the original upload
                    st.markdown("**Compliance Feedback on Upload:**")
                    st.success(feedback)
                        
            if st.session_state.analysis_mode == "Placeholder":
                st.info("💡 Run a real analysis (with a valid API key) to generate compliant platform images and receive feedback here.")


            # --- SECTION 5: Conditional Image Generation (Display) ---
            st.markdown("---")
            st.header(f"5. 📸 AI Lifestyle Image Generation ({len(DIVERSE_STYLES)} Styles)")

            generated_images_map = st.session_state.generated_images_map

            if client and generated_images_map:
                st.success("Multiple professional lifestyle images generated successfully! Scroll down to download.")

                # Display and Download Logic in two rows of two images
                
                # First row
                cols_row1 = st.columns(2)
                for i, (style, img) in enumerate(list(generated_images_map.items())[:2]):
                    with cols_row1[i]:
                        st.image(img, caption=f"Style: {style}", use_container_width=True)
                        img_filename = f"Kushals_AI_Jewelry_{style.replace(' ', '_')}.jpg"
                        st.markdown(
                            get_image_download_link(img, img_filename, "⬇️ Download Image (JPG)"),
                            unsafe_allow_html=True
                        )
                
                # Second row (if more than two styles exist)
                if len(generated_images_map) > 2:
                    cols_row2 = st.columns(2)
                    for i, (style, img) in enumerate(list(generated_images_map.items())[2:]):
                        with cols_row2[i]:
                            st.image(img, caption=f"Style: {style}", use_container_width=True)
                            img_filename = f"Kushals_AI_Jewelry_{style.replace(' ', '_')}.jpg"
                            st.markdown(
                                get_image_download_link(img, img_filename, "⬇️ Download Image (JPG)"),
                                unsafe_allow_html=True
                            )
            
            elif st.session_state.analysis_mode == "Real" and not client:
                st.warning("Please ensure the Gemini API key is valid to generate general lifestyle images.")
            elif st.session_state.analysis_mode == "Placeholder":
                st.info("💡 Run a real analysis (with a valid API key) to generate general lifestyle images here.")
