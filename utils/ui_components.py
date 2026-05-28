"""
Premium CSS configurations and HTML components for Streamlit UI.
Implements dark modern theme, glassmorphism cards, animated buttons, and responsive grid patterns.
"""

import streamlit as st


def apply_premium_styles() -> None:
    """Injects high-end, responsive custom styling into the Streamlit session."""
    st.markdown(
        """
        <style>
        /* Import premium font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Main application container overrides */
        .stApp {
            background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d111c 100%);
            font-family: 'Outfit', sans-serif;
            color: #f3f4f6;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(17, 24, 39, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Custom Header Gradient */
        .gradient-text {
            background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            letter-spacing: -1px;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .gradient-subtitle {
            font-size: 1.15rem;
            color: #9ca3af;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Glassmorphism Card Wrapper */
        .glass-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            transform: translateY(-4px);
            border-color: rgba(192, 132, 252, 0.35);
            box-shadow: 0 12px 40px 0 rgba(129, 140, 248, 0.15);
        }

        .recommendation-card {
            padding: 1rem;
            border-left: 4px solid rgba(139, 92, 246, 0.95);
            margin-bottom: 1rem;
        }

        .recommendation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            align-items: start;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            background: rgba(139, 92, 246, 0.18);
            color: #ddd6fe;
            border: 1px solid rgba(139, 92, 246, 0.35);
            border-radius: 999px;
            padding: 0.25rem 0.6rem;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .chip-soft {
            background: rgba(244, 114, 182, 0.18);
            border-color: rgba(244, 114, 182, 0.35);
            color: #fbcfe8;
        }

        .mini-section {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 0.75rem;
            margin-bottom: 0.55rem;
        }

        .mini-title {
            color: #c084fc;
            font-size: 0.95rem;
            margin: 0 0 0.35rem 0;
            font-weight: 700;
        }

        /* Metrics grid */
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 1.5rem;
        }

        .metric-tile {
            background: rgba(17, 24, 39, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: border-color 0.2s ease;
        }

        .metric-tile:hover {
            border-color: rgba(244, 114, 182, 0.25);
        }

        .metric-val {
            font-size: 2.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #c084fc 0%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        /* Action Buttons styling */
        div.stButton > button {
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.8rem !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }

        div.stButton > button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        }
        
        /* Secondary stop button overrides */
        .stop-btn div.stButton > button {
            background: linear-gradient(90deg, #ef4444 0%, #b91c1c 100%) !important;
            box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;
        }
        
        .stop-btn div.stButton > button:hover {
            background: linear-gradient(90deg, #dc2626 0%, #991b1b 100%) !important;
            box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6) !important;
        }

        /* Success alerts overrides */
        .stAlert {
            background-color: rgba(16, 185, 129, 0.15) !important;
            border: 1px solid rgba(16, 185, 129, 0.3) !important;
            color: #10b981 !important;
            border-radius: 12px !important;
        }

        /* File uploader styling */
        div[data-testid="stFileUploader"] {
            border: 2px dashed rgba(192, 132, 252, 0.4) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            background: rgba(30, 41, 59, 0.25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title_section(title: str, subtitle: str) -> None:
    """Helper to display the stylized app header."""
    st.markdown(f'<div class="gradient-text">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="gradient-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def draw_card(title: str, body_html: str, border_color_rgba: str = "rgba(255, 255, 255, 0.08)") -> None:
    """Helper to draw a custom glassmorphic layout card."""
    st.markdown(
        f"""
        <div class="glass-card" style="border-color: {border_color_rgba}">
            <h3 style="margin-top:0; color:#c084fc; font-weight:600; letter-spacing:-0.5px;">{title}</h3>
            <div>{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
