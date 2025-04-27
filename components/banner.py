import streamlit as st


# Function to create a new banner
def create_banner():
    st.session_state["banner_key"] += 1  # Increment the key to create a new div


# Initialize session state for the banner key
if "banner_key" not in st.session_state:
    st.session_state["banner_key"] = 0

# Button to trigger the banner
st.button("Show Banner", on_click=create_banner)

# Custom CSS for the banner with animation
st.markdown(
    """
    <style>
    @keyframes fadeOut {
        0% { opacity: 1; }
        90% { opacity: 0.1; }
        100% { opacity: 0; display: none; }
    }

    .banner-alert {
        background-color: #ffcccb;
        color: #000;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        animation: fadeOut 3s forwards; /* 3-second fade-out animation */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a new banner div for each key
for i in range(st.session_state["banner_key"]):
    st.markdown(
        f'<div class="banner-alert" key="{i}">🚨 This is a banner alert! 🚨</div>',
        unsafe_allow_html=True,
    )

# Add other Streamlit components
st.write("This is the rest of your Streamlit app.")
