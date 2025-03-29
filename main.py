import streamlit as st
import streamlit_extras.switch_page_button as spb


def main():
    st.set_page_config(page_title="Login", page_icon=":lock:")
    st.title("🔐 User Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.success("Login successful! Redirecting...")
        spb.switch_page("major")  # Redirects to `major.py`


if __name__ == "__main__":
    main()
