import streamlit as st
import requests
import json
import time
FASTAPI_URL = "http://localhost:8000"

st.title("📹 Video Surveillance System")

st.markdown("""
Upload one or more video files to be analyzed for safety violations.
The analysis will run in the background on a separate server.
""")
uploaded_files = st.file_uploader(
    "Choose video files",
    type=['mp4', 'avi', 'mov', 'mkv'],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Start Analysis"):
        with st.spinner("Uploading and starting analysis..."):
            files = [('files', (file.name, file, file.type)) for file in uploaded_files]

            try:
                response = requests.post(f"{FASTAPI_URL}/analyze_videos/", files=files)
                response.raise_for_status()

                data = response.json()
                request_id = data.get("request_id")

                st.success(f"Analysis started successfully! Request ID: {request_id}")
                st.info("Waiting for alerts... This page will update automatically.")
                placeholder = st.empty()
                while True:
                    time.sleep(2)
                    results_response = requests.get(f"{FASTAPI_URL}/results/{request_id}")
                    results_data = results_response.json()
                    
                    if results_data.get("alerts"):
                        with placeholder.container():
                            st.subheader("🚨 New Alerts")
                            for alert in results_data["alerts"]:
                                st.write(json.dumps(alert, indent=2))
                                st.warning("---")
                    if 'message' in results_data and 'Invalid or expired request ID.' in results_data['message']:
                        st.success("Analysis complete.")
                        break

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while communicating with the backend: {e}")
            except json.JSONDecodeError:
                st.error("Failed to decode JSON response from the server.")
st.sidebar.subheader("Backend Status")
try:
    requests.get(FASTAPI_URL, timeout=1)
    st.sidebar.success("Backend is running.")
except requests.exceptions.RequestException:
    st.sidebar.error("Backend is not reachable.")