import streamlit as st
import tempfile
import os
import sys
import subprocess

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from inference.predict import predict_video


st.set_page_config(
    page_title="LipRead AI",
    layout="centered"
)


st.title("LipRead AI")
st.caption("Sentence-Level Lip Reading System")

st.divider()


uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mpg", "avi", "mov"]
)


if uploaded_file is not None:

    original_suffix = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    original_path = None
    preview_path = None

    try:

        # Save original uploaded video
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=original_suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            original_path = temp_file.name


        # ----------------------------------------------------
        # Convert video to MP4 for browser playback
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            preview_path = temp_file.name


        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                original_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                preview_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )


        # ----------------------------------------------------
        # Browser preview
        # ----------------------------------------------------

        st.subheader("Video Preview")

        with open(preview_path, "rb") as video_file:
            st.video(video_file.read())


        st.divider()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if st.button(
            "Predict Sentence",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Processing video..."):

                prediction, confidence = predict_video(
                    original_path
                )


            st.subheader("Prediction")

            st.markdown(
                f"""
                <div style="
                    padding: 22px;
                    border-radius: 8px;
                    background-color: #f1f3f6;
                    color: #111111;
                    font-size: 24px;
                    font-weight: 600;
                    text-align: center;
                    border: 1px solid #d9dde3;
                ">
                    {prediction}
                </div>
                """,
                unsafe_allow_html=True
            )


            st.subheader("Model Confidence")

            st.progress(
                min(confidence / 100, 1.0)
            )

            st.write(
                f"{confidence:.2f}%"
            )


    except Exception as e:

        st.error(
            f"Video processing failed: {e}"
        )


    finally:

        if original_path and os.path.exists(original_path):
            os.remove(original_path)

        if preview_path and os.path.exists(preview_path):
            os.remove(preview_path)