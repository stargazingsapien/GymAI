{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd859c10-04aa-43f5-ab7a-b58913b79495",
   "metadata": {},
   "outputs": [],
   "source": [
    "# streamlit_app.py\n",
    "\n",
    "import streamlit as st\n",
    "from streamlit_webrtc import webrtc_streamer, VideoTransformerBase\n",
    "import av\n",
    "import cv2\n",
    "import numpy as np\n",
    "import mediapipe as mp\n",
    "\n",
    "# Setup\n",
    "mp_pose = mp.solutions.pose\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "\n",
    "# BMI Calculator and Diet Plan\n",
    "st.set_page_config(page_title=\"AI Fitness Assistant\", layout=\"centered\")\n",
    "st.title(\"🏋️ AI Fitness Assistant\")\n",
    "\n",
    "with st.expander(\"📊 BMI Calculator\"):\n",
    "    weight = st.number_input(\"Enter your weight (kg):\", min_value=1.0, step=0.5)\n",
    "    height = st.number_input(\"Enter your height (meters):\", min_value=0.5, step=0.01)\n",
    "\n",
    "    if st.button(\"Calculate BMI\"):\n",
    "        bmi = weight / (height ** 2)\n",
    "        st.success(f\"Your BMI is: {bmi:.2f}\")\n",
    "\n",
    "        if bmi < 18.5:\n",
    "            st.info(\"Underweight: Include high-calorie, protein-rich foods.\")\n",
    "        elif 18.5 <= bmi < 24.9:\n",
    "            st.success(\"Normal: Maintain a balanced diet and stay active.\")\n",
    "        elif 25 <= bmi < 29.9:\n",
    "            st.warning(\"Overweight: Watch your calorie intake and stay active.\")\n",
    "        else:\n",
    "            st.error(\"Obese: Consult a doctor and reduce calorie intake.\")\n",
    "\n",
    "st.markdown(\"---\")\n",
    "st.header(\"🎥 Real-time Pose Detection\")\n",
    "\n",
    "class PoseDetector(VideoTransformerBase):\n",
    "    def __init__(self):\n",
    "        self.pose = mp_pose.Pose()\n",
    "\n",
    "    def transform(self, frame: av.VideoFrame) -> np.ndarray:\n",
    "        img = frame.to_ndarray(format=\"bgr24\")\n",
    "        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "        results = self.pose.process(img_rgb)\n",
    "\n",
    "        if results.pose_landmarks:\n",
    "            mp_drawing.draw_landmarks(\n",
    "                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,\n",
    "                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),\n",
    "                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)\n",
    "            )\n",
    "        return img\n",
    "\n",
    "webrtc_streamer(\n",
    "    key=\"pose\",\n",
    "    video_processor_factory=PoseDetector,\n",
    "    rtc_configuration={\n",
    "        \"iceServers\": [{\"urls\": [\"stun:stun.l.google.com:19302\"]}]\n",
    "    },\n",
    "    media_stream_constraints={\"video\": True, \"audio\": False},\n",
    ")\n",
    "\n",
    "st.markdown(\"---\")\n",
    "st.caption(\"Built with ❤️ using Streamlit and Mediapipe\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
