import cv2
import face_recognition
import os
import pickle
import threading
import logging
import json
import numpy as np
import faiss
from queue import Queue
import signal
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

KNOWN_FACES_DIR = "faces"
ENCODINGS_FILE = "encodings.pkl"
NAMES_MAP_FILE = "names_map.json"
TOLERANCE = 0.6
FRAME_SKIP = 4
DISPLAY_TIME = 30

known_face_encodings = []
known_face_names = []

cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

if os.path.exists(NAMES_MAP_FILE):
    with open(NAMES_MAP_FILE, "r") as f:
        names_map = json.load(f)
else:
    logging.warning(
        f"{NAMES_MAP_FILE} não encontrado. Usando nomes baseados nos arquivos."
    )
    names_map = {}

def load_or_generate_encodings():
    if os.path.exists(ENCODINGS_FILE):
        logging.info("Carregando codificações de rostos conhecidos.")
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        logging.info("Gerando codificações de rostos conhecidos.")
        encodings = []
        names = []
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith((".png", ".jpg")):
                filepath = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(filepath)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    encodings.append(encoding[0])
                    name = names_map.get(filename, os.path.splitext(filename)[0])
                    names.append(name)
                else:
                    logging.warning(
                        f"Não foi possível encontrar rostos na imagem: {filename}"
                    )
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump((encodings, names), f)
        return encodings, names

class VideoStream:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FPS, 60)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.video_capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.video_capture.release()

known_face_encodings, known_face_names = load_or_generate_encodings()

logging.info("Inicializando o índice FAISS.")
try:
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(128)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    logging.info("FAISS configurado para usar GPU.")
except Exception as e:
    logging.warning(f"Fallo ao configurar GPU. Rodando em CPU: {e}")
    index = faiss.IndexFlatL2(128)

if known_face_encodings:
    index.add(np.array(known_face_encodings, dtype=np.float32))

video_stream = VideoStream()

face_display_data = {}
frame_queue = Queue(maxsize=100)
processed_queue = Queue(maxsize=100)
frame_counter = 0

def capture_frames(video_stream, frame_queue):
    global frame_counter
    while video_stream.running:
        ret, frame = video_stream.read()
        frame_counter += 1
        if ret and not frame_queue.full() and frame_counter % FRAME_SKIP == 0:
            frame_queue.put(frame)

def process_frames(frame_queue, processed_queue):
    global face_display_data
    while video_stream.running:
        if not frame_queue.empty():
            frame = frame_queue.get()

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                distances, indices = index.search(
                    np.array([face_encoding], dtype=np.float32), 1
                )
                name = "Desconhecido"
                confidence = 0.0

                if distances[0][0] < TOLERANCE:
                    best_match_index = indices[0][0]
                    name = known_face_names[best_match_index]
                    confidence = 1 - distances[0][0]

                top, right, bottom, left = face_location
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                face_display_data[name] = {
                    "coords": (top, right, bottom, left),
                    "frames_left": DISPLAY_TIME,
                    "confidence": confidence,
                }

            processed_queue.put(frame)

def display_frames(processed_queue):
    global face_display_data
    while video_stream.running:
        if not processed_queue.empty():
            frame = processed_queue.get()
            for name in list(face_display_data.keys()):
                data = face_display_data[name]
                top, right, bottom, left = data["coords"]
                confidence = data["confidence"]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{name} ({confidence:.2f})",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                face_display_data[name]["frames_left"] -= 1
                if face_display_data[name]["frames_left"] <= 0:
                    del face_display_data[name]

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_program()
                break

def stop_program():
    logging.info("Encerrando o programa...")
    video_stream.stop()
    cv2.destroyAllWindows()
    sys.exit(0)

def signal_handler(sig, frame):
    stop_program()

signal.signal(signal.SIGINT, signal_handler)

capture_thread = threading.Thread(target=capture_frames, args=(video_stream, frame_queue))
process_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_queue))
display_thread = threading.Thread(target=display_frames, args=(processed_queue,))

capture_thread.start()
process_thread.start()
display_thread.start()

capture_thread.join()
process_thread.join()
display_thread.join()

stop_program()
