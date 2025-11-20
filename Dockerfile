# Use an official Python runtime as a parent image
FROM python:3.10

# Install system dependencies required for face-recognition and dlib
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    libssl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pip and upgrade it
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow and Keras first (Avoids conflicts)
RUN pip install --no-cache-dir tensorflow==2.15.0 keras==2.15.0

# Install required Python packages
RUN pip install --no-cache-dir dlib face-recognition
Run pip install absl-py altair annotated-types anyio arrow astunparse attrs binaryornot blinker cachetools certifi chardet charset-normalizer click colorama comtypes contourpy cookiecutter cvzone cycler dlib dm-tree dnspython email_validator exceptiongroup fastapi fastapi-cli filelock filterpy Flask flatbuffers flet flet-core flet-runtime fonttools fsspec gast gitdb GitPython google-auth google-auth-oauthlib google-pasta grpcio h11 h5py httpcore httptools httpx idna imageio itsdangerous Jinja2 jsonschema jsonschema-specifications  kiwisolver lazy_loader libclang Markdown markdown-it-py MarkupSafe matplotlib mdurl ml-dtypes mpmath namex networkx numpy oauthlib opencv-python-headless opt-einsum orjson packaging pandas pefile pillow protobuf psutil py-cpuinfo pyarrow pyasn1 pyasn1-modules pydantic pydantic_core python-dateutil python-dotenv python-multipart python-slugify pyttsx3 pytz PyYAML qrcode referencing repath requests requests-oauthlib rich rpds-py rsa scikit-image scipy seaborn shellingham six smmap sniffio starlette streamlit sympy tenacity tensorboard tensorboard-data-server termcolor text-unidecode thop tifffile toml toolz torch torchvision tornado tqdm typer types-python-dateutil typing_extensions tzdata ujson ultralytics urllib3 uvicorn watchdog watchfiles websockets Werkzeug wrapt

# RUN pip install --no-cache-dir -r requirements.txt || true

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
