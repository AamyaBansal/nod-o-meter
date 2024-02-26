FROM python:3.8

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*  


WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "web/app.py"]

# run command: docker run -it --rm --device=/dev/video0 -p 5000:5000 my-flask-app
