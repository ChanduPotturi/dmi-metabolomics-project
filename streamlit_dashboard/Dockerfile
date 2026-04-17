FROM python:3.12.3

# PYTHONUNBUFFERED=1  -> python output logs immediately
# PIP_NO_CACHE_DIR=1  -> pip will not store unnecessary caches
# PYTHONPATH=/app     -> adding /app to the Python's module search path

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# create a user instead using root user
RUN useradd -m appuser

# working dir
# all next comands will be executed in the /app
WORKDIR /app


# dependencies for cache
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy all project
COPY . .

# give root user rights to /app
RUN chown -R appuser:appuser /app

# run it not from root
USER appuser

EXPOSE 8501

# this thing start the run app in docker
# CMD[0] and CMD[1] - running a streamlit
# CMD[2] - path to app.py
# CMD[3] - port 
# CMD[4] - setup setting 0.0.0.0 (or you will not see app beyond the container)
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
