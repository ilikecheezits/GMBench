FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY inspect_benchmark ./inspect_benchmark
COPY main.py ./
RUN pip install --no-cache-dir .
# Mount logs/artifacts/profiles and pass OPENAI_API_KEY at runtime, e.g.
#   docker run --rm -it -e OPENAI_API_KEY -v "$PWD/out:/app/out" npbench run --log-dir out/logs --report-dir out/report
ENTRYPOINT ["npbench"]
CMD ["--help"]
