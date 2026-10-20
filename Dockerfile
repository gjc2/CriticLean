FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY LICENSES ./LICENSES
COPY src ./src
RUN pip install --no-cache-dir '.[datasets]'
COPY configs ./configs
COPY examples ./examples
RUN useradd --create-home critic && mkdir -p /work && chown critic:critic /work
USER critic
WORKDIR /work
ENTRYPOINT ["lean-critic"]
CMD ["--help"]
