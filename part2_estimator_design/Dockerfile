# Part II image.  Build from the repository root (Part II imports Part I's src/):
#
#   docker build -f part2_estimator_design/Dockerfile -t part2 .
#   docker run --rm part2                                   # runs the test suite
#   docker run --rm -v "$PWD/part2_estimator_design/runs:/work/part2_estimator_design/runs" \
#       part2 python scripts/step4_learned_designs.py --cases LiH_R3p0_HF --workers 60
#
# On a cluster without Docker, convert it (apptainer build part2.sif docker-daemon://part2:latest)
# or create a venv from requirements.txt directly.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY part2_estimator_design/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
COPY part1_full_reproducibility_package/src part1_full_reproducibility_package/src
COPY part1_full_reproducibility_package/runs part1_full_reproducibility_package/runs
COPY part2_estimator_design part2_estimator_design
WORKDIR /work/part2_estimator_design

CMD ["python", "-m", "pytest", "tests", "-q"]
