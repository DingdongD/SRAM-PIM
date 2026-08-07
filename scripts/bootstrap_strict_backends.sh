#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: bootstrap_strict_backends.sh REPO_ROOT}"
ROOT="$(cd "${ROOT}" && pwd)"
THIRD_PARTY="${ROOT}/third_party"
BIN="${THIRD_PARTY}/bin"

SCALESIM_COMMIT="9f98c4371055a54c75209c2e02b640b897550532"
RAMULATOR_COMMIT="b30320bc9385b708e86b67ebb9f48858cc66d798"
BOOKSIM_COMMIT="28f43299f1706a3160ffac721ca461d74eb6e618"
CACTI_COMMIT="1ffd8dfb10303d306ecd8d215320aea07651e878"

mkdir -p "${THIRD_PARTY}" "${BIN}"

test ! -e "${THIRD_PARTY}/scalesim"
test ! -e "${THIRD_PARTY}/ramulator2"
test ! -e "${THIRD_PARTY}/booksim2"
test ! -e "${THIRD_PARTY}/cacti"

git clone https://github.com/scalesim-project/SCALE-Sim.git "${THIRD_PARTY}/scalesim"
git -C "${THIRD_PARTY}/scalesim" checkout "${SCALESIM_COMMIT}"
python3 -m pip install -e "${THIRD_PARTY}/scalesim"

git clone https://github.com/CMU-SAFARI/ramulator2.git "${THIRD_PARTY}/ramulator2"
git -C "${THIRD_PARTY}/ramulator2" checkout "${RAMULATOR_COMMIT}"
cmake -S "${THIRD_PARTY}/ramulator2" -B "${THIRD_PARTY}/ramulator2/build" -DRAMULATOR_PYTHON_BINDINGS=ON
cmake --build "${THIRD_PARTY}/ramulator2/build" -j
PYTHONPATH="${THIRD_PARTY}/ramulator2/python" \
  python3 -m ramulator export "${ROOT}/configs/external/ramulator2_external.py" \
  -o "${THIRD_PARTY}/ramulator2/strict_external.yaml"
g++ -std=c++20 -O2 \
  -I"${THIRD_PARTY}/ramulator2/src" \
  "${ROOT}/bridges/ramulator2_bridge.cpp" \
  -L"${THIRD_PARTY}/ramulator2" -lramulator \
  -Wl,-rpath,"${THIRD_PARTY}/ramulator2" \
  -o "${BIN}/ramulator2_bridge"

git clone https://github.com/booksim/booksim2.git "${THIRD_PARTY}/booksim2"
git -C "${THIRD_PARTY}/booksim2" checkout "${BOOKSIM_COMMIT}"
make -C "${THIRD_PARTY}/booksim2/src" -j
BOOKSIM_INCLUDES=(
  -I"${THIRD_PARTY}/booksim2/src"
  -I"${THIRD_PARTY}/booksim2/src/arbiters"
  -I"${THIRD_PARTY}/booksim2/src/allocators"
  -I"${THIRD_PARTY}/booksim2/src/routers"
  -I"${THIRD_PARTY}/booksim2/src/networks"
  -I"${THIRD_PARTY}/booksim2/src/power"
)
g++ -std=c++11 -O2 "${BOOKSIM_INCLUDES[@]}" \
  -c "${ROOT}/bridges/booksim2_bridge.cpp" -o "${BIN}/booksim2_bridge.o"
mapfile -t BOOKSIM_OBJECTS < <(find "${THIRD_PARTY}/booksim2/src" -name '*.o' ! -name 'main.o' | sort)
g++ "${BIN}/booksim2_bridge.o" "${BOOKSIM_OBJECTS[@]}" -o "${BIN}/booksim2_bridge"
rm "${BIN}/booksim2_bridge.o"

git clone https://github.com/HewlettPackard/cacti.git "${THIRD_PARTY}/cacti"
git -C "${THIRD_PARTY}/cacti" checkout "${CACTI_COMMIT}"
make -C "${THIRD_PARTY}/cacti" -j
python3 "${ROOT}/scripts/prepare_cacti_config.py" \
  --source "${THIRD_PARTY}/cacti/cache.cfg" \
  --output "${THIRD_PARTY}/cacti/strict_sram.cfg"

git -C "${THIRD_PARTY}/scalesim" diff --quiet
git -C "${THIRD_PARTY}/ramulator2" diff --quiet
git -C "${THIRD_PARTY}/booksim2" diff --quiet
git -C "${THIRD_PARTY}/cacti" diff --quiet
PYTHONPATH="${ROOT}" python3 "${ROOT}/scripts/smoke_strict_backends.py" \
  --config "${ROOT}/configs/cmodel_centralized.yaml"
