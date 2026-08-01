CC ?= cc
CXX ?= c++
PYTHON ?= python3

CFLAGS ?= -O3 -march=native -Wall -Wextra -Werror -Wno-unused-function
CXXFLAGS ?= -O2 -Wall -Wextra -Werror

# These flags define the program's semantics and remain in force when callers
# override CFLAGS for optimization, instrumentation, or packaging.
REQUIRED_CFLAGS := -fwrapv -std=c11
MGPT_CPPFLAGS := -Isrc \
	-DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES \
	-DMGPT_NO_TRAIN -DMGPT_PRECISION_TRACE
LLAMA_CPPFLAGS := -Isrc/llama \
	-DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES

BIN := gpt_int_infer_trace
ROUND_BIN := mgw_round
LLAMA_MGWI_BIN := llama_mgwi
GPU_REAL_HOST_TEST_BIN := gpu_real_weight_math_test
GPU_P13_HOST_TEST_BIN := gpu_p13_int16_math_test
SAFETENSORS_CONVERSION_TEST_BIN := safetensors_conversion_test
MODEL := model.mgw

.PHONY: all test inspect sweep sensitivity mixed llama-mgwi \
	gpu-host-tests clean

all: $(BIN) $(ROUND_BIN)

$(BIN): src/microgpt_int.c src/fp_math.h
	$(CC) $(CPPFLAGS) $(MGPT_CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) \
		-o $@ src/microgpt_int.c

$(ROUND_BIN): tools/mgw_round.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) \
		-o $@ tools/mgw_round.c

$(LLAMA_MGWI_BIN): src/llama/llama_int.c src/llama/fp_math.h \
		src/llama/safetensors.h src/llama/tokenizer.h
	$(CC) $(CPPFLAGS) $(LLAMA_CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) \
		-o $@ src/llama/llama_int.c

$(SAFETENSORS_CONVERSION_TEST_BIN): \
		tests/test_safetensors_conversion.c src/llama/safetensors.h
	$(CC) $(CPPFLAGS) -Isrc/llama $(CFLAGS) $(REQUIRED_CFLAGS) \
		-o $@ tests/test_safetensors_conversion.c

$(GPU_REAL_HOST_TEST_BIN): \
		spikes/gpu-real-weight/test_real_weight_math.cpp \
		spikes/gpu-real-weight/real_weight_math.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 \
		-Ispikes/gpu-real-weight \
		-o $@ spikes/gpu-real-weight/test_real_weight_math.cpp

$(GPU_P13_HOST_TEST_BIN): \
		spikes/gpu-p13-next/test_p13_int16_math.cpp \
		spikes/gpu-p13-next/p13_int16_math.h \
		spikes/gpu-p13-next/p13_int16_selftest.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 \
		-Ispikes/gpu-p13-next \
		-o $@ spikes/gpu-p13-next/test_p13_int16_math.cpp

llama-mgwi: $(LLAMA_MGWI_BIN)

gpu-host-tests: $(GPU_REAL_HOST_TEST_BIN) $(GPU_P13_HOST_TEST_BIN)
	./$(GPU_REAL_HOST_TEST_BIN)
	./$(GPU_P13_HOST_TEST_BIN)

test: $(BIN) $(ROUND_BIN) $(LLAMA_MGWI_BIN) \
		$(GPU_REAL_HOST_TEST_BIN) $(GPU_P13_HOST_TEST_BIN) \
		$(SAFETENSORS_CONVERSION_TEST_BIN)
	$(PYTHON) -I -B -m unittest discover -s tests -v
	./$(BIN) --load $(MODEL) >/dev/null
	./$(GPU_REAL_HOST_TEST_BIN)
	./$(GPU_P13_HOST_TEST_BIN)
	./$(SAFETENSORS_CONVERSION_TEST_BIN)
	$(PYTHON) -I -B spikes/mcu-f12-microgpt/prepare.py >/dev/null
	sh -n tools/run_heldout_gate.sh

inspect:
	$(PYTHON) tools/mgw_precision.py inspect $(MODEL)

sweep: $(BIN)
	$(PYTHON) tools/run_ladder.py \
		--binary ./$(BIN) \
		--model $(MODEL) \
		--bits all \
		--csv results/weight_ladder.csv \
		--json results/weight_ladder.json

sensitivity: $(BIN)
	$(PYTHON) tools/run_tensor_sensitivity.py \
		--binary ./$(BIN) \
		--model $(MODEL) \
		--bits 11,10,9 \
		--csv results/tensor_sensitivity.csv \
		--json results/tensor_sensitivity.json

mixed: $(BIN)
	$(PYTHON) tools/run_mixed_ladder.py \
		--binary ./$(BIN) \
		--model $(MODEL) \
		--csv results/mixed_ladder.csv \
		--json results/mixed_ladder.json

clean:
	rm -f $(BIN) $(ROUND_BIN) $(LLAMA_MGWI_BIN) \
		$(GPU_REAL_HOST_TEST_BIN) $(GPU_P13_HOST_TEST_BIN) \
		$(SAFETENSORS_CONVERSION_TEST_BIN)
