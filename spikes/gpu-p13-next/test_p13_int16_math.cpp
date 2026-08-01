#include "p13_int16_selftest.h"

#include <cstdio>
#include <exception>

int main() {
    try {
        const p13_int16::HostSelfTestResult result =
            p13_int16::run_host_math_selftests();
        std::printf(
            "p13 int16 math tests: PASS checks=%zu\n",
            result.checks
        );
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
