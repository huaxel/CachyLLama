# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # CachyLLama server-focused build: only build tools needed by the server
    # and diagnostic/benchmarking tooling. Shedded tools (completion,
    # perplexity, parser, gguf-split, fit-params, results,
    # cvector-generator, export-lora) are omitted for a leaner build.
    # batched-bench is included for benchmark.sh's batched test suite.
    add_subdirectory(cli)
    add_subdirectory(imatrix)
    add_subdirectory(batched-bench)
    add_subdirectory(llama-bench)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    # Extra tools invoked by CI (ci/run.sh, llama2c workflow tests). Not part
    # of the lean server-focused default build; enable with LLAMA_BUILD_EXTRA_TOOLS=ON.
    if (LLAMA_BUILD_EXTRA_TOOLS)
        add_subdirectory(completion)
        add_subdirectory(perplexity)
        add_subdirectory(fit-params)
    endif()
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
endif()
