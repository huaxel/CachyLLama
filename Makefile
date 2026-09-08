# llama.cpp convenience Makefile
# This wraps CMake for convenience. Full build system is CMake — see docs/build.md.
#
# Targets:
#   sync           Fetch upstream + rebase onto upstream/master + build + push
#   sync-merge     Fetch upstream + merge into master + build + push (legacy)
#   all            Build everything (default)
#   server         Build llama-server only
#   cli            Build llama-cli only
#   bench          Build llama-bench only
#   quantize       Build llama-quantize only
#   release        Full Release build with optimizations
#   debug          Debug build
#   clean          Remove build directory
#   rebuild        Full clean + release build
#   test           Run tests
#   info           Show detected features and build settings

# --- User-configurable variables (override via environment or command line) ---
CMAKE       ?= cmake
BUILD_DIR   ?= build
PARALLEL    ?= $(shell nproc 2>/dev/null || echo 4)
BUILD_TYPE  ?= Release

# Git remotes for syncing with upstream
UPSTREAM    ?= upstream
ORIGIN      ?= origin
BRANCH      ?= master

# Backend flags — set to ON or OFF (Vulkan is opt-in for portable defaults)
GGML_CUDA   ?= OFF
GGML_VULKAN ?= OFF
GGML_METAL  ?= OFF
GGML_HIP    ?= OFF
GGML_SYCL   ?= OFF
GGML_OPENCL ?= OFF
GGML_CANN   ?= OFF

# --- Derived flags ---
CMAKE_BACKEND_FLAGS = \
	-DGGML_CUDA=$(GGML_CUDA) \
	-DGGML_VULKAN=$(GGML_VULKAN) \
	-DGGML_METAL=$(GGML_METAL) \
	-DGGML_HIP=$(GGML_HIP) \
	-DGGML_SYCL=$(GGML_SYCL) \
	-DGGML_OPENCL=$(GGML_OPENCL) \
	-DGGML_CANN=$(GGML_CANN)
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_BACKEND_FLAGS)

# Detect Vulkan availability (glslc)
HAVE_GLSLC := $(shell which glslc 2>/dev/null && echo yes || echo no)

# Pretty-print
BLUE := \033[34;1m
GREEN := \033[32;1m
YELLOW := \033[33;1m
CYAN := \033[36;1m
NC := \033[0m

.PHONY: all sync sync-merge server cli bench quantize release debug clean rebuild test info

# --- Sync with upstream ---

sync-merge:
	@if test -n "$$(git status --porcelain)"; then \
		echo "Refusing to sync with a dirty working tree" >&2; exit 1; \
	fi
	@printf "$(BLUE)⟳ Fetching $(UPSTREAM)...$(NC)\n"
	git fetch $(UPSTREAM)
	@printf "$(BLUE)⟳ Merging $(UPSTREAM)/$(BRANCH) into $(BRANCH)...$(NC)\n"
	git checkout $(BRANCH) && git merge $(UPSTREAM)/$(BRANCH)
	@$(MAKE) release
	@printf "$(BLUE)⟳ Pushing to $(ORIGIN)...$(NC)\n"
	git push $(ORIGIN) $(BRANCH)
	@printf "$(GREEN)✓ $(BRANCH) synced with upstream$(NC)\n"

sync:
	@if test -n "$$(git status --porcelain)"; then \
		echo "Refusing to sync with a dirty working tree" >&2; exit 1; \
	fi
	@printf "$(BLUE)⟳ Fetching $(UPSTREAM)...$(NC)\n"
	git fetch $(UPSTREAM)
	@printf "$(BLUE)⟳ Rebasing $(BRANCH) onto $(UPSTREAM)/$(BRANCH)...$(NC)\n"
	git checkout $(BRANCH) && git pull --rebase $(UPSTREAM) $(BRANCH)
	@$(MAKE) release
	@printf "$(BLUE)⟳ Pushing to $(ORIGIN)...$(NC)\n"
	git push $(ORIGIN) $(BRANCH) --force-with-lease || { \
		echo "Push rejected; fetch $(ORIGIN) and reconcile before retrying" >&2; exit 1; \
	}
	@printf "$(GREEN)✓ $(BRANCH) synced with upstream (clean history)$(NC)\n"

# --- Build targets ---

all: release
	@printf "$(GREEN)✓ Build complete: $(BUILD_DIR)/bin/$(NC)\n"
	@ls -1 $(BUILD_DIR)/bin/ 2>/dev/null | while read f; do printf "  $(CYAN)$$f$(NC)\n"; done

server:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-server --config $(BUILD_TYPE) -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-server$(NC)\n"

cli:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-cli --config $(BUILD_TYPE) -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-cli$(NC)\n"

bench:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-bench --config $(BUILD_TYPE) -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-bench$(NC)\n"

quantize:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-quantize --config $(BUILD_TYPE) -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-quantize$(NC)\n"

release:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) -j $(PARALLEL)

debug:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_BACKEND_FLAGS) -DCMAKE_BUILD_TYPE=Debug
	@$(CMAKE) --build $(BUILD_DIR) --config Debug -j $(PARALLEL)

# --- Housekeeping ---

clean:
	@set -eu; \
	case "$(BUILD_DIR)" in \
		""|/*|.|..|../*|*/../*|*/..) \
			echo "Refusing to remove unsafe BUILD_DIR: $(BUILD_DIR)" >&2; exit 1 ;; \
		esac; \
	rm -rf -- "$(BUILD_DIR)"
	@printf "$(YELLOW)✓ Removed $(BUILD_DIR)/$(NC)\n"

rebuild: clean release

test:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS) -DLLAMA_BUILD_TESTS=ON
	@$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) -j $(PARALLEL)
	@cd $(BUILD_DIR) && ctest -C $(BUILD_TYPE) --output-on-failure -j $(PARALLEL)

# --- Info ---

info:
	@printf "$(BLUE)=== llama.cpp Build Configuration ===$(NC)\n"
	@printf "  CMAKE:           $(shell $(CMAKE) --version 2>/dev/null | head -1)\n"
	@printf "  BUILD_DIR:       $(BUILD_DIR)\n"
	@printf "  BUILD_TYPE:      $(BUILD_TYPE)\n"
	@printf "  PARALLEL_JOBS:   $(PARALLEL)\n"
	@printf "  Cores available: $(shell nproc 2>/dev/null || echo unknown)\n"
	@printf "\n"
	@printf "$(BLUE)Backends:$(NC)\n"
	@printf "  GGML_CUDA:       $(GGML_CUDA)\n"
	@printf "  GGML_VULKAN:     $(GGML_VULKAN)  $(if $(findstring yes,$(HAVE_GLSLC)),$(GREEN)[glslc detected]$(NC),$(YELLOW)[glslc not found]$(NC))\n"
	@printf "  GGML_METAL:      $(GGML_METAL)\n"
	@printf "  GGML_HIP:        $(GGML_HIP)\n"
	@printf "  GGML_SYCL:       $(GGML_SYCL)\n"
	@printf "  GGML_OPENCL:     $(GGML_OPENCL)\n"
	@printf "  GGML_CANN:       $(GGML_CANN)\n"
	@printf "\n"
	@printf "$(BLUE)Flags:$(NC)\n"
	@printf "  $(CMAKE_FLAGS)\n"
	@printf "\n"
	@printf "$(BLUE)Git remotes:$(NC)\n"
	@printf "  UPSTREAM:        $(UPSTREAM)\n"
	@printf "  ORIGIN:          $(ORIGIN)\n"
	@printf "  BRANCH:          $(BRANCH)\n"
	@printf "\n"
	@printf "$(BLUE)Available targets:$(NC)\n"
	@printf "  make              Shortcut for 'make release'\n"
	@printf "  make sync         Fetch upstream + rebase + build + push\n"
	@printf "  make sync-merge   Fetch upstream + merge + build + push\n"
	@printf "  make all          Full build\n"
	@printf "  make server       llama-server only\n"
	@printf "  make cli          llama-cli only\n"
	@printf "  make bench        llama-bench only\n"
	@printf "  make quantize     llama-quantize only\n"
	@printf "  make release      Release build\n"
	@printf "  make debug        Debug build\n"
	@printf "  make clean        Remove build directory\n"
	@printf "  make rebuild      Clean + release build\n"
	@printf "  make test         Build + run tests\n"
	@printf "  make info         Show this configuration\n"
	@printf "\n"
	@printf "Override backends on the command line, e.g.:\n"
	@printf "  make GGML_CUDA=ON GGML_VULKAN=OFF\n"
	@printf "  make server GGML_VULKAN=ON\n"
	@printf "\n"
	@printf "Override git remotes if your setup differs:\n"
	@printf "  make sync UPSTREAM=upstream ORIGIN=origin BRANCH=master\n"

.DEFAULT_GOAL := all
