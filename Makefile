# llama.cpp convenience Makefile
# This wraps CMake for convenience. Full build system is CMake — see docs/build.md.
#
# Targets:
#   sync           Fetch upstream + rebase onto upstream/master + push + build (default)
#   sync-merge     Fetch upstream + merge into master + push + build (legacy)
#   all            Build everything (default)
#   server         Build llama-server only
#   cli            Build llama-cli only
#   bench          Build llama-bench only
#   quantize       Build llama-quantize only
#   perplexity     Build llama-perplexity only
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

# Backend flags — set to ON or OFF
GGML_CUDA   ?= OFF
GGML_VULKAN ?= ON
GGML_METAL  ?= OFF
GGML_HIP    ?= OFF
GGML_SYCL   ?= OFF
GGML_OPENCL ?= OFF
GGML_CANN   ?= OFF

# Deployment path for systemd (set to empty to skip auto-deploy)
DEPLOY_DIR  ?= /opt/cachy-llama/bin
# Set ENABLE_DEPLOY=1 to automatically copy binaries after a release build.
# Disabled by default so development builds don't try to sudo/stop services.
ENABLE_DEPLOY ?= 0

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

.PHONY: all sync sync-merge server cli bench quantize perplexity release debug clean rebuild deploy restart test info

# --- Opt-in deploy (guarded by ENABLE_DEPLOY=1) ---
#
# To enable automatic deploy after every release build:
#   export ENABLE_DEPLOY=1
# Or pass it on the command line:
#   make ENABLE_DEPLOY=1
#
# This is opt-in so development builds don't attempt sudo or systemctl.

# --- Sync with upstream ---

sync-merge:
	@printf "$(BLUE)⟳ Fetching $(UPSTREAM)...$(NC)\n"
	git fetch $(UPSTREAM)
	@printf "$(BLUE)⟳ Merging $(UPSTREAM)/$(BRANCH) into $(BRANCH)...$(NC)\n"
	git checkout $(BRANCH) && git merge $(UPSTREAM)/$(BRANCH)
	@$(MAKE) release
	@printf "$(BLUE)⟳ Pushing to $(ORIGIN)...$(NC)\n"
	git push $(ORIGIN) $(BRANCH)
	@printf "$(GREEN)✓ $(BRANCH) synced with upstream$(NC)\n"

sync:
	@printf "$(BLUE)⟳ Fetching $(UPSTREAM)...$(NC)\n"
	git fetch $(UPSTREAM)
	@printf "$(BLUE)⟳ Rebasing $(BRANCH) onto $(UPSTREAM)/$(BRANCH)...$(NC)\n"
	git checkout $(BRANCH) && git pull --rebase $(UPSTREAM) $(BRANCH)
	@$(MAKE) release
	@printf "$(BLUE)⟳ Pushing to $(ORIGIN)...$(NC)\n"
	git push $(ORIGIN) $(BRANCH) --force-with-lease || { echo; printf "$(YELLOW)⚠ Push rejected. This usually means someone else pushed to $(ORIGIN)/$(BRANCH) while you were rebasing. Try:$(NC)\n"; printf "  git fetch $(ORIGIN) && git rebase $(ORIGIN)/$(BRANCH) && git push $(ORIGIN) $(BRANCH) --force-with-lease\n"; exit 1; }
	@printf "$(GREEN)✓ $(BRANCH) synced with upstream (clean history)$(NC)\n"

# --- Build targets ---

all: release
	@printf "$(GREEN)✓ Build complete: $(BUILD_DIR)/bin/$(NC)\n"
	@ls -1 $(BUILD_DIR)/bin/ 2>/dev/null | while read f; do printf "  $(CYAN)$$f$(NC)\n"; done

server:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-server -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-server$(NC)\n"

cli:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-cli -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-cli$(NC)\n"

bench:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-bench -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-bench$(NC)\n"

quantize:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-quantize -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-quantize$(NC)\n"

perplexity:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --target llama-perplexity -j $(PARALLEL)
	@printf "$(GREEN)✓ $(BUILD_DIR)/bin/llama-perplexity$(NC)\n"

release:
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) --config Release -j $(PARALLEL)
	@if [ "$(ENABLE_DEPLOY)" = "1" ]; then $(MAKE) deploy; fi

deploy:
ifneq ($(DEPLOY_DIR),)
	@set -eu; \
	parent=$$(dirname -- "$(DEPLOY_DIR)"); \
	sudo mkdir -p -- "$$parent"; \
	stage=$$(sudo mktemp -d "$$parent/.cachy-llama-stage.XXXXXX"); \
	backup="$$parent/.cachy-llama-backup-$$$$"; \
	old_present=0; new_deployed=0; service_stopped=0; \
	rollback() { \
		if [ "$$new_deployed" -eq 1 ]; then sudo rm -rf -- "$(DEPLOY_DIR)"; fi; \
		sudo rm -rf -- "$$stage"; \
		if [ "$$old_present" -eq 1 ]; then sudo mv -- "$$backup" "$(DEPLOY_DIR)"; fi; \
		if [ "$$service_stopped" -eq 1 ]; then sudo systemctl restart llama.cpp || true; fi; \
	}; \
	trap rollback EXIT; \
	sudo test -x "$(BUILD_DIR)/bin/llama-server"; \
	sudo cp -a "$(BUILD_DIR)/bin/." "$$stage/"; \
	sudo test -x "$$stage/llama-server"; \
	sudo chmod 755 -- "$$stage"; \
	sudo systemctl stop llama.cpp; service_stopped=1; \
	if sudo test -e "$(DEPLOY_DIR)"; then sudo mv -- "$(DEPLOY_DIR)" "$$backup"; old_present=1; fi; \
	sudo mv -- "$$stage" "$(DEPLOY_DIR)"; new_deployed=1; \
	sudo systemctl restart llama.cpp; service_stopped=0; \
	trap - EXIT; \
	sudo rm -rf -- "$$backup"; \
	printf "$(GREEN)✓ Deployed to $(DEPLOY_DIR) and restarted llama.cpp$(NC)\n"
endif

restart:
	@printf "$(BLUE)⟳ Restarting llama.cpp service...$(NC)\n"
	sudo systemctl restart llama.cpp
	@printf "$(GREEN)✓ llama.cpp restarted$(NC)\n"
	sudo systemctl status llama.cpp --no-pager -l | head -10

debug:
	@$(CMAKE) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_BACKEND_FLAGS)
	@$(CMAKE) --build $(BUILD_DIR) -j $(PARALLEL)

# --- Housekeeping ---

clean:
	@set -eu; \
	repo_dir=$$(pwd -P); \
	build_dir=$$(realpath -m -- "$(BUILD_DIR)"); \
	case "$$build_dir" in \
		"$$repo_dir"/*) ;; \
		*) echo "Refusing to remove BUILD_DIR outside the repository: $(BUILD_DIR)" >&2; exit 1 ;; \
	esac; \
	rm -rf -- "$(BUILD_DIR)"
	@printf "$(YELLOW)✓ Removed $(BUILD_DIR)/$(NC)\n"

rebuild: clean release

test:
	@$(MAKE) release
	@cd $(BUILD_DIR) && ctest --output-on-failure -j $(PARALLEL)

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
	@printf "  make sync         Fetch upstream + rebase + push + build (default)\n"
	@printf "  make sync-merge   Fetch upstream + merge + push + build\n"
	@printf "  make deploy       Copy binaries to $(DEPLOY_DIR)\n"
	@printf "  make restart      systemctl restart llama.cpp\n"
	@printf "  make all          Full build\n"
	@printf "  make server       llama-server only\n"
	@printf "  make cli          llama-cli only\n"
	@printf "  make bench        llama-bench only\n"
	@printf "  make quantize     llama-quantize only\n"
	@printf "  make perplexity   llama-perplexity only\n"
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
