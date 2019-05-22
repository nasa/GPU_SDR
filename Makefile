CC=g++

CC_DEF_FLAGS =-std=c++11 -DBOOST_LOG_DYN_LINK -Wall -O3 -Ofast -march=native
CC+=$(CC_DEF_FLAGS)

SRC_DIR := cpp
HPP_DIR := headers
OBJ_DIR := obj

HDF5_PATH_INC = /usr/include/hdf5/serial

CINCLUDE =-I$(HDF5_PATH_INC) -I$(HPP_DIR)
CLINK = -lz -lsz -ldl
CLINK += -lpthread -lboost_system -lboost_program_options -lboost_chrono -lboost_thread -lboost_log -lboost_log_setup

CLINK += -luhd
CLINK += -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_serial

NVCC = nvcc

NV_DEF_FLAGS = -std=c++11 -arch=sm_61
NVCC+=$(NV_DEF_FLAGS)

NVINCLUDE = -I/usr/local/cuda/include/
NVLINK = -lcudart -lcuda -lcufft -lcublas -lculibos

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)

HPP_FILES := $(wildcard $(HPP_DIR)/*.hpp)
CUDA_HPP_FILES := $(wildcard $(HPP_DIR)/*.cuh)

OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
CUDA_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SRC_FILES))

SPHINXBUILD = sphinx-build

SPHINXSOURCEDIR = pyUSRP
SPHINXBUILDDIR = lib_docs
SPHINXOPTS =
all: server

doc:
	$(info Generating C++ documentation...)
	doxygen server_docs/doc_gen
	$(info Generating Python documentation...)
	$(SPHINXBUILD) "$(SPHINXSOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)

server: $(CUDA_OBJ_FILES) $(OBJ_FILES)
	$(info Linking all using nvcc...)
	@$(NVCC) $(CLINK) $(NVLINK) -o $@ $^
	$(info Cleaning object files...)
	@rm -rf $(OBJ_DIR)

$(OBJ_FILES): $(SRC_FILES) $(HPP_FILES) $(CUDA_HPP_FILES)
	$(info Compiling $(patsubst $(OBJ_DIR)/%.o,$(SRC_DIR)/%.cpp,$@) ...)
	@mkdir -p $(OBJ_DIR)
	@$(CC) $(CINCLUDE) $(NVINCLUDE) -c -o $@  $(patsubst $(OBJ_DIR)/%.o,$(SRC_DIR)/%.cpp,$@)

$(CUDA_OBJ_FILES): $(CUDA_SRC_FILES) $(CUDA_HPP_FILES)
	$(info Compiling $(patsubst $(OBJ_DIR)/%.o,$(SRC_DIR)/%.cu,$@) ...)
	@mkdir -p $(OBJ_DIR)
	@$(NVCC) $(CINCLUDE) $(NVINCLUDE) -dc -o $@ $(patsubst $(OBJ_DIR)/%.o,$(SRC_DIR)/%.cu,$@)

clean:
	$(info Cleaning all...)
	@rm -f server
	@rm -rf $(OBJ_DIR)
	@rm -rf docs/build
