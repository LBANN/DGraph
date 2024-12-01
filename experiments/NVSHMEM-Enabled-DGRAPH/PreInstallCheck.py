import os
import glob


def _check_nvshmem():
    print("Checking if NVSHMEM_HOME is set")

    error_signal = False
    if "NVSHMEM_HOME" not in os.environ:
        print("ERROR: NVSHMEM_HOME is not set\n")
        error_signal = True
    else:
        print(f"NVSHMEM_HOME: {os.environ['NVSHMEM_HOME']}\n")

    return error_signal


def _check_mpi():
    print("Checking if MPI_HOME is set")

    error_signal = False
    if "MPI_HOME" not in os.environ:
        print("ERROR: MPI_HOME is not set\n")
        error_signal = True
    else:
        print(f"MPI_HOME: {os.environ['MPI_HOME']}\n")

    return error_signal


def _check_cuda():
    print("Checking if CUDA_HOME is set")

    error_signal = False
    if "CUDA_HOME" not in os.environ:
        print("ERROR: CUDA_HOME is not set\n")
        error_signal = True
    else:
        print(f"CUDA_HOME: {os.environ['CUDA_HOME']}\n")

    return error_signal


def _check_ld_library_path():
    print("Checking if LD_LIBRARY_PATH is correctly set")

    error_signal = False
    available_libs = []
    if "LD_LIBRARY_PATH" not in os.environ:
        print("ERROR: LD_LIBRARY_PATH is not set\n")
        error_signal = True
    else:
        library_paths = os.environ["LD_LIBRARY_PATH"].split(":")

        for path in library_paths:
            SOs = glob.glob(f"{path}/*.so")
            static_libs = glob.glob(f"{path}/*.a")
            parsed_SO = [os.path.basename(SO) for SO in SOs]
            parsed_static_libs = [os.path.basename(lib) for lib in static_libs]
            available_libs.extend(parsed_SO)
            available_libs.extend(parsed_static_libs)

    necessary_libs = [
        "libmpi.so",
        "libnvshmem_host.so",
        "libcudart.so",
        "libnvshmem.a",
        "libmpi.a",
        "nvshmem_bootstrap_mpi.so",
    ]
    for lib in necessary_libs:
        if lib not in available_libs:
            print(f"ERROR: Could not find {lib} in LD_LIBRARY_PATH\n")
            error_signal = True
    return error_signal


def _check_install_flag():
    print("Checking if DISABLE_DGRAPH_NVSHMEM is set")

    error_signal = False
    if "DISABLE_DGRAPH_NVSHMEM" in os.environ:
        if os.environ["DISABLE_DGRAPH_NVSHMEM"] == "1":
            print("DISABLE_DGRAPH_NVSHMEM is set to 1. NVSHMEM will not be enabled\n")
            error_signal = True

    return error_signal


def main():
    print("Running pre-install checks for NVSHMEM-enabled DGraph")
    print("*" * 80)
    nvshmem_error_signal = _check_nvshmem()

    mpi_error_signal = _check_mpi()
    cuda_error_signal = _check_cuda()
    ld_library_error_signal = _check_ld_library_path()
    install_flag_error_signal = _check_install_flag()

    error_sig = (
        nvshmem_error_signal
        or mpi_error_signal
        or cuda_error_signal
        or ld_library_error_signal
        or install_flag_error_signal
    )

    if error_sig:
        print("Pre-install checks failed")
    else:
        print("Pre-install checks passed!")
    print("*" * 80)


if __name__ == "__main__":
    main()
