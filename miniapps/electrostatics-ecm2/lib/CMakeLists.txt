# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

list(APPEND ELECTROSTATICS_COMMON_SOURCES
    electrostatics_solver.cpp)
list(APPEND ELECTROSTATICS_COMMON_HEADER
    electrostatics_solver.hpp)

convert_filenames_to_full_paths(ELECTROSTATICS_COMMON_SOURCES)
convert_filenames_to_full_paths(ELECTROSTATICS_COMMON_HEADER)

set(ELECTROSTATICS_COMMON_FILES
    EXTRA_SOURCES ${ELECTROSTATICS_COMMON_SOURCES}
    EXTRA_HEADERS ${ELECTROSTATICS_COMMON_HEADER})
