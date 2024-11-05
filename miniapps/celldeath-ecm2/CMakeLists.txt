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

if (MFEM_USE_MPI)
  #add_subdirectory(lib)

  set(CELLDEATH_DIR ${CMAKE_SOURCE_DIR}/miniapps/celldeath-ecm2)

  # Add the include directories
  include_directories(
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${CELLDEATH_DIR}
  )

  # Eigen version
  list(APPEND CELLDEATH_SRCS
    lib/celldeath_solver.cpp
  )

  list(APPEND CELLDEATH_HDRS
    lib/celldeath_solver.hpp
  )

  convert_filenames_to_full_paths(CELLDEATH_SRCS)
  convert_filenames_to_full_paths(CELLDEATH_HDRS)

  mfem_add_library(celldeath-ecm2 ${CELLDEATH_SRCS} ${CELLDEATH_HDRS})
    
  target_link_libraries(celldeath-ecm2
    mfem
    mfem-common-ecm2
    mfem-common
    ${TPL_LIBRARIES})

 # Gotran version
 list(APPEND CELLDEATH_GOTRAN_SRCS
 lib/celldeath_solver_gotran.cpp
  lib/ThreeStateCellDeath.cpp
)

list(APPEND CELLDEATH_GOTRAN_HDRS
 lib/celldeath_solver_gotran.hpp
 lib/ThreeStateCellDeath.h
)

convert_filenames_to_full_paths(CELLDEATH_GOTRAN_SRCS)
convert_filenames_to_full_paths(CELLDEATH_GOTRAN_HDRS)

mfem_add_library(celldeath-gotran-ecm2 ${CELLDEATH_GOTRAN_SRCS} ${CELLDEATH_GOTRAN_HDRS})
 
target_link_libraries(celldeath-gotran-ecm2
 mfem
 mfem-common-ecm2
 mfem-common
 ${TPL_LIBRARIES})

add_mfem_miniapp(test_square
    MAIN tests/test_square.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common celldeath-ecm2 celldeath-gotran-ecm2)

endif () #MFEM_USE_MPI

