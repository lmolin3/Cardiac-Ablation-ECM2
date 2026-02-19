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

  list(APPEND CELLDEATH_COMMON_SOURCES
    lib/celldeath_solver.cpp
    lib/ThreeStateCellDeath.cpp
  )

  list(APPEND CELLDEATH_COMMON_HEADERS
    lib/celldeath_solver.hpp
    lib/ThreeStateCellDeath.h
  )

  set(CELLDEATH_COMMON_FILES
  EXTRA_SOURCES ${CELLDEATH_COMMON_SOURCES}
  EXTRA_HEADERS ${CELLDEATH_COMMON_HEADERS})

  convert_filenames_to_full_paths(CELLDEATH_COMMON_SOURCES)
  convert_filenames_to_full_paths(CELLDEATH_COMMON_HEADERS)

  mfem_add_library(celldeath-ecm2 ${CELLDEATH_COMMON_SOURCES} ${CELLDEATH_COMMON_HEADERS})
    
  target_link_libraries(celldeath-ecm2 mfem mfem-common-ecm2 mfem-common)
    
add_mfem_miniapp(test_celldeath
    MAIN tests/test_celldeath.cpp
    ${CELLDEATH_COMMON_FILES}
    LIBRARIES celldeath-ecm2)

add_mfem_miniapp(test_transfer
    MAIN tests/test_transfer.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

endif () #MFEM_USE_MPI

