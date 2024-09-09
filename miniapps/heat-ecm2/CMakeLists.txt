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

  set(HEAT_DIR ${CMAKE_SOURCE_DIR}/miniapps/heat-ecm2)

  # Add the include directories
  include_directories(
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${HEAT_DIR}
  )

  list(APPEND HEAT_SRCS
    lib/heat_solver.cpp
    lib/heat_operator.cpp
    bc/heat_bchandler.cpp
  )

  list(APPEND HEAT_HDRS
    lib/heat_solver.cpp
    lib/heat_operator.cpp
    bc/heat_bchandler.cpp
  )

  convert_filenames_to_full_paths(HEAT_SRCS)
  convert_filenames_to_full_paths(HEAT_HDRS)

  mfem_add_library(heat-ecm2 ${HEAT_SRCS} ${HEAT_HDRS})
    
  target_link_libraries(heat-ecm2
    mfem
    mfem-common-ecm2
    mfem-common
    ${TPL_LIBRARIES})


add_mfem_miniapp(heat_test
    MAIN tests/heat_test.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common heat-ecm2)



add_mfem_miniapp(convergence
    MAIN tests/convergence.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common heat-ecm2)


  add_mfem_miniapp(heat_test_3Daniso
    MAIN tests/heat_test_3Daniso.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common heat-ecm2)

endif () #MFEM_USE_MPI

