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

  list(APPEND ELECTROSTATICS_COMMON_SOURCES
      lib/electrostatics_solver.cpp
      bc/electrostatics_bchandler.cpp)

  list(APPEND ELECTROSTATICS_COMMON_HEADERS
      lib/electrostatics_solver.hpp
      bc/electrostatics_bchandler.hpp)

  convert_filenames_to_full_paths(ELECTROSTATICS_COMMON_SOURCES)
  convert_filenames_to_full_paths(ELECTROSTATICS_COMMON_HEADERS)

  set(ELECTROSTATICS_COMMON_FILES
      EXTRA_SOURCES ${ELECTROSTATICS_COMMON_SOURCES}
      EXTRA_HEADERS ${ELECTROSTATICS_COMMON_HEADERS})

  set(ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${MFEM_MINIAPPS_COMMON_HEADERS})


  add_library(electrostatics_ecm2 ${ELECTROSTATICS_COMMON_SOURCES} ${ELECTROSTATICS_COMMON_HEADERS})   
  target_link_libraries(electrostatics_ecm2 mfem mfem-common-ecm2 mfem-common)

  add_mfem_miniapp(electrostatics_test2D
    MAIN tests/electrostatics_test2D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES electrostatics_ecm2)

  add_mfem_miniapp(electrostatics_test3D
    MAIN tests/electrostatics_test3D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES electrostatics_ecm2)

  add_mfem_miniapp(electrostatics_test3D_aniso_rotated
    MAIN tests/electrostatics_test3D_aniso_rotated.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES electrostatics_ecm2)

  add_mfem_miniapp(mms_2D
    MAIN tests/mms_2D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES electrostatics_ecm2)

  add_mfem_miniapp(convergence_2D
    MAIN tests/convergence_2D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES electrostatics_ecm2)

  add_mfem_miniapp(test_spaces
    MAIN tests/test_spaces.cpp
    LIBRARIES mfem mfem-common)

endif () #MFEM_USE_MPI
