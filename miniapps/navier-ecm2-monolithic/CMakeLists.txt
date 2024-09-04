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

  list(APPEND SNAVIER_COMMON_SOURCES
      custom_bilinteg.cpp utils.cpp navier_unsteady_solver.cpp)
  list(APPEND SNAVIER_COMMON_HEADERS
      custom_bilinteg.hpp utils.hpp navier_unsteady_solver.hpp)

  convert_filenames_to_full_paths(SNAVIER_COMMON_SOURCES)
  convert_filenames_to_full_paths(SNAVIER_COMMON_HEADERS)

  set(SNAVIER_COMMON_FILES
      EXTRA_SOURCES ${SNAVIER_COMMON_SOURCES}
      EXTRA_HEADERS ${SNAVIER_COMMON_HEADERS})


  # Tests NS
  add_mfem_miniapp(navier_ecm2_mms_mono
    MAIN navier_mms.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_ecm2_poiseulle_mono
    MAIN navier_poiseulle.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_ecm2_LidDriven_mono
    MAIN navier_lid_driven.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_ecm2_flowaroundcyl_mono
    MAIN navier_FlowAroundCyl.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_ecm2_backfacingstep_mono
    MAIN navier_BackFacingStep.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_ecm2_benchmarkFDA_mono
    MAIN navier_BenchmarkFDA.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  # Create a symbolic link for meshes
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_CURRENT_SOURCE_DIR}/Mesh ${CMAKE_CURRENT_BINARY_DIR}/Mesh
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

endif ()
