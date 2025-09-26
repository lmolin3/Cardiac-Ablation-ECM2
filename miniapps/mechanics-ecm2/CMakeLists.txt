# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

  set(MECHANICS_DIR ${CMAKE_SOURCE_DIR}/miniapps/mechanics-ecm2)
  set(MECHANICS_EXTRA_INCLUDE_DIRECTORIES
    ${MECHANICS_DIR}
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/hooke/kernels
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2)

  set(MECHANICS_SOURCES
    ${MECHANICS_DIR}/operators/elasticity_operator.cpp
    ${MECHANICS_DIR}/operators/elasticity_jacobian_operator.cpp
    ${MECHANICS_DIR}/preconditioners/elasticity_jacobian_preconditioner.cpp
    ${MECHANICS_DIR}/solvers/quasi_static_elasticity_solver.cpp
  )

  set(MECHANICS_HEADERS
    ${MECHANICS_DIR}/operators/elasticity_operator.hpp
    ${MECHANICS_DIR}/operators/elasticity_jacobian_operator.hpp
    ${MECHANICS_DIR}/preconditioners/elasticity_jacobian_preconditioner.hpp
    ${MECHANICS_DIR}/solvers/quasi_static_elasticity_solver.hpp
  )

  # If CUDA is enabled, tag source files for CUDA compilation
  if (MFEM_USE_CUDA)
      set_source_files_properties(${MECHANICS_SOURCES} PROPERTIES LANGUAGE CUDA)
  endif()

  mfem_add_library(elasticity ${MECHANICS_SOURCES})

  target_include_directories(elasticity PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${TPL_INCLUDE_DIRS}
    ${MECHANICS_EXTRA_INCLUDE_DIRECTORIES})

  target_link_libraries(elasticity
    mfem
    ${TPL_LIBRARIES}
    mfem-common-ecm2)


add_mfem_miniapp(test_beam
    MAIN tests/test_beam.cpp
    LIBRARIES elasticity)

endif () #MFEM_USE_MPI
