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

  set(EP_DIR ${CMAKE_SOURCE_DIR}/miniapps/electrophysiology-ecm2)
  set(EP_EXTRA_INCLUDE_DIRECTORIES
    ${EP_DIR}
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2)

  set(EP_SOURCES
    ${EP_DIR}/lib/monodomain_solver.cpp
    ${EP_DIR}/lib/implicit_solver.cpp
    ${EP_DIR}/lib/reaction_solver.cpp
    ${EP_DIR}/bc/ep_bchandler.cpp
  )

  set(EP_HEADERS
    ${EP_DIR}/lib/monodomain_solver.hpp
    ${EP_DIR}/lib/implicit_solvers.hpp
    ${EP_DIR}/lib/reaction_solver.hpp
    ${EP_DIR}/bc/ep_bchandler.hpp
    ${EP_DIR}/ionic_models/gotranx_wrapper.hpp
    ${EP_DIR}/ionic_models/mitchell_schaeffer_2003.h
  )

  # If CUDA is enabled, tag source files for CUDA compilation
  if (MFEM_USE_CUDA)
      set_source_files_properties(${EP_SOURCES} PROPERTIES LANGUAGE CUDA)
  endif()

  mfem_add_library(electrophysiology ${EP_SOURCES})

  target_include_directories(electrophysiology PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${TPL_INCLUDE_DIRS}
    ${EP_EXTRA_INCLUDE_DIRECTORIES})

  target_link_libraries(electrophysiology
    mfem
    ${TPL_LIBRARIES}
    mfem-common-ecm2)


add_mfem_miniapp(test_monodomain
    MAIN tests/test_monodomain.cpp
    LIBRARIES electrophysiology)

add_mfem_miniapp(test_stimulation
    MAIN tests/test_stimulation.cpp
    LIBRARIES electrophysiology)

endif () #MFEM_USE_MPI