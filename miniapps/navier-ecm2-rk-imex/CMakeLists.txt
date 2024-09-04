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
  add_subdirectory(lib)

  set(NAVIER_DIR ${CMAKE_SOURCE_DIR}/miniapps/navier-ecm2-rk-imex)
  set(NAVIER_EXTRA_INCLUDE_DIRECTORIES
    ${NAVIER_DIR}
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2)

  add_library(navier_rk
    ${NAVIER_DIR}/lib/navierstokes_srk.cpp
    ${NAVIER_DIR}/integrators/custom_bilinteg.cpp
    ${NAVIER_DIR}/lib/navierstokes_preconditioners.cpp
    ${NAVIER_DIR}/bc/navierstokes_bchandler.cpp
    ${NAVIER_DIR}/lib/navierstokes_residual.cpp
    ${NAVIER_DIR}/lib/navierstokes_operator.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2/utils.cpp)


  target_include_directories(navier_rk PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${TPL_INCLUDE_DIRS}
    ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

    
  target_link_libraries(navier_rk
    mfem
    ${TPL_LIBRARIES})

  # Create a symbolic link for meshes
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh ${CMAKE_CURRENT_BINARY_DIR}/mesh
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_mfem_miniapp(navier_ecm2_mms_rk
    MAIN tests/navier_mms.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier_rk)

  target_include_directories(navier_ecm2_mms_rk
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

endif () #MFEM_USE_MPI

#  if (MFEM_ENABLE_TESTING)
#    foreach (test "mms" )
#      add_test(NAME navier_${test}_np${MFEM_MPI_NP}
#        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
#        ${MPIEXEC_PREFLAGS}
#        $<TARGET_FILE:navier_${test}> -cr -no-vis
#        ${MPIEXEC_POSTFLAGS})
#    endforeach()
# endif() #MFEM_ENABLE_TESTING

