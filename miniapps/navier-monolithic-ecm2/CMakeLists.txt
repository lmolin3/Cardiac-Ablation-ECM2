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

  set(NAVIER_MONO_DIR ${CMAKE_SOURCE_DIR}/miniapps/navier-monolithic-ecm2)

  include_directories(
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${NAVIER_MONO_DIR}
  )

    list(APPEND NAVIER_MONO_SRCS
    lib/navier_solver.cpp
    lib/navier_preconditioners.cpp
    lib/pressure_correction.cpp
    lib/schur_preconditioners.cpp
    bc/navier_bchandler.cpp
    integrators/custom_bilinear_integrators.cpp
    integrators/custom_linear_integrators.cpp
  )

  list(APPEND NAVIER_MONO_HDRS
    lib/navier_solver.hpp
    lib/navier_preconditioners.hpp
    lib/pressure_correction.hpp
    lib/schur_preconditioners.hpp
    lib/navier_qoi.hpp
    bc/navier_bchandler.hpp
    integrators/custom_bilinear_integrators.hpp
    integrators/custom_linear_integrators.hpp
  )

  mfem_add_library(navier_monolithic_ecm2 ${NAVIER_MONO_SRCS} ${NAVIER_MONO_HDRS})

    
  target_link_libraries(navier_monolithic_ecm2
    mfem
    mfem-common
    mfem-common-ecm2
    ${TPL_LIBRARIES})

  add_mfem_miniapp(navier-poiseulle-monolithic
    MAIN tests/navier_poiseulle.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_monolithic_ecm2)

  add_mfem_miniapp(navier-mms-monolithic
    MAIN tests/navier_mms.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_monolithic_ecm2)

  add_mfem_miniapp(navier-convergence-monolithic
    MAIN tests/navier_convergence.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_monolithic_ecm2)

  add_mfem_miniapp(navier-flowaroundcyl-monolithic
    MAIN tests/navier_flowaroundcyl.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_monolithic_ecm2)

  add_mfem_miniapp(navier-liddriven-monolithic
    MAIN tests/navier_liddriven.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_monolithic_ecm2)

  add_mfem_miniapp(navier-backstep-monolithic
    MAIN tests/navier_backstep.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_monolithic_ecm2)
      
  add_custom_command(TARGET navier-poiseulle-monolithic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/Mesh Mesh
            COMMENT "Copying the navier-ecm2 Mesh data directory ...")


  add_mfem_miniapp(test-grad-assembly
    MAIN tests/test_grad_assembly.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem)

  #target_include_directories(navier_test2D
  #  PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})


endif () #MFEM_USE_MPI
