if (MFEM_USE_MPI)

    list(APPEND NAVIER_MONO_COMMON_SOURCES
    lib/navier_solver.cpp
    lib/navier_preconditioners.cpp
    lib/pressure_correction.cpp
    lib/schur_preconditioners.cpp
    lib/time_adaptivity.cpp
    bc/navier_bchandler.cpp
    integrators/custom_bilinear_integrators.cpp
    integrators/custom_linear_integrators.cpp
  )

  list(APPEND NAVIER_MONO_COMMON_HEADERS
    lib/navier_solver.hpp
    lib/navier_types.hpp
    lib/navier_preconditioners.hpp
    lib/pressure_correction.hpp
    lib/schur_preconditioners.hpp
    lib/time_adaptivity.hpp
    lib/navier_qoi.hpp
    bc/navier_bchandler.hpp
    integrators/custom_bilinear_integrators.hpp
    integrators/custom_linear_integrators.hpp
  )

  convert_filenames_to_full_paths(NAVIER_MONO_COMMON_SOURCES)
  convert_filenames_to_full_paths(NAVIER_MONO_COMMON_HEADERS)

  mfem_add_library(navier-monolithic-ecm2 ${NAVIER_MONO_COMMON_SOURCES} ${NAVIER_MONO_COMMON_HEADERS})

    
  target_link_libraries(navier-monolithic-ecm2 mfem mfem-common mfem-common-ecm2)

  add_mfem_miniapp(navier-poiseulle-monolithic
    MAIN tests/navier_poiseulle.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)

  add_mfem_miniapp(navier-mms-monolithic
    MAIN tests/navier_mms.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)

  add_mfem_miniapp(navier-convergence-monolithic
    MAIN tests/navier_convergence.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)

  add_mfem_miniapp(navier-time-convergence-monolithic
    MAIN tests/navier_time_convergence.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)

  add_mfem_miniapp(navier-flowaroundcyl-monolithic
    MAIN tests/navier_flowaroundcyl.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)
    
  add_mfem_miniapp(navier-liddriven-monolithic
    MAIN tests/navier_liddriven.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)

  add_mfem_miniapp(navier-backstep-monolithic
    MAIN tests/navier_backstep.cpp
    ${NAVIER_MONO_COMMON_SOURCES}
    LIBRARIES navier-monolithic-ecm2)
      
  add_custom_command(TARGET navier-poiseulle-monolithic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/Mesh Mesh
            COMMENT "Copying the navier-ecm2 Mesh data directory ...")


  add_mfem_miniapp(test-grad-assembly
    MAIN tests/test_grad_assembly.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem)


endif () #MFEM_USE_MPI
