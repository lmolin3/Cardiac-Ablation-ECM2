# Parallel apps.
if (MFEM_USE_MPI)
  add_mfem_miniapp(multidomain-dirichlet
    MAIN multidomain-dirichlet.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2)
  add_dependencies(multidomain-dirichlet copy_miniapps_multidomain_test_data)


  add_mfem_miniapp(multidomain-two-domains-rf
    MAIN multidomain-two-domains-rf.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2)  


  add_mfem_miniapp(multidomain-three-domains-celldeath-rf 
    MAIN multidomain-three-domains-celldeath-rf.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2)
    

  add_mfem_miniapp(multidomain-three-domains-celldeath-rf-navier
    MAIN multidomain-three-domains-celldeath-rf-navier.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2 navier-monolithic-ecm2)  

  add_mfem_miniapp(multidomain-one-domain-celldeath-rf-aniso
    MAIN multidomain-one-domain-celldeath-rf-aniso.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2 )
      
endif()

