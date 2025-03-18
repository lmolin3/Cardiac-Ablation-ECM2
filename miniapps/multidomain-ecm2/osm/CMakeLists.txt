# Parallel apps.
if (MFEM_USE_MPI)

  add_mfem_miniapp(multidomain-two-domains-rf-osm
    MAIN multidomain-two-domains-rf.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 electrostatics_ecm2)  
  
  add_mfem_miniapp(multidomain-three-domains-celldeath-rf-navier-osm
    MAIN multidomain-three-domains-celldeath-rf-navier.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2 navier-monolithic-ecm2)    

    
endif()

