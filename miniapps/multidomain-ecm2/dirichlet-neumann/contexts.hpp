/**
 * @file contexts.hpp
 * @brief File containing structs with parameters for the multiphysics and domain decomposition.
 */


#pragma once

#include "mfem.hpp"

#ifdef MFEM_NAVIER_UNSTEADY_HPP
#include "lib/navier_solver.hpp"
#endif

#ifndef CONTEXTS_HPP
#define CONTEXTS_HPP

using namespace mfem;

struct s_RFContext
{
   // Physical parameters 
   real_t sigma_fluid = 1.0;
   real_t sigma_solid = 0.54;
   real_t sigma_cylinder = 1.0;
   real_t phi_applied = 10.0;
   real_t phi_gnd = 0.0;
   real_t aniso_ratio = 1.0;
    // FE
    int order = 1;
    // Solver
    bool pa = false; // Enable partial assembly
} RF_ctx;

struct s_HeatContext
{   
    // Physical parameters  
    real_t T_solid =37.0;      // Solid temperature, in Celsius 
    real_t T_fluid = 37.0;      // Fluid temperature, in Celsius
    real_t T_cylinder = 37.0;   // Cylinder temperature, in Celsius
    real_t k_fluid = 1.0;       // Fluid thermal conductivity
    real_t k_solid = 1.0;       // Solid thermal conductivity
    real_t k_cylinder = 1.0;    // Cylinder thermal conductivity
    real_t c_fluid = 1.0;       // Fluid specific heat
    real_t c_solid = 1.0;       // Solid specific heat
    real_t c_cylinder = 1.0;    // Cylinder specific heat
    real_t rho_fluid = 1.0;     // Fluid density
    real_t rho_solid = 1.0;     // Solid density
    real_t rho_cylinder = 1.0;  // Cylinder density
    real_t aniso_ratio = 1.0;   // Anisotropy ratio
    // FE
    int order = 1;
    // Solver
    bool pa = false; // Enable partial assembly
    int ode_solver_type = 1; // 1 = Backward Euler, 2 = SDIRK2, 3 = SDIRK3, 4 = Implicit Midpoint, 5 = SDIRK23, 6 = SDIRK34, 7 = Forward Euler, 8 = RK2, 9 = RK3 SSP, 10 = RK4
} Heat_ctx;



struct s_CellDeathContext
{
    // Physical parameters
    real_t A1 = 3.68*1e30;
    real_t A2 = 5.68*1e3;
    real_t A3 = 2.58*1e5;
    real_t deltaE1 = 210*1e3;
    real_t deltaE2 = 38.6*1e3;
    real_t deltaE3 = 47.2*1e3;
    // FE
    int order = -1;
    // Solver
    int solver_type = 0; // 0 = Eigen, 1 = Gotran
} CellDeath_ctx;

#ifdef MFEM_NAVIER_UNSTEADY_HPP
struct s_NavierContext // Navier Stokes params
{
   int uorder = 2;
   int porder = 1;
   real_t kinvis = 1.0;
   real_t u_inflow =  0.1;
   bool verbose = true;
   int bdf = 3;
   // int splitting_type = 0;  // 0 = Chorin-Temam, 1 = Yosida, 2 = High-Order Yosida 
   // int correction_order = 1; // Correction order for High-Order Yosida   
   navier::BlockPreconditionerType pc_type = navier::BlockPreconditionerType::BLOCK_DIAGONAL;       // 0: Block Diagonal, 1: BlowLowerTri, 2: BlockUpperTri, 3: Chorin-Temam, 4: Yosida, 5: Chorin-Temam Pressure Corrected, 6: Yosida Pressure Corrected
   navier::SchurPreconditionerType schur_pc_type = navier::SchurPreconditionerType::APPROXIMATE_DISCRETE_LAPLACIAN; // 0: Pressure Mass, 1: Pressure Laplacian, 2: PCD, 3: Cahouet-Chabard, 4: LSC, 5: Approximate Inverse   
   navier::TimeAdaptivityType time_adaptivity_type = navier::TimeAdaptivityType::NONE; // Time adaptivity type (NONE, CFL, HOPC)
   int pressure_correction_order = 2; // Order of the pressure correction
   bool mass_lumping = false; // Mass lumping
   bool stiff_strain = false; // Stiff strain

   static void NoSlip(const Vector &x, real_t t, Vector &u)
   {
      u = 0.0;
   }

} Navier_ctx;
#endif

struct s_DomainDecompositionContext
{
   // Relaxation parameter
   real_t omega_heat = 0.8;    // General relaxation parameter for heat transfer problem (applies to all domains)
   real_t omega_heat_fluid = 0.8;
   real_t omega_heat_solid = 0.8;
   real_t omega_heat_cyl = 0.8;
   real_t omega_rf = 0.8;      // General relaxation parameter for RF problem (applies to all domains)
   real_t omega_rf_fluid = 0.8; 
   real_t omega_rf_solid = 0.8; 
   real_t omega_rf_cyl = 0.8; 
} DD_ctx;


struct s_MeshContext
{
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool hex = false;
} Mesh_ctx;


struct s_SimulationContext
{
   real_t t_final = 1.0;
   real_t dt = 1.0e-2;
   bool print_timing = false;
   bool paraview = false;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "./Output/Test";
   bool save_convergence = false;
} Sim_ctx;

#endif // CONTEXTS_HPP
