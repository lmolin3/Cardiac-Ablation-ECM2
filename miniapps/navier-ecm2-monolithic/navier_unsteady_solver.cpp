#include "navier_unsteady_solver.hpp"
#include <fstream>
#include <iomanip>


namespace mfem
{

/// Constructor
NavierUnsteadySolver::NavierUnsteadySolver(ParMesh* mesh,
                                             int uorder,
                                             int porder,
                                             double kin_vis_,
                                             bool verbose):
pmesh(mesh), uorder(uorder), porder(porder), verbose(verbose)
{

   // mesh
   dim=pmesh->Dimension();

   // FE collection and spaces for velocity and pressure
   ufec=new H1_FECollection(uorder,dim);
   pfec=new H1_FECollection(porder);
   ufes=new ParFiniteElementSpace(pmesh,ufec,dim);
   pfes=new ParFiniteElementSpace(pmesh,pfec,1);

   // determine spaces dimension (algebraic tdofs)
   udim = ufes->GetTrueVSize();
   pdim = pfes->GetTrueVSize(); 
   
   // offsets
   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = udim;
   block_offsets[2] = pdim;
   block_offsets.PartialSum();

   // initialize vectors of essential attributes
   vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());      vel_ess_attr=0;
   vel_ess_attr_x.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_x=0;
   vel_ess_attr_y.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_y=0;
   vel_ess_attr_z.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_z=0;
   trac_attr_tmp.SetSize(pmesh->bdr_attributes.Max());    trac_attr_tmp=0;
   custom_traction_attr.SetSize(pmesh->bdr_attributes.Max());    custom_traction_attr=0;
   traction_attr.SetSize(pmesh->bdr_attributes.Max());    traction_attr=0;
   ess_attr_tmp.SetSize(pmesh->bdr_attributes.Max());      ess_attr_tmp=0;

   // initialize GridFunctions
   u_gf      = new ParGridFunction(ufes);      *u_gf = 0.0;
   p_gf      = new ParGridFunction(pfes);      *p_gf = 0.0;
   u_bdf_gf  = new ParGridFunction(ufes);  *u_bdf_gf = 0.0;
   u_ext_gf  = new ParGridFunction(ufes);  *u_ext_gf = 0.0;

   // VectorCoefficient storing extrapolated velocity for Convective term linearization
   u_ext_vc  = new VectorGridFunctionCoefficient(u_ext_gf); 

   // initialize vectors
   x   = new BlockVector(block_offsets);  *x = 0.0;
   rhs = new BlockVector(block_offsets); *rhs = 0.0;

   un     = new Vector(udim);     *un = 0.0; 
   un1    = new Vector(udim);    *un1 = 0.0; 
   un2    = new Vector(udim);    *un2 = 0.0; 
   u_bdf  = new Vector(udim);  *u_bdf = 0.0; 
   u_ext  = new Vector(udim);  *u_ext = 0.0; 
   fv     = new Vector(udim);     *fv = 0.0; 
   fp     = new Vector(pdim);     *fp = 0.0; 

   // set default parameters gamma
   gamma  = 1.0;

   // set kinematic viscosity
   kin_vis = kin_vis_;

   // Print informations
   PrintLogo();
   PrintInfo();

}



/// Public Interface

// Boundary conditions
void NavierUnsteadySolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                  "Duplicate boundary definition detected.");
      if (attr[i] == 1)
      {
         vel_ess_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC (full) to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierUnsteadySolver::AddVelDirichletBC(VecFuncT func, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void NavierUnsteadySolver::AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir)
{
   // Add bc container to list of componentwise velocity bcs
   vel_dbcs_xyz.emplace_back(attr, coeff, dir);

   // Check for duplicate and add attributes for current bc to global list (for that specific component)
   for (int i = 0; i < attr.Size(); ++i)
   {
      switch (dir) {
            case 0: // x 
               dir_string = "x";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i])  && attr[i] ) == 0,
                           "Duplicate boundary definition for x component detected.");
               if (attr[i] == 1){vel_ess_attr_x[i] = 1;}
               break;
            case 1: // y
               dir_string = "y";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_y[i])  && attr[i] ) == 0,
                           "Duplicate boundary definition for y component detected.");
               if (attr[i] == 1){vel_ess_attr_y[i] = 1;}
               break;
            case 2: // z
               dir_string = "z";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_z[i])  && attr[i] ) == 0,
                           "Duplicate boundary definition for z component detected.");
               if (attr[i] == 1){vel_ess_attr_z[i] = 1;}
               break;
            default:;
         }      
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC ( " << dir_string << " component) to boundary attributes: " << std::endl;
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << ", ";
         }
      }
      mfem::out << std::endl;
   }
}


void NavierUnsteadySolver::AddVelDirichletBC(VectorCoefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, ess_attr_tmp);
}

void NavierUnsteadySolver::AddVelDirichletBC(VecFuncT func, int &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void NavierUnsteadySolver::AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, ess_attr_tmp, dir);
}


void NavierUnsteadySolver::AddTractionBC(VectorCoefficient *coeff, Array<int> &attr)
{
   traction_bcs.emplace_back(attr, coeff);

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i] ) && attr[i] ) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
      MFEM_ASSERT( (custom_traction_attr[i] && attr[i]) == 0,
                  "Boundary has already traction enforced.");
      if (attr[i] == 1){traction_attr[i] = 1;}
   }

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Traction (Neumann) BC to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierUnsteadySolver::AddTractionBC(VecFuncT func, Array<int> &attr)
{
   AddTractionBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void NavierUnsteadySolver::AddTractionBC(VectorCoefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   trac_attr_tmp = 0;
   trac_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddTractionBC(coeff, trac_attr_tmp);
}

void NavierUnsteadySolver::AddCustomTractionBC(Coefficient *alpha, ParGridFunction *u, Coefficient *beta, ParGridFunction *p, Array<int> &attr)
{
   custom_traction_bcs.emplace_back(attr, alpha, u, beta, p);

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i] ) && attr[i] ) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
      MFEM_ASSERT(( (traction_attr[i] ) && attr[i]) == 0,
                  "Boundary has already traction enforced.");
      if (attr[i] == 1){custom_traction_attr[i] = 1;}
   }

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Custom Traction (Neumann) BC to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierUnsteadySolver::AddCustomTractionBC(Coefficient *alpha, ParGridFunction *u, Coefficient *beta, ParGridFunction *p, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   trac_attr_tmp = 0;
   trac_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddCustomTractionBC(alpha, u, beta, p, trac_attr_tmp);
}


void NavierUnsteadySolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Acceleration term to domain attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierUnsteadySolver::AddAccelTerm(VecFuncT func, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}


// Solver setup
void NavierUnsteadySolver::SetSolver(SolverParams params)
{
   sParams = params;
}


void NavierUnsteadySolver::SetInitialConditionVel(VectorCoefficient &u_in)
{
   // Project coefficient onto velocity ParGridFunction (predicted and corrected)
   u_gf->ProjectCoefficient(u_in);
   u_gf->GetTrueDofs(x->GetBlock(0));
}

void NavierUnsteadySolver::SetInitialConditionPrevVel(VectorCoefficient &u_in)
{
   // Project coefficient onto velocity ParGridFunction (predicted and corrected)
   ParGridFunction tmp_gf(ufes);         
   tmp_gf.ProjectCoefficient(u_in);
   tmp_gf.GetTrueDofs(*un);
}

void NavierUnsteadySolver::SetInitialConditionPres(Coefficient &p_in)
{
   // Project coefficient onto pressure ParGridFunction (predicted and corrected)
   p_gf->ProjectCoefficient(p_in);
   p_gf->GetTrueDofs(x->GetBlock(1));
}

void NavierUnsteadySolver::Setup(double dt)
{

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "Setup solver (using Full Assembly):" << std::endl;
   }

   sw_setup.Start();


   /// 1. Extract to list of true dofs
   ufes->GetEssentialTrueDofs(vel_ess_attr_x,vel_ess_tdof_x,0);
   ufes->GetEssentialTrueDofs(vel_ess_attr_y,vel_ess_tdof_y,1);
   ufes->GetEssentialTrueDofs(vel_ess_attr_z,vel_ess_tdof_z,2);
   ufes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof_full);
   vel_ess_tdof.Append(vel_ess_tdof_x);
   vel_ess_tdof.Append(vel_ess_tdof_y);
   vel_ess_tdof.Append(vel_ess_tdof_z);
   vel_ess_tdof.Append(vel_ess_tdof_full);


   /// 2. Setup and assemble bilinear forms 
   int skip_zeros = 0;

   // Velocity laplacian K (not scaled by viscosity)
   K_form = new ParBilinearForm(ufes);  
   K_form->AddDomainIntegrator(new VectorDiffusionIntegrator());
   K_form->Assemble(skip_zeros); K_form->Finalize(skip_zeros);
   K = K_form->ParallelAssemble();

   // Velocity mass
   M_form = new ParBilinearForm(ufes);  
   M_form->AddDomainIntegrator(new VectorMassIntegrator());
   M_form->Assemble(skip_zeros); M_form->Finalize(skip_zeros);
   M = M_form->ParallelAssemble();

   // Divergence
   D_form = new ParMixedBilinearForm(ufes, pfes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator());
   D_form->Assemble(); D_form->Finalize();
   D = D_form->ParallelAssemble();
   *(D) *= -1.0; 
   De = D->EliminateCols(vel_ess_tdof);

   // Gradient
   G = D->Transpose();

   /// 3. Assemble linear form for rhs
   f_form = new ParLinearForm(ufes);
   // Adding forcing terms
   for (auto &accel_term : accel_terms)
   {
      f_form->AddDomainIntegrator( new VectorDomainLFIntegrator( *(accel_term.coeff) ), accel_term.attr );  
   }
   // Adding traction bcs
   for (auto &traction_bc : traction_bcs)
   {
      f_form->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator( *(traction_bc.coeff)) , traction_bc.attr);
   }

   // Adding custom traction bcs
   for (auto &traction_bc : custom_traction_bcs)
   {
      f_form->AddBoundaryIntegrator(new VectorNeumannLFIntegrator( *(traction_bc.u),*(traction_bc.p),*(traction_bc.alpha),*(traction_bc.beta)) , traction_bc.attr);
   }

   // Update time in Dirichlet velocity coefficients and project on predicted/corrected vectors and gf
   UpdateTimeBCS( 0.0 );
   updatedBCS = false;
   
   // Update time in coefficients and assemble vector fv
   UpdateTimeRHS( 0.0 );
   updatedRHS = false;


   /// 4. Construct the operators for preconditioners
   //
   //     Here we use BoomerAMG for:
   //     * convective-diffusive part A =  alpha/dt M + K + C,
   //     * velocity mass matrix M
   //     * pressure laplacian (for schur complement) 
   //

   // Pressure Schur Complement ( BoomerAMG with operator S = 1/kin_vis Mp + dt Sp) --> NOTE: assuming constant dt and viscosity, otherwise this neds to be updated
   S = new HypreParMatrix();
   S_form = new ParBilinearForm(pfes);
   S_visccoeff.constant = 1.0 / kin_vis;
   S_bdfcoeff.constant = dt ;
   //S_form->AddDomainIntegrator(new MassIntegrator( S_visccoeff ));
   S_form->AddDomainIntegrator(new DiffusionIntegrator( S_bdfcoeff ));
   S_form->Assemble(); S_form->Finalize(); 
   S = S_form->ParallelAssemble();
   invS = new HypreBoomerAMG(*S);    
   invS->SetPrintLevel(0);
   invS->SetSystemsOptions(dim);
   invS->iterative_mode = false;

   // Velocity Prediction (BoomerAMG; operator will be assigned inside loop)
   invC = new HypreBoomerAMG();
   invC->SetPrintLevel(0);
   invC->SetSystemsOptions(dim);
   //invC_pc->SetElasticityOptions(ufes);
   //invC_pc->SetAdvectiveOptions(1, "", "FA"); // AIR solver
   invC->iterative_mode = false;


   /// 5. Construct solver and preconditioner
   //
   //     Here we use:
   //     * GMRES for solving the linear system;
   //     * BlockDiagonal preconditioner ;
   //

   // NS operator
   nsOp = new BlockOperator(block_offsets);
   nsOp->SetBlock(0, 1, G);
   nsOp->SetBlock(1, 0, D);

   // Preconditioner
   nsPrec = new BlockDiagonalPreconditioner(block_offsets);
   nsPrec->SetDiagonalBlock(1, invS);

   // Solver
   nsSolver = new GMRESSolver(pmesh->GetComm());
   nsSolver->iterative_mode = false;
   nsSolver->SetAbsTol(sParams.atol);
   nsSolver->SetRelTol(sParams.rtol);
   nsSolver->SetMaxIter(sParams.maxIter);
   nsSolver->SetOperator(*nsOp);
   nsSolver->SetPreconditioner(*nsPrec);
   nsSolver->SetPrintLevel(sParams.pl);

   sw_setup.Stop();


#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "setup", 0);} // Export matrices/vectors after setup
#endif

}


void NavierUnsteadySolver::Step(double &time, double dt, int current_step)
{
   /// 0.1 Update BDF time integration coefficients
   SetTimeIntegrationCoefficients( current_step );

   /// 0.2 Update time coefficients for rhs and bcs
   time += dt;
   UpdateTimeBCS( time );
   UpdateTimeRHS( time );

   /// 0.3 Assemble Convective term (linearized: u_ext \cdot \grad u)

   sw_conv_assembly.Start();

   // Extrapolate velocity    u_ext = b1 un + b2 u_{n-1} + b3 u_{n-2}
   add(b1, *un, b2, *un1, *u_ext);
   u_ext->Add(b3,*un2);
   u_ext_gf->SetFromTrueDofs(*u_ext);  

   int skip_zeros = 0;
   delete NL_form; NL_form = nullptr;
   delete NL; NL = nullptr;
   delete A; A = nullptr;
   delete C; C = nullptr;
   delete Ce; Ce = nullptr;
   NL_form = new ParBilinearForm(ufes);
   NL_form->AddDomainIntegrator(new VectorConvectionIntegrator(*u_ext_vc, 1.0));  // += C 
   NL_form->Assemble(skip_zeros); NL_form->Finalize(skip_zeros);
   NL = NL_form->ParallelAssemble();  
   
   C_visccoeff.constant = kin_vis;
   C_bdfcoeff.constant = alpha / dt;

   A = Add(C_visccoeff.constant, *K, 1.0, *NL); // A = kin_vis K + NL
   C = Add(C_bdfcoeff.constant, *M, 1.0, *A);   // C = alpha/dt M + A

   Ce = C->EliminateRowsCols(vel_ess_tdof);


   // Update block operator and block preconditioner
   invC->SetOperator(*C);     
   nsOp->SetBlock(0, 0, C);
   nsPrec->SetDiagonalBlock(0, invC);

   sw_conv_assembly.Stop();

#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "conv_assembly", current_step);} // Export matrices/vectors
#endif

   /// 1. Solve 

   sw_solve.Start();

   // Assemble rhs     fv + 1/dt u_bdf
   add(a1, *un, a2, *un1, *u_bdf);     
   u_bdf->Add(a3,*un2);
   M->AddMult(*u_bdf,rhs->GetBlock(0),1.0/dt);

   // Apply bcs
   C->EliminateBC(*Ce, vel_ess_tdof, x->GetBlock(0), rhs->GetBlock(0)); // rhs_v -= Ce*u

   // Solve current iteration.
   nsSolver->Mult(*rhs, *x);
   iter_solve = nsSolver->GetNumIterations();
   res_solve = nsSolver->GetFinalNorm();

   // Update gfs 
   u_gf->SetFromTrueDofs(x->GetBlock(0));
   p_gf->SetFromTrueDofs(x->GetBlock(1)); // Remove nullspace by removing mean of the pressure solution 
   if (traction_bcs.empty() && custom_traction_bcs.empty()) // TODO: change, since homogeneous neumann are not included in the list (include them in a vector like homogeneous_neumann.empty(). Can be done at setup time )
   {
      MeanZero(*p_gf);
      p_gf->GetTrueDofs(x->GetBlock(1));
   }
   sw_solve.Stop();

#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "solve", current_step);} // Export matrices/vectors
#endif

   // 5. Relaxation (velocity)  u = gamma u + (1 - gamma) un
   add(gamma,x->GetBlock(0),(1.0-gamma),*un,x->GetBlock(0));
   u_gf->SetFromTrueDofs(x->GetBlock(0));


   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "Step" << std::setw(10) << "Time" << std::setw(15) << "dt" << std::setw(10) << "It" 
                << std::setw(15) << "Resid" << std::setw(15) << "Reltol" << std::endl;
      mfem::out << std::setw(9) << current_step << std::scientific << std::setprecision(5) << std::setw(15) << time 
                << std::setw(15) << dt << std::setw(6) << iter_solve << std::scientific << std::setprecision(5) 
               << std::setw(15) << res_solve << std::setw(15) << sParams.rtol << std::endl;
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }

   // Update solution at previous timesteps and time
   UpdateSolution(); 
   updatedRHS = false;
   updatedBCS = false;

}


/// Private Interface

void NavierUnsteadySolver::SetTimeIntegrationCoefficients(int step)
{
   // Maximum BDF order to use at current time step
   // step + 1 <= order <= max_bdf_order
   int bdf_order = std::min(step + 1, max_bdf_order);


   if (step == 0 && bdf_order == 1)
   {
      alpha = 1.0;
      a1 = 1.0; 
      a2 = 0.0; 
      a3 = 0.0; 
      b1 = 1.0; 
      b2 = 0.0; 
      b3 = 0.0; 
   }
   else if (step >= 1 && bdf_order == 2)
   {
      alpha = 3.0/2.0;
      a1 = 2.0; 
      a2 = -1.0/2.0; 
      a3 = 0.0; 
      b1 = 2.0;  
      b2 = -1.0; 
      b3 = 0.0;  
   }
   else if (step >= 2 && bdf_order == 3)
   {
      alpha = 11.0/6.0;
      a1 = 3.0; 
      a2 = -3.0/2.0; 
      a3 = 1.0/3.0; 
      b1 = 3.0;
      b2 = -3.0;
      b3 = 1.0;
   }

}


void NavierUnsteadySolver::UpdateSolution()
{
   // Update solution at previous timesteps for BDF
   *un2 = *un1;
   *un1 = *un;
   *un  = x->GetBlock(0);
}

void NavierUnsteadySolver::UpdateTimeRHS( double new_time )
{
   if ( !updatedRHS )
   {
      *fv = 0.0;   
      *fp = 0.0;

      // Update acceleration terms
      for (auto &accel_term : accel_terms)
      {
         accel_term.coeff->SetTime(new_time);
      }

      // Update traction bcs
      for (auto &traction_bc : traction_bcs)
      {
         traction_bc.coeff->SetTime(new_time);
      }

      // Update custom traction bcs
      for (auto &traction_bc : custom_traction_bcs)
      {
         traction_bc.alpha->SetTime(new_time);
         traction_bc.beta->SetTime(new_time);
      }

      f_form->Assemble(); 
      f_form->ParallelAssemble(*fv); 


      // Update pressure block of rhs (bcs from velocity)
      De->AddMult(x->GetBlock(0), *fp, -1.0);
      
      // Assign blocks of rhs vector
      rhs->GetBlock(0) = *fv;
      rhs->GetBlock(1) = *fp;

      updatedRHS = true;
   }
}

void NavierUnsteadySolver::UpdateTimeBCS( double new_time )
{
   if ( !updatedBCS )
   {

      // Projection of coeffs (full velocity applied)
      for (auto &vel_dbc : vel_dbcs)
      {
         vel_dbc.coeff->SetTime(new_time);
         u_gf->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
      }
      u_gf->GetTrueDofs(x->GetBlock(0));

      // Projection of coeffs (velocity component applied)
      ParGridFunction tmp_gf(ufes);        // temporary velocity gf for projection
      Vector          tmp_vec(udim);       // temporary velocity vector for projection
      Array<int>      tmp_tdofs;
      for (auto &vel_dbc : vel_dbcs_xyz)
      {
         vel_dbc.coeff->SetTime(new_time);
         VectorArrayCoefficient tmp_coeff(dim);                           // Set coefficient with right component
         tmp_coeff.Set(vel_dbc.dir, vel_dbc.coeff, false);
         tmp_gf.ProjectBdrCoefficient(tmp_coeff, vel_dbc.attr);           // Project on dummy gf
         tmp_gf.GetTrueDofs(tmp_vec);

         ufes->GetEssentialTrueDofs(vel_dbc.attr,tmp_tdofs,vel_dbc.dir);  // Update solution dofs
         for(int i=0;i<tmp_tdofs.Size();i++)
         {
            (x->GetBlock(0))[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
         }      
      }
      // Initialize solution gf with vector containing projected coefficients 
      // and update grid function and vector for provisional velocity
      u_gf->SetFromTrueDofs(x->GetBlock(0));

      updatedBCS = true;
   }
}

void NavierUnsteadySolver::MeanZero(ParGridFunction &gf)
{
   // Make sure not to recompute the inner product linear form every
   // application.
   if (mass_lf == nullptr)
   {
      ConstantCoefficient onecoeff(1.0);
      mass_lf = new ParLinearForm(gf.ParFESpace());
      auto *dlfi = new DomainLFIntegrator(onecoeff);
      mass_lf->AddDomainIntegrator(dlfi);
      mass_lf->Assemble();

      ParGridFunction one_gf(gf.ParFESpace());
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   }

   double integ = mass_lf->operator()(gf);

   gf -= integ / volume;
}

void NavierUnsteadySolver::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, ufes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, ufes->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}

void NavierUnsteadySolver::PrintLogo()
{
   int fes_sizeVel = ufes->GlobalVSize();
   int fes_sizePres = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {

      mfem::out << "  _______   ________  _____ ______     _______          ________   ________  ___      ___ ___  _______   ________     " << std::endl
               << " |\\  ___ \\ |\\   ____\\|\\   _ \\  _   \\  /  ___  \\        |\\   ___  \\|\\   __  \\|\\  \\    /  /|\\  \\|\\  ___ \\ |\\   __  \\    " << std::endl
               << " \\ \\   __/|\\ \\  \\___|\\ \\  \\\\\\__\\ \\  \\/__/|_/  /|       \\ \\  \\\\ \\  \\ \\  \\|\\  \\ \\  \\  /  / | \\  \\ \\   __/|\\ \\  \\|\\  \\ " << std::endl  
               << "  \\ \\  \\_|/_\\ \\  \\    \\ \\  \\\\|__| \\  \\__|//  / /        \\ \\  \\\\ \\  \\ \\   __  \\ \\  \\/  / / \\ \\  \\ \\  \\_|/_\\ \\   _  _\\" << std::endl  
               << "   \\ \\  \\_|\\ \\ \\  \\____\\ \\  \\    \\ \\  \\  /  /_/__        \\ \\  \\\\ \\  \\ \\  \\ \\  \\ \\    / /   \\ \\  \\ \\  \\_|\\ \\ \\  \\\\  \\|" << std::endl 
               << "    \\ \\_______\\ \\_______\\ \\__\\    \\ \\__\\|\\________\\       \\ \\__\\\\ \\__\\ \\__\\ \\__\\ \\__/ /     \\ \\__\\ \\_______\\ \\__\\\\ _\\" << std::endl 
               << "     \\|_______|\\|_______|\\|__|     \\|__| \\|_______|        \\|__| \\|__|\\|__|\\|__|\\|__|/       \\|__|\\|_______|\\|__|\\|__|" << std::endl;
      mfem::out << std::endl;

   }
}

void NavierUnsteadySolver::PrintInfo()
{
   int fes_sizeVel = ufes->GlobalVSize();
   int fes_sizePres = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NAVIER version: " << MFEM_NAVIER_UNSTEADY_VERSION << std::endl
               << "MFEM version: " << MFEM_VERSION << std::endl
               << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
               << "Velocity #DOFs: " << fes_sizeVel << std::endl
               << "Pressure #DOFs: " << fes_sizePres << std::endl;
      mfem::out << std::endl;

   }
}

void NavierUnsteadySolver::PrintTimingData()
{
   double my_rt[6], rt_max[6];

   my_rt[0] = sw_setup.RealTime();
   my_rt[1] = sw_conv_assembly.RealTime();
   my_rt[2] = sw_solve.RealTime();

   MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, ufes->GetComm());

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "CONV-ASS"
                << std::setw(10) << "SOLVE" << "\n";
      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2] << "\n";
      mfem::out << std::setprecision(8);
   }
}

#ifdef MFEM_DEBUG

void NavierUnsteadySolver::PrintMatricesVectors( const char* id, int num )
{
   // Create folder
   std::string folderName(outfolder);
   folderName += "/MatVecs_iter";
   folderName += std::to_string(num);

   if (mkdir(folderName.c_str(), 0777) == -1) {} //{mfem::err << "Error :  " << strerror(errno) << std::endl;}

   //Create files
   //std::ofstream K_file(std::string(folderName) + '/' + "K_" + std::string(id) + ".dat");
   std::ofstream M_file(std::string(folderName) + '/' + "M_" + std::string(id) + ".dat");
   std::ofstream A_file(std::string(folderName) + '/' + "A_" + std::string(id) + ".dat");
   std::ofstream D_file(std::string(folderName) + '/' + "D_" + std::string(id) + ".dat");
   std::ofstream G_file(std::string(folderName) + '/' + "G_" + std::string(id) + ".dat");

   std::ofstream De_file(std::string(folderName) + '/' + "De_" + std::string(id) + ".dat");
   std::ofstream Ce_file(std::string(folderName) + '/' + "Ce_" + std::string(id) + ".dat");

   std::ofstream u_file(std::string(folderName) + '/' + "u_" + std::string(id) + ".dat");
   std::ofstream p_file(std::string(folderName) + '/' + "p_" + std::string(id) + ".dat");
   std::ofstream un_file(std::string(folderName) + '/' + "un_" + std::string(id) + ".dat");
   std::ofstream un1_file(std::string(folderName) + '/' + "un1_" + std::string(id) + ".dat");
   std::ofstream un2_file(std::string(folderName) + '/' + "un2_" + std::string(id) + ".dat");
   std::ofstream uext_file(std::string(folderName) + '/' + "uext_" + std::string(id) + ".dat");
   std::ofstream ubdf_file(std::string(folderName) + '/' + "ubdf_" + std::string(id) + ".dat");

   std::ofstream fv_file(std::string(folderName) + '/' + "fv_" + std::string(id) + ".dat");
   std::ofstream fp_file(std::string(folderName) + '/' + "fp_" + std::string(id) + ".dat");
   std::ofstream rhs_file(std::string(folderName) + '/' + "rhs_v1_" + std::string(id) + ".dat");

   std::ofstream dofs_file(std::string(folderName) + '/' + "dofs_" + std::string(id) + ".dat");


   // Print matrices in matlab format
   M->PrintMatlab(M_file);

   if(A==nullptr)
   {
      A = new HypreParMatrix();
      A->PrintMatlab(A_file);
      delete A; A = nullptr;
   }
   else
   {
      A->PrintMatlab(A_file);
   }

   D->PrintMatlab(D_file);
   G->PrintMatlab(G_file);

   if(Ce==nullptr)
   {
      Ce = new HypreParMatrix();
      Ce->PrintMatlab(Ce_file);
      delete Ce; Ce = nullptr;
   }
   else
   {
      Ce->PrintMatlab(Ce_file);
   }
   

   // Print Vectors
   (x->GetBlock(0)).Print(u_file,1);
   (x->GetBlock(1)).Print(p_file,1);
   un->Print(un_file, 1);
   un1->Print(un1_file, 1);
   un2->Print(un2_file, 1);
   u_ext->Print(uext_file, 1);
   u_bdf->Print(ubdf_file, 1);
   fv->Print(fv_file,1);
   (rhs->GetBlock(1)).Print(fp_file,1);
   rhs->Print(rhs_file,1);
   fp->Print(fp_file, 1);


   for (int i = 0; i < vel_ess_tdof.Size(); ++i)
   {
      dofs_file << vel_ess_tdof[i] << std::endl;
   }
   dofs_file.close();

}

#endif

/// Destructor
NavierUnsteadySolver::~NavierUnsteadySolver()
{
   delete ufec; ufec = nullptr;
   delete pfec; pfec = nullptr;
   delete ufes; ufes = nullptr;
   delete pfes; pfes = nullptr;
   //delete pmesh; pmesh = nullptr;

   delete K_form;   K_form = nullptr;
   delete M_form;   M_form = nullptr;
   delete D_form;   D_form = nullptr;
   delete NL_form;  NL_form = nullptr;
   delete S_form;   S_form = nullptr;
   delete f_form;   f_form = nullptr;

   delete u_gf;           u_gf = nullptr;
   delete p_gf;           p_gf = nullptr;
   delete u_bdf_gf;   u_bdf_gf = nullptr;
   delete u_ext_gf;   u_ext_gf = nullptr;

   delete u_bdf;     u_bdf = nullptr;
   delete u_ext;     u_ext = nullptr;
   delete un;           un = nullptr;
   delete un1;         un1 = nullptr;
   delete un2;         un2 = nullptr;
   delete fv;           fv = nullptr;
   delete fp;           fp = nullptr;

   delete x;           x = nullptr;
   delete rhs;       rhs = nullptr;

   delete invC;      invC = nullptr;
   delete invS;      invS = nullptr;
   delete nsOp;      nsOp = nullptr;
   delete nsPrec;  nsPrec = nullptr;

   delete M;         M = nullptr;
   delete K;         K = nullptr;
   delete C;         C = nullptr;
   delete NL;        NL = nullptr;
   delete D;         D = nullptr;
   delete G;         G = nullptr;
   delete A;         A = nullptr;
   delete S;         S = nullptr;    
   delete Ce;        Ce = nullptr;
   delete De;        De = nullptr;
   delete mass_lf;   mass_lf = nullptr;
}


// Linear Integrator for Neumann BC
void VectorNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = U -> FESpace() -> GetMesh() -> SpaceDimension();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);
   nor.SetSize(vdim);
   pn.SetSize(vdim);
   gradUn.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);

      // Compute normal and normalize
      CalcOrtho(Tr.Jacobian(), nor);
      nor /= nor.Norml2();

      // Compute pn
      pn = nor;
      double beta = Q2.Eval(Tr, ip);
      double pval = P->GetValue(Tr, ip);
      pn *=  beta * pval; ;

      // Compute Q1 * n.grad(u)
      U->GetVectorGradient(Tr, gradU);
      gradU.Mult(nor,gradUn);
      double alpha = Q1.Eval(Tr, ip);
      gradUn *= alpha;

      // Compute vec = Q1 n.grad(u) + Q2 pn
      add(gradUn, pn, vec);

      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
   }
}

void VectorNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("VectorNeumannLFIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as face integrator!\n"
              "  Use LinearForm::AddBoundaryIntegrator instead of\n"
              "  LinearForm::AddBdrFaceIntegrator.");
}


}
