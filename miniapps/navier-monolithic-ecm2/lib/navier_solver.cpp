#include "navier_solver.hpp"
#include <fstream>
#include <iomanip>


namespace mfem{

namespace navier{

/// Constructor
MonolithicNavierSolver::MonolithicNavierSolver(std::shared_ptr<ParMesh> pmesh_,
                                           BCHandler *bcs_,
                                           real_t kin_vis_,
                                           int uorder_,
                                           int porder_,
                                           bool verbose_) : pmesh(pmesh_), bcs(bcs_), uorder(uorder_), porder(porder_), verbose(verbose_)
{

   // pmesh
   sdim = pmesh->Dimension();

   // FE collection and spaces for velocity and pressure
   ufes = new H1_ParFESpace(pmesh.get(), uorder, sdim, BasisType::GaussLobatto, sdim);
   pfes = new H1_ParFESpace(pmesh.get(), porder, sdim, BasisType::GaussLobatto);

   // determine spaces dimension (algebraic tdofs)
   udim = ufes->GetTrueVSize();
   pdim = pfes->GetTrueVSize(); 
   
   // initialize GridFunctions
   u_gf      = new ParGridFunction(ufes);      *u_gf = 0.0;
   p_gf      = new ParGridFunction(pfes);      *p_gf = 0.0;
   u_ext_gf = new ParGridFunction(ufes); *u_ext_gf = 0.0;

   // offsets
   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = udim;
   block_offsets[2] = pdim;
   block_offsets.PartialSum();

   // initialize vectors
   x   = new BlockVector(block_offsets);  *x = 0.0;
   rhs = new BlockVector(block_offsets); *rhs = 0.0;

   // VectorCoefficient storing extrapolated velocity for Convective term linearization
   u_ext_vc  = new VectorGridFunctionCoefficient(u_ext_gf); 

   // initialize vectors
   un     = new Vector(udim);     *un = 0.0; 
   un1    = new Vector(udim);    *un1 = 0.0; 
   un2    = new Vector(udim);    *un2 = 0.0; 
   u_bdf  = new Vector(udim);  *u_bdf = 0.0; 
   u_ext  = new Vector(udim);  *u_ext = 0.0; 

   // Create coefficient for kinematic viscosity
   kin_vis = new ConstantCoefficient(kin_vis_);

}


void MonolithicNavierSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
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

void MonolithicNavierSolver::AddAccelTerm(VecFuncT func, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}


// Solver setup
void MonolithicNavierSolver::SetSolver(SolverParams params)
{
   sParams = params;
}

void MonolithicNavierSolver::SetInitialConditionVel(VectorCoefficient &u_in)
{
   // Project coefficient onto velocity ParGridFunction (predicted and corrected)
   u_gf->ProjectCoefficient(u_in);
   u_gf->GetTrueDofs(x->GetBlock(0));
}

void MonolithicNavierSolver::SetInitialConditionPrevVel(VectorCoefficient &u_in)
{
   // Project coefficient onto velocity ParGridFunction (predicted and corrected)
   ParGridFunction tmp_gf(ufes);         
   tmp_gf.ProjectCoefficient(u_in);
   tmp_gf.GetTrueDofs(*un);
}

void MonolithicNavierSolver::SetInitialConditionPres(Coefficient &p_in)
{
   // Project coefficient onto pressure ParGridFunction (predicted and corrected)
   p_gf->ProjectCoefficient(p_in);
   p_gf->GetTrueDofs(x->GetBlock(1));
}

void MonolithicNavierSolver::Setup(real_t dt, int pc_type_, int schur_pc_type_, bool mass_lumping_, bool stiff_strain_)
{
   pc_type = pc_type_;              // preconditioner type
   schur_pc_type = schur_pc_type_;  // preconditioner type for Schur Complement
   mass_lumping = mass_lumping_;    // enable mass lumping (and forward to preconditioners if needed)
   stiff_strain = stiff_strain_;    // enable stiff strain integrator

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "Setup solver (using Full Assembly):" << std::endl;
   }

   sw_setup.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            out << "Setting up Navier solver... " << std::endl;
         }

         /// 1. Check partial assembly
         bool tensor = UsesTensorBasis(*ufes);

         MFEM_VERIFY(!(pa && !tensor), "Partial assembly is only supported for tensor elements.");

         MFEM_VERIFY(!pa, "Partial assembly is not supported for now (due to VectorConvectionInteg).");

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            if (pa)
            {
               out << "Using Partial Assembly. " << std::endl;
            }
            else
            {
               out << "Using Full Assembly. " << std::endl;
            }
         }

   /// 2. Determine the essential BC degrees of freedom
   if ( (bcs->GetVelDbcs()).size() > 0) // Dirichlet velocity bcs
   {
      ufes->GetEssentialTrueDofs(bcs->GetVelEssAttr(), vel_ess_tdof_full);
      vel_ess_tdof.Append(vel_ess_tdof_full);
   }
   if ( (bcs->GetVelDbcsXYZ()).size() > 0) // Dirichlet velocity bcs (x component)
   {
      ufes->GetEssentialTrueDofs(bcs->GetVelEssAttrX(), vel_ess_tdof_x, 0);
      ufes->GetEssentialTrueDofs(bcs->GetVelEssAttrY(), vel_ess_tdof_y, 1);
      ufes->GetEssentialTrueDofs(bcs->GetVelEssAttrZ(), vel_ess_tdof_z, 2);
      vel_ess_tdof.Append(vel_ess_tdof_x);
      vel_ess_tdof.Append(vel_ess_tdof_y);
      vel_ess_tdof.Append(vel_ess_tdof_z);
   }
 /*  if ((bcs->GetPresDbcs()).size() > 0) // Dirichlet pressure bcs
   {
      pfes->GetEssentialTrueDofs(bcs->GetPresEssAttr(), pres_ess_tdof); 
   } */

   /// 3. Setup and assemble bilinear forms 
   int skip_zeros = 0;
   Array<int> empty;

   // Velocity laplacian K (not scaled by viscosity --> done in assembly of Momentum operator)
   K_form = new ParBilinearForm(ufes);
   if (stiff_strain)
      K_form->AddDomainIntegrator(new StiffStrainIntegrator());
   else
      K_form->AddDomainIntegrator(new VectorDiffusionIntegrator());
   K_form->Assemble(skip_zeros); 
   //K_form->Finalize();
   K_form->FormSystemMatrix(empty, opK);

   // Velocity mass (not modified with bcs)
   M_form = new ParBilinearForm(ufes);
   if (mass_lumping)
      M_form->AddDomainIntegrator(new LumpedVectorMassIntegrator());
   else
      M_form->AddDomainIntegrator(new VectorMassIntegrator());

   M_form->Assemble(skip_zeros); 
   //M_form->Finalize();
   M_form->FormSystemMatrix(empty, opM);

   // Divergence
   D_form = new ParMixedBilinearForm(ufes, pfes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator());
   D_form->Assemble();
   D_form->Finalize();
   D_form->FormRectangularSystemMatrix(empty, empty, opD);
   opDe.Reset(opD.As<HypreParMatrix>()->EliminateCols(vel_ess_tdof));
   
  /* ConstantCoefficient zero_coeff(0.0);
   P_form = new ParBilinearForm(pfes);
   P_form->AddDomainIntegrator(new MassIntegrator(zero_coeff));
   P_form->Assemble();
   P_form->Finalize();
   P_form->FormSystemMatrix(empty, opP);
   opPe.Reset(opP.As<HypreParMatrix>()->EliminateRowsCols(pres_ess_tdof));*/

   // Gradient                         // NOTE: 1) We can replace with Dt and avoid assembly. 2) Also we can just maybe use MultTranspose D, and RAP operator with G instead of TripleProductOperator
   ConstantCoefficient negone(-1.0);      
   G_form = new ParMixedBilinearForm(pfes, ufes);  
   //G_form->AddDomainIntegrator(new GradientIntegrator());   // NOT WORKING
   G_form->AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator(negone))); 
   G_form->Assemble();
   G_form->Finalize();
   G_form->FormRectangularSystemMatrix(empty, vel_ess_tdof, opG);

   /// 4. Assemble linear form for rhs
   f_form = new ParLinearForm(ufes);
   // Adding forcing terms
   for (auto &accel_term : accel_terms)
   {
      f_form->AddDomainIntegrator( new VectorDomainLFIntegrator( *(accel_term.coeff) ), accel_term.attr );  
   }
   // Adding traction bcs
   for (auto &traction_bc : bcs->GetTractionBcs())
   {
      f_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator( *(traction_bc.coeff)) , traction_bc.attr);
   }

   // Adding custom traction bcs
   for (auto &traction_bc : bcs->GetCustomTractionBcs())
   {
      f_form->AddBoundaryIntegrator(new VectorNeumannLFIntegrator( *(traction_bc.u),*(traction_bc.p),*(traction_bc.alpha),*(traction_bc.beta)) , traction_bc.attr);
   }
   
   // Update time in Dirichlet velocity coefficients and project on predicted/corrected vectors and gf
   UpdateTimeBCS( 0.0 );

   // Update time in coefficients for rhs (acceleration/neumann) and assemble vector fv
   UpdateTimeRHS( 0.0 );


   /// 5. Construct the NS Operator, Solver and Preconditioner
   
   // Navier-Stokes Operator (opC will be set in each iteration)
   nsOp = new BlockOperator(block_offsets);
   nsOp->SetBlock(0, 1, opG.Ptr());
   nsOp->SetBlock(1, 0, opD.Ptr());

   // Schur Complement Preconditioner invS
   // Preconditioner for discrete pressure laplacian
   switch (schur_pc_type)
   {
   case 0: // Pressure Mass
      invS = new PMass(pfes, pres_ess_tdof, C_visccoeff.constant);
      break;
   case 1: // Pressure Laplacian
      invS = new PLap(pfes, pres_ess_tdof, C_bdfcoeff.constant);
      break;
   case 2: // PCD
      invS = new PCD(pfes, pres_ess_tdof, &C_bdfcoeff, &C_visccoeff, u_ext_vc);
      break;
   case 3: // Cahouet-Chabard
      invS = new CahouetChabard(pfes, pres_ess_tdof, dt, C_visccoeff.constant);
      break;
   case 4: // LSC
      invS = new LSC(pfes, pres_ess_tdof, opD.As<HypreParMatrix>(), opG.As<HypreParMatrix>(), opM.As<HypreParMatrix>());
      break;
   case 5: // Approximate discrete pressure laplacian
      invS = new ApproximateDiscreteLaplacian(pfes, pres_ess_tdof, opD.As<HypreParMatrix>(), opG.As<HypreParMatrix>(), opM.As<HypreParMatrix>(), C_bdfcoeff.constant);
      break;
   default:
      MFEM_ABORT("MonolithicNavierSolver::Setup() >> Unknown Schur preconditioner type: " << schur_pc_type);
      break;
   }

   // Navier-Stokes Preconditioner 
   switch (pc_type)
   {
   case 0: // Block diagonal
      nsPrec = new NavierBlockDiagonalPreconditioner(block_offsets);
      break;
   case 1: // Block Lower Triangular
      nsPrec = new NavierBlockLowerTriangularPreconditioner(block_offsets);
      break;
   case 2: // Block Upper Triangular
      nsPrec = new NavierBlockUpperTriangularPreconditioner(block_offsets);
      break;
   case 3: // Chorin-Temam
      nsPrec = new ChorinTemamPreconditioner(block_offsets);
      break;
   case 4: // Yosida
      nsPrec = new YosidaPreconditioner(block_offsets);
      break;
   case 5: // Chorin-Temam Pressure Corrected
      nsPrec = new ChorinTemamPressureCorrectedPreconditioner(block_offsets);
      break;
   case 6: // Yosida Pressure Corrected
      nsPrec = new YosidaPressureCorrectedPreconditioner(block_offsets);
      break;
   default:
      MFEM_ABORT("MonolithicNavierSolver::Setup() >> Unknown preconditioner type: " << pc_type);
      break;
   }
   nsPrec->SetSchurSolver(invS);

   // Navier-Stokes Solver
   nsSolver = new FGMRESSolver(pmesh->GetComm());
   nsSolver->iterative_mode = true;
   nsSolver->SetAbsTol(sParams.atol);
   nsSolver->SetRelTol(sParams.rtol);
   nsSolver->SetMaxIter(sParams.maxIter);
   nsSolver->SetOperator(*nsOp);
   nsSolver->SetPreconditioner(*nsPrec);
   nsSolver->SetPrintLevel(sParams.pl);

   sw_setup.Stop();
}

void MonolithicNavierSolver::Step(real_t &time, real_t dt, int current_step)
{
   /// 0.1 Update BDF time integration coefficients
   SetTimeIntegrationCoefficients( current_step );

   /// 0.2 Update time coefficients for rhs and bcs
   time += dt;
   UpdateTimeBCS( time );
   UpdateTimeRHS( time );

   /////////////////////////////////////////////////////////////////////////////////////////////////////////
   /// 0. Assemble Convective term (linearized: u_ext \cdot \grad u)  and update Operator/Preconditioner ///
   /////////////////////////////////////////////////////////////////////////////////////////////////////////

   sw_conv_assembly.Start();

   //// 0.1 Update NS Operator
   // Extrapolate velocity    u_ext = b1 un + b2 u_{n-1} + b3 u_{n-2}    
   // NOTE: we can create method for extrapolating that incorporates the SetTimeIntegrationCoefficients,
   // receives vectors, computes coefficients, return extrap (can be used to extrapolate pressure for Incremental version)
   add(b1, *un, b2, *un1, *u_ext);
   u_ext->Add(b3,*un2);
   u_ext_gf->SetFromTrueDofs(*u_ext);  

   Array<int> empty;
   int skip_zeros = 0;
   delete NL_form; NL_form = nullptr;
   opNL.Clear();
   opC.Clear();
   delete Ce; Ce = nullptr;
   NL_form = new ParBilinearForm(ufes);
   NL_form->AddDomainIntegrator(new VectorConvectionIntegrator(*u_ext_vc, 1.0));  
   NL_form->Assemble(skip_zeros); 
   NL_form->FormSystemMatrix(empty, opNL);

   C_visccoeff.constant = kin_vis->constant;
   C_bdfcoeff.constant = alpha / dt;
   auto Cmat = Add(C_bdfcoeff.constant, *(opM.As<HypreParMatrix>()), C_visccoeff.constant, *(opK.As<HypreParMatrix>()));
   Cmat->Add(1.0, *(opNL.As<HypreParMatrix>()));
   opC.Reset( Cmat, true );     // C = alpha/dt M + K + NL
   Ce = opC.As<HypreParMatrix>()->EliminateRowsCols(vel_ess_tdof);
   
   nsOp->SetBlock(0, 0, opC.Ptr());

   //// 0.2 Update NS Preconditioner

   // Create sigmaM for Chorin-Temam, and Pressure Corrected Preconditioners
   if (pc_type == 3 || pc_type > 4)
   {
      opSigmaM.Reset(new HypreParMatrix(*(opM.As<HypreParMatrix>())));
      *opSigmaM.As<HypreParMatrix>() *= C_bdfcoeff.constant;
      auto sigmaMe = opSigmaM.As<HypreParMatrix>()->EliminateRowsCols(vel_ess_tdof);
      delete sigmaMe;

      switch (pc_type)
      {
      case 3: // Chorin-Temam
         static_cast<ChorinTemamPreconditioner *>(nsPrec)->SetH2Operator(opSigmaM.Ptr());
         break;
      case 5: // Chorin-Temam Pressure Corrected
         static_cast<ChorinTemamPressureCorrectedPreconditioner *>(nsPrec)->SetH1Operator(opSigmaM.Ptr());
         static_cast<ChorinTemamPressureCorrectedPreconditioner *>(nsPrec)->SetH2Operator(opSigmaM.Ptr());
         break;
      case 6: // Yosida Pressure Corrected
         static_cast<YosidaPressureCorrectedPreconditioner *>(nsPrec)->SetH1Operator(opSigmaM.Ptr());
         break;
      default:
         MFEM_ABORT("MonolithicNavierSolver::Step() >> Unknown preconditioner type: " << pc_type);
         break;
      }
   }

   // Update schur complement preconditioner
   switch (schur_pc_type)
   {
   case 0:
      static_cast<PMass *>(invS)->SetCoefficients(C_visccoeff.constant);
      break;
   case 1: // Pressure Laplacian
      static_cast<PLap *>(invS)->SetCoefficients(C_bdfcoeff.constant);
      break;
   case 2: // PCD
      static_cast<PCD *>(invS)->Rebuild();
      break;
   case 3: // Cahouet-Chabard
      static_cast<CahouetChabard *>(invS)->SetCoefficients(dt, C_visccoeff.constant);
      break;
   case 4: // LSC
      static_cast<LSC *>(invS)->SetOperator(*opC.Ptr());
      break;
   case 5: // Approximate discrete pressure laplacian
      static_cast<ApproximateDiscreteLaplacian *>(invS)->SetCoefficients(C_bdfcoeff.constant);
      break;
   default:
      MFEM_ABORT("MonolithicNavierSolver::Step() >> Unknown Schur preconditioner type: " << schur_pc_type);
      break;
   }

   // Update Block Preconditioner
   nsPrec->SetOperator(*nsOp);


   sw_conv_assembly.Stop();

   //////////////////////////////////////////////
   /// 1. Solve velocity prediction step      ///
   //////////////////////////////////////////////
   /// C u_pred = fv + 1/dt M u_bdf
   
   sw_solve.Start(); 

   // Assemble rhs     fv + 1/dt M u_bdf
   add(a1, *un, a2, *un1, *u_bdf);     
   u_bdf->Add(a3,*un2);

   opM->AddMult(*u_bdf, rhs->GetBlock(0), 1.0/dt);

   // Apply bcs
   opC.As<HypreParMatrix>()->EliminateBC(*Ce, vel_ess_tdof, x->GetBlock(0), rhs->GetBlock(0)); // rhs_v1 -= Ce*u_pred

   // Solve current iteration.
   nsSolver->Mult(*rhs, *x);
   iter_solve = nsSolver->GetNumIterations();
   res_solve = nsSolver->GetFinalNorm();
   MFEM_VERIFY(nsSolver->GetConverged(), "GMRES solver did not converge. Aborting!");

   // Update gf 
   // Update gfs 
   u_gf->SetFromTrueDofs(x->GetBlock(0));
   p_gf->SetFromTrueDofs(x->GetBlock(1)); // Remove nullspace by removing mean of the pressure solution 
   if ((bcs->GetPresDbcs()).empty()) // TODO: change, since homogeneous neumann are not included in the list (include them in a vector like homogeneous_neumann.empty(). Can be done at setup time )
   {
      MeanZero(*p_gf);
      p_gf->GetTrueDofs(x->GetBlock(1));
   }

   sw_solve.Stop();

   // Print summary
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

}

/// Private Interface
void MonolithicNavierSolver::SetTimeIntegrationCoefficients(int step)
{
   // Maximum BDF order to use at current time step
   // step + 1 <= order <= max_bdf_order
   int bdf_order = std::min(step, max_bdf_order);

   if (step == 1 && bdf_order == 1)
   {
      alpha = 1.0;
      a1 = 1.0; 
      a2 = 0.0; 
      a3 = 0.0; 
      b1 = 1.0; 
      b2 = 0.0; 
      b3 = 0.0; 
   }
   else if (step >= 2 && bdf_order == 2)
   {
      alpha = 3.0/2.0;
      a1 = 2.0; 
      a2 = -1.0/2.0; 
      a3 = 0.0; 
      b1 = 2.0;  
      b2 = -1.0; 
      b3 = 0.0;  
   }
   else if (step >= 3 && bdf_order == 3)
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

void MonolithicNavierSolver::UpdateSolution()
{
   // Update solution at previous timesteps for BDF
   *un2 = *un1;
   *un1 = *un;
   *un  = x->GetBlock(0);
}

void MonolithicNavierSolver::UpdateTimeRHS( real_t new_time )
{
      Vector fv;
      fv.MakeRef(*rhs, block_offsets[0],
                   block_offsets[1]-block_offsets[0]);
      Vector fp;
      fp.MakeRef(*rhs, block_offsets[1],
                   block_offsets[2]-block_offsets[1]);

      fv = 0.0;
      fp = 0.0;

      // Update acceleration terms
      for (auto &accel_term : accel_terms)
      {
         accel_term.coeff->SetTime(new_time);
      }

      // Update traction bcs
      for (auto &traction_bc : bcs->GetTractionBcs())
      {
         traction_bc.coeff->SetTime(new_time);
      }

      // Update custom traction bcs
      for (auto &custom_traction_bc : bcs->GetCustomTractionBcs())
      {
         custom_traction_bc.alpha->SetTime(new_time);
         custom_traction_bc.beta->SetTime(new_time);
      }

      f_form->Assemble(); 
      f_form->ParallelAssemble(fv); 

      // Update pressure block of rhs (bcs from velocity)
      opDe->AddMult(x->GetBlock(0), fp, -1.0);
}

void MonolithicNavierSolver::UpdateTimeBCS( real_t new_time )
{
      Vector u;
      u.MakeRef(*x, block_offsets[0],
                   block_offsets[1]-block_offsets[0]);
      Vector p;
      p.MakeRef(*x, block_offsets[1],
                   block_offsets[2]-block_offsets[1]);

      // Projection of coeffs (pressure)
      for (auto &pres_dbc : bcs->GetPresDbcs())
      {
         pres_dbc.coeff->SetTime(new_time);
         p_gf->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
      }
      p_gf->GetTrueDofs(p);

      // Projection of coeffs (full velocity applied)
      for (auto &vel_dbc : bcs->GetVelDbcs())
      {
         vel_dbc.coeff->SetTime(new_time);
         u_gf->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
      }
      u_gf->GetTrueDofs(u);

      // Projection of coeffs (velocity component applied)
      ParGridFunction tmp_gf(ufes);        // temporary velocity gf for projection
      Vector          tmp_vec(udim);       // temporary velocity vector for projection
      Array<int>      tmp_tdofs;
      for (auto &vel_dbc : bcs->GetVelDbcsXYZ())
      {
         vel_dbc.coeff->SetTime(new_time);
         VectorArrayCoefficient tmp_coeff(sdim);                           // Set coefficient with right component
         tmp_coeff.Set(vel_dbc.dir, vel_dbc.coeff, false);
         tmp_gf.ProjectBdrCoefficient(tmp_coeff, vel_dbc.attr);           // Project on dummy gf
         tmp_gf.GetTrueDofs(tmp_vec);

         ufes->GetEssentialTrueDofs(vel_dbc.attr,tmp_tdofs,vel_dbc.dir);  // Update solution dofs
         for(int i=0;i<tmp_tdofs.Size();i++)
         {
            (u)[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
         }      
      }
      // Initialize solution gf with vector containing projected coefficients 
      // and update grid function and vector for provisional velocity
      u_gf->SetFromTrueDofs(u);
}

void MonolithicNavierSolver::MeanZero(ParGridFunction &gf)
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

   real_t integ = mass_lf->operator()(gf);

   gf -= integ / volume;
}

void MonolithicNavierSolver::Orthogonalize(Vector &v)
{
   real_t loc_sum = v.Sum();
   real_t global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, ufes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, ufes->GetComm());

   v -= global_sum / static_cast<real_t>(global_size);
}

void MonolithicNavierSolver::PrintLogo()
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

void MonolithicNavierSolver::PrintInfo()
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

      void
      MonolithicNavierSolver::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
      {
         paraview_dc = &paraview_dc_;

         if (uorder > 1)
         {
            paraview_dc->SetHighOrderOutput(true);
            paraview_dc->SetLevelsOfDetail(uorder);
         }

         paraview_dc->RegisterField("pressure", p_gf);
         paraview_dc->RegisterField("velocity", u_gf);
      }

      void
      MonolithicNavierSolver::RegisterVisItFields(VisItDataCollection &visit_dc_)
      {
         visit_dc = &visit_dc_;

         if (uorder > 1)
         {
            visit_dc->SetLevelsOfDetail(uorder);
         }

         visit_dc->RegisterField("pressure", p_gf);
         visit_dc->RegisterField("velocity", u_gf);
      }

      void MonolithicNavierSolver::AddParaviewField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(paraview_dc,
                     "Paraview data collection not initialized. Call RegisterParaviewFields first.");
         paraview_dc->RegisterField(field_name, gf);
      }

      void MonolithicNavierSolver::AddVisItField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(visit_dc,
                     "VisIt data collection not initialized. Call RegisterVisItFields first.");
         visit_dc->RegisterField(field_name, gf);
      }

      void
      MonolithicNavierSolver::WriteFields(const int &it, const real_t &time)
      {
         if (visit_dc)
         {
            if (pmesh->GetMyRank() == 0 && verbose)
            {
               out << "Writing VisIt files ..." << std::flush;
            }

            visit_dc->SetCycle(it);
            visit_dc->SetTime(time);
            visit_dc->Save();

            if (pmesh->GetMyRank() == 0 && verbose)
            {
               out << " done." << std::endl;
            }
         }

         if (paraview_dc)
         {
            if (pmesh->GetMyRank() == 0 && verbose)
            {
               out << "Writing Paraview files ..." << std::flush;
            }

            paraview_dc->SetCycle(it);
            paraview_dc->SetTime(time);
            paraview_dc->Save();

            if (pmesh->GetMyRank() == 0 && verbose)
            {
               out << " done." << std::endl;
            }
         }
      }


/// Destructor
MonolithicNavierSolver::~MonolithicNavierSolver()
{
   delete ufes; ufes = nullptr;
   delete pfes; pfes = nullptr;

   delete bcs; bcs = nullptr;

   delete K_form;   K_form = nullptr;
   delete G_form;   G_form = nullptr;
   delete M_form;   M_form = nullptr;
   delete D_form;   D_form = nullptr;
   delete NL_form;  NL_form = nullptr;
   delete f_form;   f_form = nullptr;

   delete u_gf;           u_gf = nullptr;
   delete p_gf;           p_gf = nullptr;
   delete u_ext_gf;       u_ext_gf = nullptr;

   delete u_bdf;     u_bdf = nullptr;
   delete u_ext;     u_ext = nullptr;
   delete un;        un = nullptr;
   delete un1;       un1 = nullptr;
   delete un2;       un2 = nullptr;

   delete x;         x = nullptr;
   delete rhs;       rhs = nullptr;

   delete Ce;        Ce = nullptr;
   delete mass_lf;   mass_lf = nullptr;

   delete nsSolver;   nsSolver = nullptr;
   delete nsOp;       nsOp = nullptr;

   delete invS;       invS = nullptr;
   delete nsPrec;     nsPrec = nullptr;

   delete kin_vis; kin_vis = nullptr;
   delete u_ext_vc; u_ext_vc = nullptr;   

   opK.Clear();
   opM.Clear();
   opD.Clear();
   opG.Clear();
   opDe.Clear();
   opNL.Clear();
   opC.Clear();
   opSigmaM.Clear();
}

} // namespace navier
} // namespace mfem