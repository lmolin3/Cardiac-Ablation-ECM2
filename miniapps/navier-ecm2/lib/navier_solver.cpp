#include "navier_solver.hpp"
#include <fstream>
#include <iomanip>


namespace mfem{

namespace navier{

/// Constructor
NavierUnsteadySolver::NavierUnsteadySolver(std::shared_ptr<ParMesh> pmesh_,
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
   u_pred_gf = new ParGridFunction(ufes); *u_pred_gf = 0.0;
   p_gf      = new ParGridFunction(pfes);      *p_gf = 0.0;
   p_pred_gf = new ParGridFunction(pfes); *p_pred_gf = 0.0;
   u_bdf_gf  = new ParGridFunction(ufes);  *u_bdf_gf = 0.0;
   u_ext_gf  = new ParGridFunction(ufes);  *u_ext_gf = 0.0;

   // VectorCoefficient storing extrapolated velocity for Convective term linearization
   u_ext_vc  = new VectorGridFunctionCoefficient(u_ext_gf); 

   // initialize vectors
   u      = new Vector(udim);      *u = 0.0; 
   p      = new Vector(pdim);      *p = 0.0; 
   u_pred = new Vector(udim); *u_pred = 0.0; 
   p_pred = new Vector(pdim); *p_pred = 0.0; 
   un     = new Vector(udim);     *un = 0.0; 
   un1    = new Vector(udim);    *un1 = 0.0; 
   un2    = new Vector(udim);    *un2 = 0.0; 
   u_bdf  = new Vector(udim);  *u_bdf = 0.0; 
   u_ext  = new Vector(udim);  *u_ext = 0.0; 
   fv     = new Vector(udim);     *fv = 0.0; 
   fp     = new Vector(pdim);     *fp = 0.0; 
   rhs_v1 = new Vector(udim); *rhs_v1 = 0.0; 
   rhs_v2 = new Vector(udim); *rhs_v2 = 0.0; 
   rhs_p1 = new Vector(pdim); *rhs_p1 = 0.0; 
   rhs_p2 = new Vector(pdim); *rhs_p2 = 0.0; 
       Gp = new Vector(udim);     *Gp = 0.0; 

   tmp1   = new Vector(udim); *tmp1 = 0.0; 
   tmp2   = new Vector(udim); *tmp2 = 0.0; 

   // set default parameters gamma
   gamma  = 1.0;

   // Create coefficient for kinematic viscosity
   kin_vis = new ConstantCoefficient(kin_vis_);

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
void NavierUnsteadySolver::SetSolvers(SolverParams params1, SolverParams params2,
                                      SolverParams params3, SolverParams params4)
{
   s1Params = params1;
   s2Params = params2;
   s3Params = params3;
   s4Params = params4;
}


void NavierUnsteadySolver::SetInitialConditionVel(VectorCoefficient &u_in)
{
   // Project coefficient onto velocity ParGridFunction (predicted and corrected)
   u_gf->ProjectCoefficient(u_in);
   u_gf->GetTrueDofs(*u);
   *u_pred = *u;
   u_pred_gf->SetFromTrueDofs(*u_pred);
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
   p_gf->GetTrueDofs(*p);
   *p_pred = *p;
   p_pred_gf->SetFromTrueDofs(*p_pred);
}

void NavierUnsteadySolver::Setup(real_t dt, int pc_type_)
{
   pc_type = pc_type_; // preconditioner type

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
   if ((bcs->GetPresDbcs()).size() > 0) // Dirichlet pressure bcs
   {
      pfes->GetEssentialTrueDofs(bcs->GetPresEssAttr(), pres_ess_tdof); 
   }

   /// 2. Setup and assemble bilinear forms 
   int skip_zeros = 0;
   Array<int> empty;


   // Velocity laplacian K (not scaled by viscosity --> done in assembly of Momentum operator)
   K_form = new ParBilinearForm(ufes);  
   K_form->AddDomainIntegrator(new VectorDiffusionIntegrator());
   K_form->Assemble(skip_zeros); 
   K_form->FormSystemMatrix(empty, opK);

   // Velocity mass (not modified with bcs)
   M_form = new ParBilinearForm(ufes);  
   M_form->AddDomainIntegrator(new VectorMassIntegrator());
   M_form->Assemble(skip_zeros); 
   M_form->FormSystemMatrix(empty, opM);
   sigmaM = new HypreParMatrix(*(opM.As<HypreParMatrix>())); // copy of mass matrix used for operator H in Schur Complement

   // Divergence
   ConstantCoefficient neg_one(-1.0);
   D_form = new ParMixedBilinearForm(ufes, pfes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));
   D_form->Assemble();
   D_form->FormRectangularSystemMatrix(empty, empty, opD);
   opDe.Reset(opD.As<HypreParMatrix>()->EliminateCols(vel_ess_tdof));

   // Gradient                         // NOTE: 1) We can replace with Dt and avoid assembly. 2) Also we can just maybe use MultTranspose D, and RAP operator with G instead of TripleProductOperator
   G_form = new ParMixedBilinearForm(pfes, ufes);
   G_form->AddDomainIntegrator(new GradientIntegrator());
   G_form->Assemble();
   G_form->FormRectangularSystemMatrix(empty, empty, opG);
   opG.EliminateRows(vel_ess_tdof);

   /// 3. Assemble linear form for rhs
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


   /// 4. Construct the operators for preconditioners
   //
   //     Here we use BoomerAMG for:
   //     * convective-diffusive part C =  alpha/dt M + kin_vis K + NL,
   //     * velocity mass matrix M
   //     * pressure laplacian (for schur complement) 
   //

   // Preconditioner for discrete pressure laplacian

   // This is just placeholder to make PCD work
   ConstantCoefficient mass_coeff(1/dt);

   switch (pc_type)
   {
   case 0: // Pressure Mass
      pc_builder = new PMassBuilder(pfes, pres_ess_tdof, dt);
      break;
   case 1: // Pressure Laplacian
      pc_builder = new PLapBuilder(pfes, pres_ess_tdof);
      break;
   case 2: // PCD
      pc_builder = new PCDBuilder(pfes, pres_ess_tdof, &mass_coeff, &C_visccoeff); // pc_builder = new PCDBuilder(pfes, pres_ess_tdof, &C_bdfcoeff, &C_visccoeff, u_ext_vc);
      break;
   case 3: // Cahouet-Chabard
      pc_builder = new CahouetChabardBuilder(pfes, pres_ess_tdof, dt, C_visccoeff.constant);
      break;
   case 4: // Approximate inverse
      pc_builder = new SchurApproxInvBuilder(pfes, pres_ess_tdof, dt);
      break;
   default:
      MFEM_ABORT("NavierStokesOperator::Assemble() >> Unknown preconditioner type: " << pc_type);
      break;
   }

   // Get the preconditioner for Schur Complement
   invDHG_pc = pc_builder->GetSolver();

   // Preconditioners for H1 and H2
   H1_pc = new HypreBoomerAMG();        
   H1_pc->SetPrintLevel(0);
   H1_pc->SetSystemsOptions(sdim);
   H1_pc->iterative_mode = false;

   H2_pc = new HypreBoomerAMG();
   H2_pc->SetPrintLevel(0);
   H2_pc->SetSystemsOptions(sdim);
   H2_pc->iterative_mode = false;

   // Velocity Prediction (BoomerAMG; operator will be assigned inside loop)
   invC_pc = new HypreBoomerAMG();
   dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetSystemsOptions(sdim);
   invC_pc->iterative_mode = false;


   /// 5. Construct the operators for solvers
   //
   //     Here we use:
   //     * GMRES for velocity prediction step;
   //     * CG on custom operator DHG for pressure steps;
   //     * CG for velocity correction step.
   //

   // Velocity prediction 
   invC = new GMRESSolver(ufes->GetComm());
   invC->iterative_mode = true;
   invC->SetAbsTol(s1Params.atol);
   invC->SetRelTol(s1Params.rtol);
   invC->SetMaxIter(s1Params.maxIter);
   invC->SetPreconditioner(*invC_pc);
   invC->SetPrintLevel(s1Params.pl);

   // Chorin-Temam operator H1 = dt/alpha M^{-1}
   H1 = new CGSolver(ufes->GetComm());
   H1->iterative_mode = false;         // keep it to false since it's in the TripleProductOperator    
   H1->SetAbsTol(s4Params.atol);
   H1->SetRelTol(s4Params.rtol);
   H1->SetMaxIter(s4Params.maxIter);
   H1->SetPreconditioner(*H1_pc);
   H1->SetOperator(*sigmaM);
   H1->SetPrintLevel(s4Params.pl);

   // Pressure prediction 
   //DHG = new TripleProductOperator(opD.Ptr(),H1,opG.Ptr(),false,false,false);      // operator through action: DHG = D  dt/alpha M^{-1} G
   //DHGc = new ConstrainedOperator(DHG, pres_ess_tdof, true);       // operator DHG constraining pressure dofs
   DHG = new DiscretePressureLaplacian(opD.Ptr(), H1, opG.Ptr(), false, false, false); // operator through action: D dt/alpha M^{-1}G
   DHGc = new ConstrainedOperator(DHG, pres_ess_tdof, true);  

   invDHG1 = new GMRESSolver(ufes->GetComm());
   invDHG1->iterative_mode = true;      
   invDHG1->SetAbsTol(s2Params.atol);
   invDHG1->SetRelTol(s2Params.rtol); 
   invDHG1->SetMaxIter(s2Params.maxIter);
   invDHG1->SetOperator(*DHGc);          
   invDHG1->SetPreconditioner(*invDHG_pc);       
   invDHG1->SetPrintLevel(s2Params.pl); 

   // Pressure correction 
   invDHG2 = new GMRESSolver(ufes->GetComm());
   invDHG2->iterative_mode = true;      
   invDHG2->SetAbsTol(s3Params.atol);
   invDHG2->SetRelTol(s3Params.rtol); 
   invDHG2->SetMaxIter(s2Params.maxIter);
   invDHG2->SetOperator(*DHGc);          
   invDHG2->SetPreconditioner(*invDHG_pc);       
   invDHG2->SetPrintLevel(s3Params.pl);

   sw_setup.Stop();

}

/// Private Interface

void NavierUnsteadySolver::SetTimeIntegrationCoefficients(int step)
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


void NavierUnsteadySolver::UpdateSolution()
{
   // Update solution at previous timesteps for BDF
   *un2 = *un1;
   *un1 = *un;
   *un  = *u;
}

void NavierUnsteadySolver::UpdateTimeRHS( real_t new_time )
{
      *fv = 0.0;   
      *fp = 0.0;

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
      f_form->ParallelAssemble(*fv); 

      // Update pressure block of rhs (bcs from velocity)
      opDe->AddMult(*u, *fp, -1.0);
}

void NavierUnsteadySolver::UpdateTimeBCS( real_t new_time )
{
      // Projection of coeffs (pressure)
      for (auto &pres_dbc : bcs->GetPresDbcs())
      {
         pres_dbc.coeff->SetTime(new_time);
         p_gf->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
         p_pred_gf->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
      }
      p_gf->GetTrueDofs(*p);
      p_pred_gf->GetTrueDofs(*p_pred);

      // Projection of coeffs (full velocity applied)
      for (auto &vel_dbc : bcs->GetVelDbcs())
      {
         vel_dbc.coeff->SetTime(new_time);
         u_gf->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
         u_pred_gf->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
      }
      u_gf->GetTrueDofs(*u);
      u_pred_gf->GetTrueDofs(*u_pred);

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
            (*u)[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
            (*u_pred)[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
         }      
      }
      // Initialize solution gf with vector containing projected coefficients 
      // and update grid function and vector for provisional velocity
      u_gf->SetFromTrueDofs(*u);
      u_pred_gf->SetFromTrueDofs(*u_pred);

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

   real_t integ = mass_lf->operator()(gf);

   gf -= integ / volume;
}

void NavierUnsteadySolver::Orthogonalize(Vector &v)
{
   real_t loc_sum = v.Sum();
   real_t global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, ufes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, ufes->GetComm());

   v -= global_sum / static_cast<real_t>(global_size);
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
   real_t my_rt[6], rt_max[6];

   my_rt[0] = sw_setup.RealTime();
   my_rt[1] = sw_conv_assembly.RealTime();
   my_rt[2] = sw_vel_pred.RealTime();
   my_rt[3] = sw_pres_pred.RealTime();
   my_rt[4] = sw_pres_corr.RealTime();
   my_rt[5] = sw_vel_corr.RealTime();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "CONV-ASS"
                << std::setw(10) << "VPRED" << std::setw(10) << "PPRED"
                << std::setw(10) << "PCORR" << std::setw(10) << "VCORR"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2]
                << std::setw(10) << my_rt[3] << std::setw(10) << my_rt[4]
                << std::setw(10) << my_rt[5] << "\n";


      mfem::out << std::setprecision(8);
   }
}

      void
      NavierUnsteadySolver::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
      {
         paraview_dc = &paraview_dc_;

         if (uorder > 1)
         {
            paraview_dc->SetHighOrderOutput(true);
            paraview_dc->SetLevelsOfDetail(uorder);
         }

         paraview_dc->RegisterField("corrected_pressure", p_gf);
         paraview_dc->RegisterField("corrected_velocity", u_gf);
         paraview_dc->RegisterField("predicted_velocity", u_pred_gf);
         paraview_dc->RegisterField("predicted_pressure", p_pred_gf);
      }

      void
      NavierUnsteadySolver::RegisterVisItFields(VisItDataCollection &visit_dc_)
      {
         visit_dc = &visit_dc_;

         if (uorder > 1)
         {
            visit_dc->SetLevelsOfDetail(uorder);
         }

         visit_dc->RegisterField("corrected_pressure", p_gf);
         visit_dc->RegisterField("corrected_velocity", u_gf);
         visit_dc->RegisterField("predicted_velocity", u_pred_gf);
         visit_dc->RegisterField("predicted_pressure", p_pred_gf);
      }

      void NavierUnsteadySolver::AddParaviewField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(paraview_dc,
                     "Paraview data collection not initialized. Call RegisterParaviewFields first.");
         paraview_dc->RegisterField(field_name, gf);
      }

      void NavierUnsteadySolver::AddVisItField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(visit_dc,
                     "VisIt data collection not initialized. Call RegisterVisItFields first.");
         visit_dc->RegisterField(field_name, gf);
      }

      void
      NavierUnsteadySolver::WriteFields(const int &it, const real_t &time)
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
NavierUnsteadySolver::~NavierUnsteadySolver()
{
   delete ufes; ufes = nullptr;
   delete pfes; pfes = nullptr;

   delete K_form;   K_form = nullptr;
   delete M_form;   M_form = nullptr;
   delete D_form;   D_form = nullptr;
   delete NL_form;  NL_form = nullptr;
   delete S_form;   S_form = nullptr;
   delete f_form;   f_form = nullptr;

   delete u_gf;           u_gf = nullptr;
   delete p_gf;           p_gf = nullptr;
   delete u_pred_gf; u_pred_gf = nullptr;
   delete p_pred_gf; p_pred_gf = nullptr;
   delete u_bdf_gf;   u_bdf_gf = nullptr;
   delete u_ext_gf;   u_ext_gf = nullptr;

   delete u;         u = nullptr;
   delete p;         p = nullptr;
   delete u_pred;    u_pred = nullptr;
   delete p_pred;    p_pred = nullptr;
   delete u_bdf;     u_bdf = nullptr;
   delete u_ext;     u_ext = nullptr;
   delete un;        un = nullptr;
   delete un1;       un1 = nullptr;
   delete un2;       un2 = nullptr;
   delete fv;        fv = nullptr;
   delete fp;        fp = nullptr;
   delete rhs_v1;    rhs_v1 = nullptr;
   delete rhs_v2;    rhs_v2 = nullptr;
   delete rhs_p1;    rhs_p1 = nullptr;
   delete rhs_p2;    rhs_p2 = nullptr;
   delete tmp1;      tmp1 = nullptr;
   delete tmp2;      tmp2 = nullptr;

   delete sigmaM;   sigmaM = nullptr; 
   delete Ce;        Ce = nullptr;
   delete DHGc;      DHGc = nullptr;
   delete mass_lf;   mass_lf = nullptr;

   delete invC;              invC = nullptr;
   delete H1;                  H1 = nullptr;
   delete H2;                  H2 = nullptr;
   delete invDHG1;        invDHG1 = nullptr;
   delete invDHG2;        invDHG2 = nullptr;
   delete invC_pc;        invC_pc = nullptr;
   delete H1_pc;        H1_pc = nullptr;
   delete H2_pc;        H2_pc = nullptr;

   delete pc_builder; pc_builder = nullptr; // Do not delete invDHG_pc, it is owned by pc_builder

   opK.Clear();
   opM.Clear();
   opD.Clear();
   opG.Clear();
   opDe.Clear();
   opNL.Clear();
   opC.Clear();
}



////////////////////////////////////////////////////////////////////////////////////////////////////////

void ChorinTemamSolver::Setup(real_t dt, int prec_type_)
{
   NavierUnsteadySolver::Setup(dt, prec_type_);

   // Assign solver for operator H2
   // Chorin-Temam operator H2 = dt/alpha M^{-1}
   H2 = new CGSolver(ufes->GetComm());

   H2->iterative_mode = true;        
   H2->SetAbsTol(s4Params.atol);
   H2->SetRelTol(s4Params.rtol);
   H2->SetMaxIter(s4Params.maxIter);
   H2->SetPreconditioner(*H2_pc);
   H2->SetPrintLevel(s4Params.pl);
}

void ChorinTemamSolver::Step(real_t &time, real_t dt, int current_step)
{
   /// 0.1 Update BDF time integration coefficients
   SetTimeIntegrationCoefficients( current_step );

   /// 0.2 Update time coefficients for rhs and bcs
   time += dt;
   UpdateTimeBCS( time );
   UpdateTimeRHS( time );

   ///////////////////////////////////////////////////////////////////////
   /// 0. Assemble Convective term (linearized: u_ext \cdot \grad u)   ///
   ///////////////////////////////////////////////////////////////////////

   sw_conv_assembly.Start();

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
   opA.Clear();
   delete Ce; Ce = nullptr;
   NL_form = new ParBilinearForm(ufes);
   NL_form->AddDomainIntegrator(new VectorConvectionIntegrator(*u_ext_vc, 1.0));  // += C 
   NL_form->Assemble(skip_zeros); 
   NL_form->FormSystemMatrix(empty, opNL);
   
   C_visccoeff.constant = kin_vis->constant;
   C_bdfcoeff.constant = alpha / dt;
   auto Cmat = Add(C_bdfcoeff.constant, *(opM.As<HypreParMatrix>()), C_visccoeff.constant, *(opK.As<HypreParMatrix>()));
   Cmat->Add(1.0, *(opNL.As<HypreParMatrix>()));
   opC.Reset( Cmat, true );     // C = alpha/dt M + K + NL
   Ce = opC.As<HypreParMatrix>()->EliminateRowsCols(vel_ess_tdof);
   invC->SetOperator(*opC);

   // Assemble solver for H = inv( alpha/dt M )
   delete sigmaM; 
   sigmaM = new HypreParMatrix(*(opM.As<HypreParMatrix>()));
   *sigmaM *= C_bdfcoeff.constant;
   auto sigmaMe = sigmaM->EliminateRowsCols(vel_ess_tdof);  
   delete sigmaMe; 
   H1->SetOperator(*sigmaM); 
   H2->SetOperator(*sigmaM); 

   // Update timestep for discrete pressure laplacian preconditioner
   switch (pc_type)
   {
   case 0:
      static_cast<PMassPC *>(invDHG_pc)->SetCoefficients(dt);
      break;
   case 1: // Pressure Laplacian
      break;
   case 2: // PCD
      pc_builder->RebuildPreconditioner();
      invDHG_pc = pc_builder->GetSolver();
      break;
   case 3: // Cahouet-Chabard
      static_cast<CahouetChabardPC *>(invDHG_pc)->SetCoefficients(dt, C_visccoeff.constant);
      break;
   case 4: // Approximate inverse
      static_cast<SchurApproxInvPC *>(invDHG_pc)->SetCoefficients(dt);
      break;
   default:
      MFEM_ABORT("NavierStokesOperator::Assemble() >> Unknown preconditioner type: " << pc_type);
      break;
   }

   sw_conv_assembly.Stop();

   //////////////////////////////////////////////
   /// 1. Solve velocity prediction step      ///
   //////////////////////////////////////////////
   /// C u_pred = fv + 1/dt M u_bdf
   
   sw_vel_pred.Start(); 

   // Assemble rhs     fv + 1/dt M u_bdf
   add(a1, *un, a2, *un1, *u_bdf);     
   u_bdf->Add(a3,*un2);

   rhs_v1->Set(1.0,*fv);               
   opM->Mult(*u_bdf,*tmp1);
   rhs_v1->Add(1.0/dt, *tmp1);

   // Apply bcs
   opC.As<HypreParMatrix>()->EliminateBC(*Ce,vel_ess_tdof,*u_pred,*rhs_v1); // rhs_v1 -= Ce*u_pred

   // Solve current iteration.
   invC->Mult(*rhs_v1, *u_pred);
   iter_v1solve = invC->GetNumIterations();
   res_v1solve = invC->GetFinalNorm();
   MFEM_VERIFY(invC->GetConverged(), "Velocity prediction step did not converge. Aborting!");

   // Update gf 
   u_pred_gf->SetFromTrueDofs(*u_pred);
 
   sw_vel_pred.Stop();

   //////////////////////////////////////////////
   /// 2. Solve pressure prediction step      ///
   //////////////////////////////////////////////
   /// DHG p_pred = D u_pred - fp

   sw_pres_pred.Start(); 

   // Assemble rhs            D u_pred - fp   = D u_pred + De u_pred
   opD->Mult(*u_pred,*rhs_p1);
   rhs_p1->Add(-1.0,*fp);

   // Apply bcs
   DHGc->EliminateRHS(*p_pred,*rhs_p1);

   // Solve current iteration.
   invDHG1->Mult(*rhs_p1,*p_pred);
   iter_p1solve = invDHG1->GetNumIterations();
   res_p1solve = invDHG1->GetFinalNorm();
   MFEM_VERIFY(invDHG1->GetConverged(), "Pressure prediction step did not converge. Aborting!");

   // Remove nullspace by removing mean of the pressure solution 
   p_pred_gf->SetFromTrueDofs(*p_pred);  
   if( (bcs->GetPresDbcs()).empty())
   {
      MeanZero(*p_pred_gf);
      p_pred_gf->GetTrueDofs(*p_pred);
   }

   sw_pres_pred.Stop();

   //////////////////////////////////////////////
   /// 3. Solve pressure correction step      ///
   //////////////////////////////////////////////
   /// DHG p = D H A H G p_pred

   sw_pres_corr.Start();
   
   // Assemble rhs            D H A H G p_pred  (A = C for Chorin-Teman, A = kin_vis K + NL  for Yosida)     
   opG->Mult(*p_pred,*Gp);            //         G p_pred = Gp   --> stored to be reused in Velocity correction
   H1->Mult(*Gp,*tmp2);               //       H G p_pred = tmp2
   opC->Mult(*tmp2,*tmp1);            //     A H G p_pred = tmp1
   H1->Mult(*tmp1,*tmp2);             //   H A H G p_pred = tmp2
   opD->Mult(*tmp2,*rhs_p2);          // D H A H G p_pred = tmp1

   // Apply bcs
   DHGc->EliminateRHS(*p,*rhs_p2);   

   // Solve current iteration.
   invDHG2->Mult(*rhs_p2,*p);
   iter_p2solve = invDHG2->GetNumIterations();
   res_p2solve = invDHG2->GetFinalNorm(); 
   MFEM_VERIFY(invDHG2->GetConverged(), "Pressure correction step did not converge. Aborting!");

   // Remove nullspace by removing mean of the pressure solution 
   p_gf->SetFromTrueDofs(*p);  

   sw_pres_corr.Stop();

   //////////////////////////////////////////////
   /// 4. Solve velocity correction step      ///
   //////////////////////////////////////////////
   /// u = u_pred - H2 G p    (G p_pred for Chorin Temam)
 
   sw_vel_corr.Start();
          
   H2->Mult(*Gp,*u);
   u->Neg();
   u->Add(1.0,*u_pred);
   iter_v2solve = H2->GetNumIterations();
   res_v2solve = H2->GetFinalNorm();
   MFEM_VERIFY(H2->GetConverged(), "Velocity correction step did not converge. Aborting!");

   sw_vel_corr.Stop();

   //// 5. Relaxation (velocity)  u = gamma u + (1 - gamma) un
   //add(gamma,*u,(1.0-gamma),*un,*u);
   
   u_gf->SetFromTrueDofs(*u);

   // Print summary
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(13) << "Step" << std::setw(10) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      mfem::out << std::setw(10) << "Velocity Prediction " << std::setw(5) << std::fixed
                   << iter_v1solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_v1solve
                   << "   " << s1Params.rtol << "\n";
      mfem::out << std::setw(10) << "Pressure Prediction " << std::setw(5) << std::fixed
                   << iter_p1solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_p1solve
                   << "   " << s2Params.rtol << "\n";
      mfem::out << std::setw(10) << "Pressure Correction " << std::setw(5) << std::fixed
                   << iter_p2solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_p2solve
                   << "   " << s3Params.rtol << "\n";
      mfem::out << std::setw(10) << "Velocity Correction " << std::setw(5) << std::fixed
                   << iter_v2solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_v2solve
                   << "   " << s4Params.rtol << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }

   // Update solution at previous timesteps and time
   UpdateSolution(); 

}


////////////////////////////////////////////////////////////////////////////////////////////////////////


void YosidaSolver::Setup(real_t dt, int prec_type_)
{
   NavierUnsteadySolver::Setup(dt, prec_type_);

   // Assign solver for operator H2
   // Chorin-Temam operator H2 = dt/alpha M^{-1}, Yosida C^{-1}
   H2 = new GMRESSolver(ufes->GetComm());

   H2->iterative_mode = true;        
   H2->SetAbsTol(s4Params.atol);
   H2->SetRelTol(s4Params.rtol);
   H2->SetMaxIter(s4Params.maxIter);
   H2->SetPreconditioner(*H2_pc);
   H2->SetPrintLevel(s4Params.pl);
}

void YosidaSolver::Step(real_t &time, real_t dt, int current_step)
{
   /// 0.1 Update BDF time integration coefficients
   SetTimeIntegrationCoefficients( current_step );

   /// 0.2 Update time coefficients for rhs and bcs
   time += dt;
   UpdateTimeBCS( time );
   UpdateTimeRHS( time );

   ///////////////////////////////////////////////////////////////////////
   /// 0. Assemble Convective term (linearized: u_ext \cdot \grad u)   ///
   ///////////////////////////////////////////////////////////////////////

   sw_conv_assembly.Start();

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
   NL_form->AddDomainIntegrator(new VectorConvectionIntegrator(*u_ext_vc, 1.0));  // += C 
   NL_form->Assemble(skip_zeros); 
   NL_form->FormSystemMatrix(empty, opNL);
   
   C_visccoeff.constant = kin_vis->constant;
   C_bdfcoeff.constant = alpha / dt; 
   auto Cmat = Add(C_bdfcoeff.constant, *(opM.As<HypreParMatrix>()), C_visccoeff.constant, *(opK.As<HypreParMatrix>()));
   Cmat->Add(1.0, *(opNL.As<HypreParMatrix>()));
   opC.Reset( Cmat, true );     // C = alpha/dt M + A
   Ce = opC.As<HypreParMatrix>()->EliminateRowsCols(vel_ess_tdof);
   invC->SetOperator(*opC);

   // Assemble solver for H = inv( alpha/dt M )
   delete sigmaM; 
   sigmaM = new HypreParMatrix(*(opM.As<HypreParMatrix>()));
   *sigmaM *= C_bdfcoeff.constant;
   auto sigmaMe = sigmaM->EliminateRowsCols(vel_ess_tdof);  
   delete sigmaMe; 
   H1->SetOperator(*sigmaM); 
   H2->SetOperator(*opC); 

   // Update timestep for discrete pressure laplacian preconditioner
   switch (pc_type)
   {
   case 0:
      static_cast<PMassPC *>(invDHG_pc)->SetCoefficients(dt);
      break;
   case 1: // Pressure Laplacian
      break;
   case 2: // PCD
      pc_builder->RebuildPreconditioner();
      invDHG_pc = pc_builder->GetSolver();
      break;
   case 3: // Cahouet-Chabard
      static_cast<CahouetChabardPC *>(invDHG_pc)->SetCoefficients(dt, C_visccoeff.constant);
      break;
   case 4: // Approximate inverse
      static_cast<SchurApproxInvPC *>(invDHG_pc)->SetCoefficients(dt);
      break;
   default:
      MFEM_ABORT("NavierStokesOperator::Assemble() >> Unknown preconditioner type: " << pc_type);
      break;
   }

   sw_conv_assembly.Stop();

   //////////////////////////////////////////////
   /// 1. Solve velocity prediction step      ///
   //////////////////////////////////////////////
   /// C u_pred = fv + 1/dt M u_bdf
   
   sw_vel_pred.Start(); 

   // Assemble rhs     fv + 1/dt M u_bdf
   add(a1, *un, a2, *un1, *u_bdf);     
   u_bdf->Add(a3,*un2);

   rhs_v1->Set(1.0,*fv);               
   opM->Mult(*u_bdf,*tmp1);
   rhs_v1->Add(1.0/dt, *tmp1);

   // Apply bcs
   opC.As<HypreParMatrix>()->EliminateBC(*Ce,vel_ess_tdof,*u_pred,*rhs_v1); // rhs_v1 -= Ce*u_pred

   // Solve current iteration.
   invC->Mult(*rhs_v1, *u_pred);
   iter_v1solve = invC->GetNumIterations();
   res_v1solve = invC->GetFinalNorm();
   MFEM_VERIFY(invC->GetConverged(), "Velocity prediction step did not converge. Aborting!");

   // Update gf 
   u_pred_gf->SetFromTrueDofs(*u_pred);
 
   sw_vel_pred.Stop();

   //////////////////////////////////////////////
   /// 2. Solve pressure prediction step      ///
   //////////////////////////////////////////////
   /// DHG p_pred = D u_pred - fp

   sw_pres_pred.Start(); 

   // Assemble rhs            D u_pred - fp   = D u_pred + De u_pred
   opD->Mult(*u_pred,*rhs_p1);
   rhs_p1->Add(-1.0,*fp);

   // Apply bcs
   DHGc->EliminateRHS(*p_pred,*rhs_p1);

   // Solve current iteration.
   invDHG1->Mult(*rhs_p1,*p_pred);
   iter_p1solve = invDHG1->GetNumIterations();
   res_p1solve = invDHG1->GetFinalNorm();
   MFEM_VERIFY(invDHG1->GetConverged(), "Pressure prediction step did not converge. Aborting!");

   // Remove nullspace by removing mean of the pressure solution 
   p_pred_gf->SetFromTrueDofs(*p_pred);  
   if( (bcs->GetPresDbcs()).empty())
   {
      MeanZero(*p_pred_gf);
      p_pred_gf->GetTrueDofs(*p_pred);
   }

   sw_pres_pred.Stop();

   //////////////////////////////////////////////
   /// 3. Solve pressure correction step      ///
   //////////////////////////////////////////////
   /// DHG p = D H A H G p_pred

   sw_pres_corr.Start();
   
   // Assemble rhs            D H A H G p_pred  (A = C for Chorin-Teman, A = kin_vis K + NL  for Yosida)     
   opG->Mult(*p_pred,*Gp);            //         G p_pred = Gp   --> stored to be reused in Velocity correction
   H1->Mult(*Gp,*tmp2);               //       H G p_pred = tmp2
   opC->Mult(*tmp2,*tmp1);            //     A H G p_pred = tmp1
   H1->Mult(*tmp1,*tmp2);             //   H A H G p_pred = tmp2
   opD->Mult(*tmp2,*rhs_p2);          // D H A H G p_pred = tmp1

   // Apply bcs
   DHGc->EliminateRHS(*p,*rhs_p2);   

   // Solve current iteration.
   invDHG2->Mult(*rhs_p2,*p);
   iter_p2solve = invDHG2->GetNumIterations();
   res_p2solve = invDHG2->GetFinalNorm(); 
   MFEM_VERIFY(invDHG2->GetConverged(), "Pressure correction step did not converge. Aborting!");

   // Remove nullspace by removing mean of the pressure solution 
   p_gf->SetFromTrueDofs(*p);  

   sw_pres_corr.Stop();

   //////////////////////////////////////////////
   /// 4. Solve velocity correction step      ///
   //////////////////////////////////////////////
   /// u = u_pred - H2 G p    (G p_pred for Chorin Temam)
 
   sw_vel_corr.Start();
          
   //opG->Mult(*p,*Gp);            //         G p  and not G p_pred    TODO: Check why this causes error
   H2->Mult(*Gp,*u);
   u->Neg();
   u->Add(1.0,*u_pred);
   iter_v2solve = H2->GetNumIterations();
   res_v2solve = H2->GetFinalNorm();
   MFEM_VERIFY(H2->GetConverged(), "Velocity correction step did not converge. Aborting!");

   sw_vel_corr.Stop();

   //// 5. Relaxation (velocity)  u = gamma u + (1 - gamma) un
   //add(gamma,*u,(1.0-gamma),*un,*u);
   
   u_gf->SetFromTrueDofs(*u);

   // Print summary
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(13) << "Step" << std::setw(10) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      mfem::out << std::setw(10) << "Velocity Prediction " << std::setw(5) << std::fixed
                   << iter_v1solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_v1solve
                   << "   " << s1Params.rtol << "\n";
      mfem::out << std::setw(10) << "Pressure Prediction " << std::setw(5) << std::fixed
                   << iter_p1solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_p1solve
                   << "   " << s2Params.rtol << "\n";
      mfem::out << std::setw(10) << "Pressure Correction " << std::setw(5) << std::fixed
                   << iter_p2solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_p2solve
                   << "   " << s3Params.rtol << "\n";
      mfem::out << std::setw(10) << "Velocity Correction " << std::setw(5) << std::fixed
                   << iter_v2solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_v2solve
                   << "   " << s4Params.rtol << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }

   // Update solution at previous timesteps and time
   UpdateSolution(); 

}

////////////////////////////////////////////////////////////////////////////////////////////////////////

HighOrderYosidaSolver::HighOrderYosidaSolver(std::shared_ptr<ParMesh> pmesh_, BCHandler *bcs, real_t kin_vis_, int uorder_, int porder_, bool verbose_)
    : NavierUnsteadySolver(pmesh_, bcs, kin_vis_, uorder_, porder_, verbose_)
{
   // Set correction order
   correction_order = 1;

   // Initialize z for storing the intermediate solutions 
   z1 = new Vector(pdim); *z1 = 0.0;
}

HighOrderYosidaSolver::HighOrderYosidaSolver(std::shared_ptr<ParMesh> pmesh_, BCHandler *bcs, real_t kin_vis_, int uorder_, int porder_, bool verbose_, int correction_order_)
    : NavierUnsteadySolver(pmesh_, bcs, kin_vis_, uorder_, porder_, verbose_)
{
   // Set correction order
   correction_order = correction_order_;

   // Initialize z for storing the intermediate solutions 
   z1 = new Vector(pdim); *z1 = 0.0;

   if (correction_order > 1)
   {
      z2 = new Vector(pdim); *z2 = 0.0;
   }
   
}

void HighOrderYosidaSolver::UpdateTimeBCS(real_t time)
{
   // Update boundary conditions
   NavierUnsteadySolver::UpdateTimeBCS(time);

   // Modify intermediate solutions
   z1->SetSubVector(pres_ess_tdof, 0.0);

   if (correction_order > 1)
   {
      z2->SetSubVector(pres_ess_tdof, 0.0);
   }
}


void HighOrderYosidaSolver::Setup(real_t dt, int prec_type_)
{
   NavierUnsteadySolver::Setup(dt, prec_type_);

   // Assign solver for operator H2
   // Chorin-Temam operator H2 = dt/alpha M^{-1}, Yosida C^{-1}
   H2 = new GMRESSolver(ufes->GetComm());

   H2->iterative_mode = true;        
   H2->SetAbsTol(s4Params.atol);
   H2->SetRelTol(s4Params.rtol);
   H2->SetMaxIter(s4Params.maxIter);
   H2->SetPreconditioner(*H2_pc);
   H2->SetPrintLevel(s4Params.pl);
}

void HighOrderYosidaSolver::Step(real_t &time, real_t dt, int current_step)
{
   /// 0.1 Update BDF time integration coefficients
   SetTimeIntegrationCoefficients( current_step );

   /// 0.2 Update time coefficients for rhs and bcs
   time += dt;
   UpdateTimeBCS( time );
   UpdateTimeRHS( time );

   ///////////////////////////////////////////////////////////////////////
   /// 0. Assemble Convective term (linearized: u_ext \cdot \grad u)   ///
   ///////////////////////////////////////////////////////////////////////

   sw_conv_assembly.Start();

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
   NL_form->AddDomainIntegrator(new VectorConvectionIntegrator(*u_ext_vc, 1.0));  // += C 
   NL_form->Assemble(skip_zeros); 
   NL_form->FormSystemMatrix(empty, opNL);
   
   C_visccoeff.constant = kin_vis->constant;
   C_bdfcoeff.constant = alpha / dt;
   auto Amat = Add(C_visccoeff.constant, *(opK.As<HypreParMatrix>()), 1.0, *(opNL.As<HypreParMatrix>()));
   auto Cmat = Add(C_bdfcoeff.constant, *(opM.As<HypreParMatrix>()), 1.0, *Amat);
   opA.Reset( Amat, true );   // A = kin_vis K + NL 
   opC.Reset( Cmat, true );     // C = alpha/dt M + A
   Ce = opC.As<HypreParMatrix>()->EliminateRowsCols(vel_ess_tdof);
   invC->SetOperator(*opC);

   auto opAe = opA.As<HypreParMatrix>()->EliminateRowsCols(vel_ess_tdof);
   delete opAe;
   
   // Assemble solver for H = inv( alpha/dt M )
   delete sigmaM; 
   sigmaM = new HypreParMatrix(*(opM.As<HypreParMatrix>()));
   *sigmaM *= C_bdfcoeff.constant;
   auto sigmaMe = sigmaM->EliminateRowsCols(vel_ess_tdof);  
   delete sigmaMe; 
   H1->SetOperator(*sigmaM); 
   H2->SetOperator(*opC); 

   // Update timestep for discrete pressure laplacian preconditioner
   switch (pc_type)
   {
   case 0:
      static_cast<PMassPC *>(invDHG_pc)->SetCoefficients(dt);
      break;
   case 1: // Pressure Laplacian
      break;
   case 2: // PCD
      pc_builder->RebuildPreconditioner();
      invDHG_pc = pc_builder->GetSolver();
      break;
   case 3: // Cahouet-Chabard
      static_cast<CahouetChabardPC *>(invDHG_pc)->SetCoefficients(dt, C_visccoeff.constant);
      break;
   case 4: // Approximate inverse
      static_cast<SchurApproxInvPC *>(invDHG_pc)->SetCoefficients(dt);
      break;
   default:
      MFEM_ABORT("NavierStokesOperator::Assemble() >> Unknown preconditioner type: " << pc_type);
      break;
   }

   sw_conv_assembly.Stop();

   //////////////////////////////////////////////
   /// 1. Solve velocity prediction step      ///
   //////////////////////////////////////////////
   /// C u_pred = fv + 1/dt M u_bdf
   
   sw_vel_pred.Start(); 

   // Assemble rhs     fv + 1/dt M u_bdf
   add(a1, *un, a2, *un1, *u_bdf);     
   u_bdf->Add(a3,*un2);

   rhs_v1->Set(1.0,*fv);               
   opM->Mult(*u_bdf,*tmp1);
   rhs_v1->Add(1.0/dt, *tmp1);

   // Apply bcs
   opC.As<HypreParMatrix>()->EliminateBC(*Ce,vel_ess_tdof,*u_pred,*rhs_v1); // rhs_v1 -= Ce*u_pred

   // Solve current iteration.
   invC->Mult(*rhs_v1, *u_pred);
   iter_v1solve = invC->GetNumIterations();
   res_v1solve = invC->GetFinalNorm();
   MFEM_VERIFY(invC->GetConverged(), "Velocity prediction step did not converge. Aborting!");

   // Update gf 
   u_pred_gf->SetFromTrueDofs(*u_pred);
 
   sw_vel_pred.Stop();

   //////////////////////////////////////////////
   /// 2. Solve pressure prediction step      ///
   //////////////////////////////////////////////
   /// DHG p_pred = D u_pred - fp

   sw_pres_pred.Start(); 

   // Assemble rhs            D u_pred - fp   = D u_pred + De u_pred
   opD->Mult(*u_pred,*rhs_p1);
   rhs_p1->Add(-1.0,*fp);

   // Apply bcs
   DHGc->EliminateRHS(*p_pred,*rhs_p1);

   // Solve current iteration.
   invDHG1->Mult(*rhs_p1,*p_pred);
   iter_p1solve = invDHG1->GetNumIterations();
   res_p1solve = invDHG1->GetFinalNorm();
   MFEM_VERIFY(invDHG1->GetConverged(), "Pressure prediction step did not converge. Aborting!");

   // Remove nullspace by removing mean of the pressure solution 
   p_pred_gf->SetFromTrueDofs(*p_pred);  
   if( (bcs->GetPresDbcs()).empty())
   {
      MeanZero(*p_pred_gf);
      p_pred_gf->GetTrueDofs(*p_pred);
   }

   sw_pres_pred.Stop();

   //////////////////////////////////////////////
   /// 3. Solve pressure correction step      ///
   //////////////////////////////////////////////
   /// DHG p = D H A H G p_pred

   sw_pres_corr.Start();
   
   /// First pressure correction
   // Assemble rhs            D H A H G p_pred  A = C
   opG->Mult(*p_pred,*Gp);            //         G p_pred = Gp   
   H1->Mult(*Gp,*tmp2);               //       H G p_pred = tmp2
   opA->Mult(*tmp2,*tmp1);            //     A H G p_pred = tmp1
   H1->Mult(*tmp1,*tmp2);             //   H A H G p_pred = tmp2
   opD->Mult(*tmp2,*rhs_p2);          // D H A H G p_pred = tmp1

   // Apply bcs
   DHGc->EliminateRHS(*z1,*rhs_p2);   

   // Solve current iteration.
   invDHG2->Mult(*rhs_p2,*z1);
   iter_p2solve = invDHG2->GetNumIterations();
   res_p2solve = invDHG2->GetFinalNorm(); 
   MFEM_VERIFY(invDHG2->GetConverged(), "First Pressure correction step did not converge. Aborting!");


   /// Second pressure correction
   if (correction_order > 1)
   {
      // Assemble rhs            D2 p_pred + D1 z1,  with D2 = -D (-H A)^2 H G, D1 = D H A H G
      // D1 z1
      opG->Mult(*z1,*Gp);            //         G z1 = Gp
      H1->Mult(*Gp,*tmp2);           //       H G z1 = tmp2
      opA->Mult(*tmp2,*tmp1);        //     A H G z1 = tmp1
      H1->Mult(*tmp1,*tmp2);         //   H A H G z1 = tmp2
      opD->Mult(*tmp2,*rhs_p2);      // D H A H G z1 = tmp1

      // D2 p_pred = D2 = -D (-H A)^2 H G p_pred
      opG->Mult(*p_pred,*Gp);            //             G p_pred = Gp
      H1->Mult(*Gp,*tmp2);               //           H G p_pred = tmp2
      opA->Mult(*tmp2,*tmp1);            //         A H G p_pred = tmp1
      H1->Mult(*tmp1,*tmp2);             //       H A H G p_pred = tmp2
      opA->Mult(*tmp2,*tmp1);            //     A H A H G p_pred = tmp1
      H1->Mult(*tmp1,*tmp2);             //   H A H A H G p_pred = tmp2
      opD->Mult(*tmp2,*tmp1);            //  D H A H A H G p_pred = tmp1

      // D2 p_pred + D1 z1
      rhs_p2->Add(-1.0,*tmp1);           //  D1 z1 -D H A H A H G p_pred

      // Apply bcs
      DHGc->EliminateRHS(*z2,*rhs_p2);

      // Solve current iteration.
      invDHG2->Mult(*rhs_p2,*z2);
      iter_p2solve = invDHG2->GetNumIterations();
      res_p2solve = invDHG2->GetFinalNorm();
      MFEM_VERIFY(invDHG2->GetConverged(), "Second Pressure correction step did not converge. Aborting!");

   }

   // Remove nullspace by removing mean of the pressure solution 
   *p = *p_pred;
   p->Add(1.0,*z1);  // p = p_pred + z1

   if (correction_order > 1)
   {
      p->Add(1.0,*z2);  // p = p_pred + z1 + z2
   }

   p_gf->SetFromTrueDofs(*p);  

   sw_pres_corr.Stop();

   //////////////////////////////////////////////
   /// 4. Solve velocity correction step      ///
   //////////////////////////////////////////////
   /// u = u_pred - H2 G p    (G p_pred for Chorin Temam)
 
   sw_vel_corr.Start();
          
   // opG->Mult(*p,*Gp);  TODO: Check why this causes error
   H2->Mult(*Gp,*u);
   u->Neg();
   u->Add(1.0,*u_pred);
   iter_v2solve = H2->GetNumIterations();
   res_v2solve = H2->GetFinalNorm();
   MFEM_VERIFY(H2->GetConverged(), "Velocity correction step did not converge. Aborting!");

   sw_vel_corr.Stop();

   //// 5. Relaxation (velocity)  u = gamma u + (1 - gamma) un
   //add(gamma,*u,(1.0-gamma),*un,*u);
   
   u_gf->SetFromTrueDofs(*u);

   // Print summary
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(13) << "Step" << std::setw(10) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      mfem::out << std::setw(10) << "Velocity Prediction " << std::setw(5) << std::fixed
                   << iter_v1solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_v1solve
                   << "   " << s1Params.rtol << "\n";
      mfem::out << std::setw(10) << "Pressure Prediction " << std::setw(5) << std::fixed
                   << iter_p1solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_p1solve
                   << "   " << s2Params.rtol << "\n";
      mfem::out << std::setw(10) << "Pressure Correction " << std::setw(5) << std::fixed
                   << iter_p2solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_p2solve
                   << "   " << s3Params.rtol << "\n";
      mfem::out << std::setw(10) << "Velocity Correction " << std::setw(5) << std::fixed
                   << iter_v2solve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_v2solve
                   << "   " << s4Params.rtol << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }

   // Update solution at previous timesteps and time
   UpdateSolution(); 

}


HighOrderYosidaSolver::~HighOrderYosidaSolver()
{
   delete z1;
   if (correction_order > 1)
   {
      delete z2;
   }
}

} // namespace navier
} // namespace mfem