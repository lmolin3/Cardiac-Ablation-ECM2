#include "navier_unsteady_solver.hpp"
#include <fstream>
#include <iomanip>


namespace mfem{

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
   
   // initialize vectors of essential attributes
   vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());      vel_ess_attr=0;
   vel_ess_attr_x.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_x=0;
   vel_ess_attr_y.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_y=0;
   vel_ess_attr_z.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_z=0;
   pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());    pres_ess_attr=0;
   ess_attr_tmp.SetSize(pmesh->bdr_attributes.Max());      ess_attr_tmp=0;
   custom_traction_attr.SetSize(pmesh->bdr_attributes.Max());    custom_traction_attr=0;
   traction_attr.SetSize(pmesh->bdr_attributes.Max());    traction_attr=0;

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
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i]) && attr[i]) == 0,
                           "Duplicate boundary definition for x component detected.");
               if (attr[i] == 1){vel_ess_attr_x[i] = 1;}
               break;
            case 1: // y
               dir_string = "y";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_y[i]) && attr[i]) == 0,
                           "Duplicate boundary definition for y component detected.");
               if (attr[i] == 1){vel_ess_attr_y[i] = 1;}
               break;
            case 2: // z
               dir_string = "z";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
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

void NavierUnsteadySolver::AddPresDirichletBC(Coefficient *coeff, Array<int> &attr)
{
   pres_dbcs.emplace_back(attr, coeff);

   // Check for duplicate                                                                                                                                                                            
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT( (pres_ess_attr[i] && attr[i]) == 0,
                  "Duplicate boundary definition detected.");
      if (attr[i] == 1)                           
      {
         pres_ess_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Pressure Dirichlet BC (full) to boundary attributes: ";
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

void NavierUnsteadySolver::AddPresDirichletBC(ScalarFuncT func, Array<int> &attr)
{
   AddPresDirichletBC(new FunctionCoefficient(func), attr);
}

void NavierUnsteadySolver::AddPresDirichletBC(Coefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddPresDirichletBC(coeff, ess_attr_tmp);
}

void NavierUnsteadySolver::AddPresDirichletBC(ScalarFuncT func, int &attr)
{
   AddPresDirichletBC(new FunctionCoefficient(func), attr);
}


void NavierUnsteadySolver::AddTractionBC(VectorCoefficient *coeff, Array<int> &attr)
{
   traction_bcs.emplace_back(attr, coeff);

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i] || pres_ess_attr[i] ) && attr[i]) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
      MFEM_ASSERT( (custom_traction_attr[i] ) == 0,
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

void NavierUnsteadySolver::AddTractionBC(VecFuncT func, int &attr)
{
   AddTractionBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
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
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);


   /// 2. Setup and assemble bilinear forms 
   int skip_zeros = 0;

   // Velocity laplacian K (not scaled by viscosity --> done in assembly of Momentum operator)
   K_form = new ParBilinearForm(ufes);  
   K_form->AddDomainIntegrator(new VectorDiffusionIntegrator());
   K_form->Assemble(skip_zeros); K_form->Finalize(skip_zeros);
   K = K_form->ParallelAssemble();

   // Velocity mass (not modified with bcs)
   M_form = new ParBilinearForm(ufes);  
   M_form->AddDomainIntegrator(new VectorMassIntegrator());
   M_form->Assemble(skip_zeros); M_form->Finalize(skip_zeros);
   M = M_form->ParallelAssemble();
   sigmaM = new HypreParMatrix(*M); // copy of mass matrix used for operator H in Schur Complement

   // Divergence
   D_form = new ParMixedBilinearForm(ufes, pfes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator());
   D_form->Assemble(); D_form->Finalize();
   D = D_form->ParallelAssemble();
   *(D) *= -1.0; 
   De = D->EliminateCols(vel_ess_tdof);

   // Gradient                         // NOTE: 1) We can replace with Dt and avoid assembly. 2) Also we can just maybe use MultTranspose D, and RAP operator with G instead of TripleProductOperator
   G = new HypreParMatrix();                              
   G_form = new ParMixedBilinearForm(pfes, ufes);
   G_form->AddDomainIntegrator(new GradientIntegrator());
   G_form->Assemble(); G_form->Finalize();
   G = G_form->ParallelAssemble();
   G->EliminateRows(vel_ess_tdof);

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
      f_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator( *(traction_bc.coeff)) , traction_bc.attr);
   }

   // Adding custom traction bcs
   for (auto &traction_bc : custom_traction_bcs)
   {
      f_form->AddBoundaryIntegrator(new VectorNeumannLFIntegrator( *(traction_bc.u),*(traction_bc.p),*(traction_bc.alpha),*(traction_bc.beta)) , traction_bc.attr);
   }
   
   // Update time in Dirichlet velocity coefficients and project on predicted/corrected vectors and gf
   UpdateTimeBCS( 0.0 );
   updatedBCS = false;

   // Update time in coefficients for rhs (acceleration/neumann) and assemble vector fv
   UpdateTimeRHS( 0.0 );
   updatedRHS = false;


   /// 4. Construct the operators for preconditioners
   //
   //     Here we use BoomerAMG for:
   //     * convective-diffusive part C =  alpha/dt M + kin_vis K + NL,
   //     * velocity mass matrix M
   //     * pressure laplacian (for schur complement) 
   //

   // Pressure Schur Complement ( BoomerAMG with pressure laplacian matrix S)
   S = new HypreParMatrix();
   S_form = new ParBilinearForm(pfes);
   S_visccoeff.constant = 1.0 / kin_vis;
   S_bdfcoeff.constant = dt ;
   //S_form->AddDomainIntegrator(new MassIntegrator( S_visccoeff ));
   //S_form->AddDomainIntegrator(new DiffusionIntegrator( S_bdfcoeff ));
   S_form->AddDomainIntegrator(new DiffusionIntegrator());
   S_form->Assemble(); 
   S_form->FormSystemMatrix(pres_ess_tdof, *S); 
   invDHG_pc = new HypreBoomerAMG(*S);    
   invDHG_pc->SetPrintLevel(0);
   invDHG_pc->SetSystemsOptions(dim);
   invDHG_pc->iterative_mode = false;

   // Velocity Correction ( BoomerAMG )
   H1_pc = new HypreBoomerAMG();        
   H1_pc->SetPrintLevel(0);
   H1_pc->SetSystemsOptions(dim);
   H1_pc->iterative_mode = false;

   // Velocity Prediction (BoomerAMG; operator will be assigned inside loop)
   invC_pc = new HypreBoomerAMG();
   dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetSystemsOptions(dim);
   //dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetElasticityOptions(ufes);
   //dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetAdvectiveOptions(1, "", "FA"); // AIR solver
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

   // Chorin-Temam operator H2 = dt/alpha M^{-1}
   H1 = new CGSolver(ufes->GetComm());
   H1->iterative_mode = false;         // keep it to false since it's in the TripleProductOperator    
   H1->SetAbsTol(s4Params.atol);
   H1->SetRelTol(s4Params.rtol);
   H1->SetMaxIter(s4Params.maxIter);
   H1->SetPreconditioner(*H1_pc);
   H1->SetOperator(*sigmaM);
   H1->SetPrintLevel(s4Params.pl);

   // Chorin-Temam operator H2 = dt/alpha M^{-1}
   H2 = new CGSolver(ufes->GetComm());
   H2->iterative_mode = true;        
   H2->SetAbsTol(s4Params.atol);
   H2->SetRelTol(s4Params.rtol);
   H2->SetMaxIter(s4Params.maxIter);
   H2->SetPreconditioner(*H1_pc);
   H2->SetOperator(*sigmaM);
   H2->SetPrintLevel(s4Params.pl);

   // Pressure prediction 
   DHG = new TripleProductOperator(D,H1,G,false,false,false);      // operator through action: DHG = D  dt/alpha M^{-1} G
   DHGc = new ConstrainedOperator(DHG, pres_ess_tdof, true);       // operator DHG constraining pressure dofs

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
   // NOTE: we can create method for extrapolating that incorporates the SetTimeIntegrationCoefficients,
   // receives vectors, computes coefficients, return extrap (can be used to extrapolate pressure for Incremental version)
   add(b1, *un, b2, *un1, *u_ext);
   u_ext->Add(b3,*un2);
   u_ext_gf->SetFromTrueDofs(*u_ext);  

   int skip_zeros = 0;
   delete NL_form; NL_form = nullptr;
   delete NL; NL = nullptr;
   delete C; C = nullptr;
   delete Ce; Ce = nullptr;
   NL_form = new ParBilinearForm(ufes);
   NL_form->AddDomainIntegrator(new VectorConvectionIntegrator(*u_ext_vc, 1.0));  // += C 
   NL_form->Assemble(skip_zeros); NL_form->Finalize(skip_zeros);
   NL = NL_form->ParallelAssemble();  
   
   C_visccoeff.constant = kin_vis;
   C_bdfcoeff.constant = alpha / dt;
   C = Add(C_bdfcoeff.constant, *M, C_visccoeff.constant, *K);   // C = alpha/dt M + kin_vis K + NL
   C->Add(1.0, *NL);  
   Ce = C->EliminateRowsCols(vel_ess_tdof);
   invC->SetOperator(*C);               

   sw_conv_assembly.Stop();

#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "conv_assembly", current_step);} // Export matrices/vectors
#endif

   /// 1. Solve velocity prediction step

   sw_vel_pred.Start(); 

   // Assemble rhs     fv + 1/dt M u_bdf
   add(a1, *un, a2, *un1, *u_bdf);     
   u_bdf->Add(a3,*un2);

   rhs_v1->Set(1.0,*fv);               
   M->Mult(*u_bdf,*tmp1);
   rhs_v1->Add(1.0/dt, *tmp1);

   // Apply bcs
   C->EliminateBC(*Ce,vel_ess_tdof,*u_pred,*rhs_v1); // rhs_v1 -= Ce*u_pred

   // Solve current iteration.
   invC->Mult(*rhs_v1, *u_pred);
   iter_v1solve = invC->GetNumIterations();
   res_v1solve = invC->GetFinalNorm();
   MFEM_VERIFY(invC->GetConverged(), "Velocity prediction step did not converge. Aborting!");

   // Update gf 
   u_pred_gf->SetFromTrueDofs(*u_pred);
 
   sw_vel_pred.Stop();

#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "vel_pred", current_step);} // Export matrices/vectors
#endif

   /// 2. Solve pressure prediction step

   sw_pres_pred.Start();

   // Assemble solver for H = inv( alpha/dt M )
   delete sigmaM; 
   sigmaM = new HypreParMatrix(*M);
   *sigmaM *= C_bdfcoeff.constant;
   auto sigmaMe = sigmaM->EliminateRowsCols(vel_ess_tdof);  
   delete sigmaMe; 
   H1->SetOperator(*sigmaM);     

   // Assemble rhs            D u_pred - fp   = D u_pred + De u_pred
   D->Mult(*u_pred,*rhs_p1);
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
   if(pres_dbcs.empty())
   {
      MeanZero(*p_pred_gf);
      p_pred_gf->GetTrueDofs(*p_pred);
   }

   sw_pres_pred.Stop();

#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "pres_pred", current_step);} // Export matrices/vectors
#endif

   /// 3. Solve pressure correction step

   sw_pres_corr.Start();
   
   // Assemble rhs            D H A H G p_pred      
   G->Mult(*p_pred,*Gp);            //         G p_pred = Gp   --> stored to be reused in Velocity correction
   H1->Mult(*Gp,*tmp2);             //       H G p_pred = tmp2
   C->Mult(*tmp2,*tmp1);            //     C H G p_pred = tmp1
   H1->Mult(*tmp1,*tmp2);           //   H C H G p_pred = tmp2
   D->Mult(*tmp2,*rhs_p2);          // D H C H G p_pred = tmp1

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

#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "pres_corr", current_step);} // Export matrices/vectors 
#endif

   /// 4. Solve velocity correction step

   sw_vel_corr.Start();

   // Assemble rhs           ( alpha/dt M) u_pred - G p_pred 
   H2->SetOperator(*sigmaM);    
   sigmaM->Mult(*u_pred,*rhs_v2); 
   rhs_v2->Add(-1.0,*Gp); 

   // Solve current iteration.
   H2->Mult(*rhs_v2,*u);  
   iter_v2solve = H2->GetNumIterations();
   res_v2solve = H2->GetFinalNorm();
   MFEM_VERIFY(H2->GetConverged(), "Velocity correction step did not converge. Aborting!");

   sw_vel_corr.Stop();

   // 5. Relaxation (velocity)  u = gamma u + (1 - gamma) un
   add(gamma,*u,(1.0-gamma),*un,*u);
   u_gf->SetFromTrueDofs(*u);


#ifdef MFEM_DEBUG
   if( ExportData ){PrintMatricesVectors( "vel_corr", current_step);} // Export matrices/vectors 
#endif

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
   *un  = *u;
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
      De->AddMult(*u, *fp, -1.0);

      updatedRHS = true;
   }
}

void NavierUnsteadySolver::UpdateTimeBCS( double new_time )
{
   if ( !updatedBCS )
   {
      // Projection of coeffs (pressure)
      for (auto &pres_dbc : pres_dbcs)
      {
         pres_dbc.coeff->SetTime(new_time);
         p_gf->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
         p_pred_gf->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
      }
      p_gf->GetTrueDofs(*p);
      p_pred_gf->GetTrueDofs(*p_pred);

      // Projection of coeffs (full velocity applied)
      for (auto &vel_dbc : vel_dbcs)
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
            (*u)[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
            (*u_pred)[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
         }      
      }
      // Initialize solution gf with vector containing projected coefficients 
      // and update grid function and vector for provisional velocity
      u_gf->SetFromTrueDofs(*u);
      u_pred_gf->SetFromTrueDofs(*u_pred);

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
   my_rt[2] = sw_vel_pred.RealTime();
   my_rt[3] = sw_pres_pred.RealTime();
   my_rt[4] = sw_pres_corr.RealTime();
   my_rt[5] = sw_vel_corr.RealTime();

   MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, ufes->GetComm());

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

#ifdef MFEM_DEBUG

void NavierUnsteadySolver::PrintMatricesVectors( const char* id, int num ){}
/*{
   // Create folder
   std::string folderName(outfolder);
   folderName += "/MatVecs_iter";
   folderName += std::to_string(num);

   if (mkdir(folderName.c_str(), 0777) == -1) {} //{mfem::err << "Error :  " << strerror(errno) << std::endl;}

   //Create files
   std::ofstream K_file(std::string(folderName) + '/' + "K_" + std::string(id) + ".dat");
   std::ofstream C_file(std::string(folderName) + '/' + "C_" + std::string(id) + ".dat");
   std::ofstream M_file(std::string(folderName) + '/' + "M_" + std::string(id) + ".dat");
   std::ofstream NL_file(std::string(folderName) + '/' + "NL_" + std::string(id) + ".dat");
   std::ofstream D_file(std::string(folderName) + '/' + "D_" + std::string(id) + ".dat");
   std::ofstream Dt_file(std::string(folderName) + '/' + "Dt_" + std::string(id) + ".dat");

   std::ofstream De_file(std::string(folderName) + '/' + "De_" + std::string(id) + ".dat");
   std::ofstream Dte_file(std::string(folderName) + '/' + "Dte_" + std::string(id) + ".dat");
   std::ofstream Ce_file(std::string(folderName) + '/' + "Ce_" + std::string(id) + ".dat");
   std::ofstream Me_file(std::string(folderName) + '/' + "Me_" + std::string(id) + ".dat");

   std::ofstream u_file(std::string(folderName) + '/' + "u_" + std::string(id) + ".dat");
   std::ofstream upred_file(std::string(folderName) + '/' + "upred_" + std::string(id) + ".dat");
   std::ofstream p_file(std::string(folderName) + '/' + "p_" + std::string(id) + ".dat");
   std::ofstream ppred_file(std::string(folderName) + '/' + "ppred_" + std::string(id) + ".dat");
   std::ofstream un_file(std::string(folderName) + '/' + "un_" + std::string(id) + ".dat");
   std::ofstream un1_file(std::string(folderName) + '/' + "un1_" + std::string(id) + ".dat");
   std::ofstream un2_file(std::string(folderName) + '/' + "un2_" + std::string(id) + ".dat");
   std::ofstream uext_file(std::string(folderName) + '/' + "uext_" + std::string(id) + ".dat");
   std::ofstream ubdf_file(std::string(folderName) + '/' + "ubdf_" + std::string(id) + ".dat");

   std::ofstream fv_file(std::string(folderName) + '/' + "fv_" + std::string(id) + ".dat");
   std::ofstream fp_file(std::string(folderName) + '/' + "fp_" + std::string(id) + ".dat");
   std::ofstream rhs_v1_file(std::string(folderName) + '/' + "rhs_v1_" + std::string(id) + ".dat");
   std::ofstream rhs_v2_file(std::string(folderName) + '/' + "rhs_v2_" + std::string(id) + ".dat");
   std::ofstream rhs_p1_file(std::string(folderName) + '/' + "rhs_p1_" + std::string(id) + ".dat");
   std::ofstream rhs_p2_file(std::string(folderName) + '/' + "rhs_p2_" + std::string(id) + ".dat");

   std::ofstream dofs_file(std::string(folderName) + '/' + "dofs_" + std::string(id) + ".dat");




   // Print matrices in matlab format
   M->PrintMatlab(M_file);

   if(C==nullptr)
   {
      C = new HypreParMatrix();
      C->PrintMatlab(C_file);
      delete C; C = nullptr;
   }
   else
   {
      C->PrintMatlab(C_file);
   }

   if(NL==nullptr)
   {
      NL = new HypreParMatrix();
      NL->PrintMatlab(NL_file);
      delete NL; NL = nullptr;
   }
   else
   {
      NL->PrintMatlab(NL_file);
   }

   D->PrintMatlab(D_file);
   G->PrintMatlab(Dt_file);

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
   u->Print(u_file, 1);
   u_pred->Print(upred_file, 1);
   p->Print(p_file, 1);
   p_pred->Print(ppred_file, 1);
   un->Print(un_file, 1);
   un1->Print(un1_file, 1);
   un2->Print(un2_file, 1);
   u_ext->Print(uext_file, 1);
   u_bdf->Print(ubdf_file, 1);
   fv->Print(fv_file, 1);
   fp->Print(fp_file, 1);
   rhs_v1->Print(rhs_v1_file, 1);
   rhs_v2->Print(rhs_v2_file, 1);
   rhs_p1->Print(rhs_p1_file, 1);
   rhs_p2->Print(rhs_p2_file, 1);

   for (int i = 0; i < vel_ess_tdof.Size(); ++i)
   {
      dofs_file << vel_ess_tdof[i] << std::endl;
   }
   dofs_file.close();

} */

#endif

/// Destructor
NavierUnsteadySolver::~NavierUnsteadySolver()
{
   delete ufec; ufec = nullptr;
   delete pfec; pfec = nullptr;
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

   delete K;         K = nullptr;
   delete M;         M = nullptr;
   delete D;         D = nullptr;
   delete G;         G = nullptr;
   delete C;         C = nullptr;
   delete S;         S = nullptr;    
   delete sigmaM;   sigmaM = nullptr; 
   delete Ce;        Ce = nullptr;
   delete De;        De = nullptr;
   delete DHGc;      DHGc = nullptr;
   delete mass_lf;   mass_lf = nullptr;

   delete invC;              invC = nullptr;
   delete H1;                  H1 = nullptr;
   delete H2;                  H2 = nullptr;
   delete invDHG1;        invDHG1 = nullptr;
   delete invDHG2;        invDHG2 = nullptr;
   delete invC_pc;        invC_pc = nullptr;
   delete invDHG_pc;  invDHG_pc = nullptr;
   delete H1_pc;        H1_pc = nullptr;

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

