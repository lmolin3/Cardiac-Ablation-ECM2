#include "navierstokes_operator.hpp"
#include "utils.hpp"

using namespace mfem;

NavierStokesOperator::NavierStokesOperator(std::shared_ptr<ParMesh> mesh,
                                           ParFiniteElementSpace *ufes,
                                           ParFiniteElementSpace *pfes,
                                           double kin_vis,
                                           std::shared_ptr<BCHandler> bcs,
                                           bool verbose) : TimeDependentOperator(ufes->GetTrueVSize()),
                                                           pmesh(mesh),
                                                           ufes(ufes),
                                                           pfes(pfes),
                                                           kin_vis(std::make_unique<ConstantCoefficient>(kin_vis)),
                                                           bcs(bcs),
                                                           verbose(verbose),
                                                           intrules(0, Quadrature1D::GaussLobatto),
                                                           zero_coeff(0.0)
{
   // mesh
   dim = pmesh->Dimension();

   // determine spaces dimension (algebraic tdofs)
   udim = ufes->GetTrueVSize();
   pdim = pfes->GetTrueVSize();

   // Initialize the offsets
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = udim;
   offsets[2] = pdim;
   offsets.PartialSum();

   // Initialize the grid functions
   u_gf.reset(new ParGridFunction(ufes));
   *u_gf = 0.0;

   p_gf.reset(new ParGridFunction(pfes));
   *p_gf = 0.0;

   // Initialize temporary vectors
   z.SetSize(udim);
   z = 0.0;
   w.SetSize(udim);
   w = 0.0;
   Hdot.SetSize(pdim);
   Hdot = 0.0;
   rhs_p.SetSize(pdim);
   rhs_p = 0.0;

   // Create preconditioner for the Jacobian inversion in newton solver (HypreBoomerAMG)
   newton_pc.reset(new HypreBoomerAMG()); // Create empty preconditioner (Operator will be set by NewtonSolver calling GetGradient() )
   dynamic_cast<HypreBoomerAMG *>(newton_pc.get())->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(newton_pc.get())->SetSystemsOptions(dim);
   newton_pc->iterative_mode = false;

   // The nonlinear convective integrators use over-integration (dealiasing) as
   // a stabilization mechanism.
   ir_nl = intrules.Get(ufes->GetFE(0)->GetGeomType(),
                        (int)(ceil(1.5 * 2 * (ufes->GetOrder(0) + 1) - 3)));

   ir = intrules.Get(ufes->GetFE(0)->GetGeomType(),
                     (int)(2 * (ufes->GetOrder(0) + 1) - 3));

   ir_face = intrules.Get(ufes->GetFaceElement(0)->GetGeomType(),
                          (int)(2 * (ufes->GetOrder(0) + 1) - 3));
}

void NavierStokesOperator::MassMult(const Vector &x, Vector &y)
{
   const BlockVector xb(x.GetData(), offsets);
   BlockVector yb(y.GetData(), offsets);

   M->Mult(xb.GetBlock(0), yb.GetBlock(0));
   yb.GetBlock(1) = 0.0;
}

void NavierStokesOperator::Mult(const Vector &x, Vector &y) const
{
   if (eval_mode == EvalMode::ADDITIVE_TERM_1)
   {
      ExplicitMult(x, y);
   }
   else if (eval_mode == EvalMode::ADDITIVE_TERM_2)
   {
      ImplicitMult(x, y);
   }
   else
   {
      MFEM_ABORT("NavierStokesOperator::Mult >> unknown EvalMode");
   }
}

void NavierStokesOperator::ProjectVelocityDirichletBC(Vector &u)
{

   // Projection of coeffs (full velocity applied)
   for (auto &vel_dbc : bcs->GetVelDbcs())
   {
      u_gf->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }
   u_gf->GetTrueDofs(u);

   // Projection of coeffs (velocity component applied)
   ParGridFunction tmp_gf(ufes); // temporary velocity gf for projection
   Vector tmp_vec(udim);         // temporary velocity vector for projection
   Array<int> tmp_tdofs;
   for (auto &vel_dbc : bcs->GetVelDbcsXYZ())
   {
      VectorArrayCoefficient tmp_coeff(dim); // Set coefficient with right component
      tmp_coeff.Set(vel_dbc.dir, vel_dbc.coeff, false);
      tmp_gf.ProjectBdrCoefficient(tmp_coeff, vel_dbc.attr); // Project on dummy gf
      tmp_gf.GetTrueDofs(tmp_vec);

      ufes->GetEssentialTrueDofs(vel_dbc.attr, tmp_tdofs, vel_dbc.dir); // Update solution dofs
      for (int i = 0; i < tmp_tdofs.Size(); i++)
      {
         (u)[tmp_tdofs[i]] = tmp_vec[tmp_tdofs[i]];
      }
   }
   // Initialize solution gf with vector containing projected coefficients
   // and update grid function and vector for provisional velocity
   u_gf->SetFromTrueDofs(u);
}

void NavierStokesOperator::ProjectPressureDirichletBC(Vector &p)
{
   // Projection of coeffs (pressure)
   for (auto &pres_dbc : bcs->GetPresDbcs())
   {
      p_gf->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
   }

   p_gf->GetTrueDofs(p);
}

void NavierStokesOperator::SetTime(double t, TimeDependentOperator::EvalMode eval_mode_)
{
   this->eval_mode = eval_mode_;

   // Update time in boundary conditions
   bcs->UpdateTimeVelocityBCs(t);

   bcs->UpdateTimePressureBCs(t);

   // Update rhs (forcing term and traction)
   if (forcing_form != nullptr && eval_mode == EvalMode::ADDITIVE_TERM_1)
   {
      time = t;

      // Extract vector of traction bcs
      const auto &traction_bcs = bcs->GetTractionBcs();

      // Update traction bcs
      for (auto &traction_bc : traction_bcs)
      {
         traction_bc.coeff->SetTime(t);
      }

      // Update forcing term
      for (auto &accel_term : accel_terms)
      {
         accel_term.coeff->SetTime(t);
      }
      forcing_form->Update();
      forcing_form->Assemble();
      forcing_form->ParallelAssemble(fu_rhs);
   }
}

void NavierStokesOperator::SetOrder(int order)
{
   rk_order = order;
}

void NavierStokesOperator::SetTime(double t)
{
   SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
}

void NavierStokesOperator::SetTimeStep(const double new_dt)
{
   dt = new_dt;
   ns_residual->SetTimeStep(dt);

   // Update the preconditioner
   switch (pc_type)
   {
   case 0:
      break;
   case 1: // Pressure Laplacian
      static_cast<PLapPC *>(invS_pc)->SetCoefficients(new_dt);
      break;
   case 2: // PCD
      break;
   case 3: // Cahouet-Chabard
      static_cast<CahouetChabardPC *>(invS_pc)->SetCoefficients(new_dt);
      break;
   case 4: // Approximate inverse
      static_cast<SchurApproxInvPC *>(invS_pc)->SetCoefficients(new_dt);
      break;
   default:
      MFEM_ABORT("NavierStokesOperator::Assemble() >> Unknown preconditioner type: " << pc_type);
      break;
   }
}

void NavierStokesOperator::SetSolvers(SolverParams params_p, SolverParams params_m, int pc_type)
{
   this->params_p = params_p;
   this->params_m = params_m;
   this->pc_type = pc_type;
}

void NavierStokesOperator::SetImplicitCoefficient(const double coeff)
{
   ns_residual->SetImplicitCoefficient(coeff);
}

void NavierStokesOperator::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
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

void NavierStokesOperator::AddAccelTerm(VecFuncT func, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void NavierStokesOperator::Setup(double initial_dt)
{
   // Extract to list of true dofs
   ufes->GetEssentialTrueDofs(bcs->GetVelEssAttr(), vel_ess_tdof_full);
   ufes->GetEssentialTrueDofs(bcs->GetVelEssAttrX(), vel_ess_tdof_x, 0);
   ufes->GetEssentialTrueDofs(bcs->GetVelEssAttrY(), vel_ess_tdof_y, 1);
   ufes->GetEssentialTrueDofs(bcs->GetVelEssAttrZ(), vel_ess_tdof_z, 2);
   vel_ess_tdof.Append(vel_ess_tdof_x);
   vel_ess_tdof.Append(vel_ess_tdof_y);
   vel_ess_tdof.Append(vel_ess_tdof_z);
   vel_ess_tdof.Append(vel_ess_tdof_full);
   pfes->GetEssentialTrueDofs(bcs->GetPresEssAttr(), pres_ess_tdof);

   // Setup the bilinear form for velocity mass
   M_form = new ParBilinearForm(ufes);
   BilinearFormIntegrator *integrator = new VectorMassIntegrator();
   integrator->SetIntRule(&ir);
   M_form->AddDomainIntegrator(integrator);

   // Setup the bilinear form for diffusion
   K_form = new ParBilinearForm(ufes);
   integrator = new VectorDiffusionIntegrator(*kin_vis);
   integrator->SetIntRule(&ir);
   K_form->AddDomainIntegrator(integrator);

   // Setup the bilinear form for divergence
   D_form = new ParMixedBilinearForm(ufes, pfes);
   integrator = new VectorDivergenceIntegrator();
   integrator->SetIntRule(&ir);
   D_form->AddDomainIntegrator(integrator);

   // Setup the bilinear form for gradient
   G_form = new ParMixedBilinearForm(pfes, ufes);
   integrator = new GradientIntegrator;
   integrator->SetIntRule(&ir);
   G_form->AddDomainIntegrator(integrator);

   // Setup the nonlinear form for convection
   NL_form = new ParNonlinearForm(ufes);
   NonlinearFormIntegrator *nl_integrator = new VectorConvectionNLFIntegrator;
   nl_integrator->SetIntRule(&ir_nl);
   NL_form->AddDomainIntegrator(nl_integrator);

   /// Setup form for forcing term
   forcing_form = new ParLinearForm(ufes);

   // Adding forcing terms
   VectorDomainLFIntegrator *forcing_integrator = nullptr;
   for (auto &accel_term : accel_terms)
   {
      forcing_integrator = new VectorDomainLFIntegrator(*(accel_term.coeff));
      forcing_integrator->SetIntRule(&ir_nl);
      forcing_form->AddDomainIntegrator(forcing_integrator, accel_term.attr);
   }

   // Setup initial dt for Navier-Stokes residual
   dt = initial_dt;
   ns_residual->SetTimeStep(dt);

   // Assemble all operators
   Assemble();
}

/// @brief Assemble all forms and matrices
void NavierStokesOperator::Assemble()
{

   int skip_zeros = 0; // To ensure same sparsity pattern matrices being added

   Array<int> empty;

   M_form->Update();
   M_form->Assemble(skip_zeros);
   M_form->Finalize(skip_zeros);
   M_form->FormSystemMatrix(empty, M);

   K_form->Update();
   K_form->Assemble(skip_zeros);
   K_form->Finalize(skip_zeros);
   K_form->FormSystemMatrix(empty, K);

   D_form->Update();
   D_form->Assemble();
   D_form->Finalize();
   D_form->FormRectangularSystemMatrix(empty, empty, D);

   G_form->Update();
   G_form->Assemble();
   G_form->Finalize();
   G_form->FormRectangularSystemMatrix(empty, empty, G);

   NL_form->Update();
   NL_form->Setup();

   // Assemble the forcing term
   fu_rhs.SetSize(ufes->GetTrueVSize());
   fu_rhs = 0.0;
   if (forcing_form != nullptr)
   {
      forcing_form->Update();
      forcing_form->Assemble();
      forcing_form->ParallelAssemble(fu_rhs);
   }

   // Setup Operator, Solver and Preconditioner for Schur Complement D M^{-1} G
   invM_pc = new HypreBoomerAMG();
   dynamic_cast<HypreBoomerAMG *>(invM_pc)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(invM_pc)->SetSystemsOptions(dim);
   // dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetElasticityOptions(ufes);
   // dynamic_cast<HypreBoomerAMG *>(invC_pc)->SetAdvectiveOptions(1, "", "FA"); // AIR solver
   invM_pc->iterative_mode = false;

   invM = new CGSolver(ufes->GetComm());
   invM->iterative_mode = false; // keep it to false since it's in the TripleProductOperator
   invM->SetAbsTol(params_m.atol);
   invM->SetRelTol(params_m.rtol);
   invM->SetMaxIter(params_m.maxIter);
   invM->SetPreconditioner(*invM_pc);
   invM->SetOperator(*M);
   invM->SetPrintLevel(params_m.pl);

   DHG = new DiscretePressureLaplacian(D.Ptr(), invM, G.Ptr(), false, false, false); // operator through action: -DM^{-1}G
   DHGc = new ConstrainedOperator(DHG, pres_ess_tdof, true);                         // operator constraining pressure dofs in -DM^{-1}G

   // Solver and Preconditioner for Schur Complement -D M^{-1} G
   switch (pc_type)
   {
   case 0: // Pressure Mass
      pc_builder = new PMassBuilder(*pfes, pres_ess_tdof);
      invS_pc = &pc_builder->GetSolver();
      break;
   case 1: // Pressure Laplacian
      pc_builder = new PLapBuilder(*pfes, pres_ess_tdof, dt);
      invS_pc = &pc_builder->GetSolver();
      break;
   case 2: // PCD
      pc_builder = new PCDBuilder(*pfes, pres_ess_tdof);
      invS_pc = &pc_builder->GetSolver();
      break;
   case 3: // Cahouet-Chabard
      pc_builder = new CahouetChabardBuilder(*pfes, pres_ess_tdof, dt);
      invS_pc = &pc_builder->GetSolver();
      break;
   case 4: // Approximate inverse
      pc_builder = new SchurApproxInvBuilder(*pfes, pres_ess_tdof, dt);
      invS_pc = &pc_builder->GetSolver();
      break;
   default:
      MFEM_ABORT("NavierStokesOperator::Assemble() >> Unknown preconditioner type: " << pc_type);
      break;
   }

   invS = new CGSolver(ufes->GetComm());
   invS->iterative_mode = true;
   invS->SetAbsTol(params_p.atol);
   invS->SetRelTol(params_p.rtol);
   invS->SetMaxIter(params_p.maxIter);
   invS->SetOperator(*DHGc);
   invS->SetPreconditioner(*invS_pc);
   invS->SetPrintLevel(params_p.pl);
}

void NavierStokesOperator::ImplicitSolve(Vector &b, Vector &x)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierStokesOperator: Solving implicit step" << std::endl;
   }
   
   BlockVector xb(x.GetData(), offsets);
   Vector &xu = xb.GetBlock(0);
   Vector &xp = xb.GetBlock(1);

   BlockVector bb(b.GetData(), offsets);
   Vector &bu = bb.GetBlock(0);
   Vector &bp = bb.GetBlock(1);

   //int N = xu.Size();
   //Vector zero(N);
   //Vector a(N);
   //zero = 1.0;
   //a = 0.0;
   //DenseMatrix A(N);

   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-4);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(100);
   krylov.SetMaxIter(100);
   krylov.SetPreconditioner(*newton_pc.get());
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-3);
   newton.SetAbsTol(1e-9);
   newton.SetMaxIter(1);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetSolver(krylov);
   newton.SetOperator(*ns_residual.get()); // setting this operator gives free() invalid pointer when NewtonSolver destructor is called
   //newton.SetOperator(A);

   bu.SetSubVector(vel_ess_tdof, 0.0);
   bp = 0.0;
   ProjectVelocityDirichletBC(xu);

   newton.Mult(bu, xu); // PROBLEM: free() invalid pointer when NewtonSolver destructor is called
   //newton.Mult(zero, a);
}

void NavierStokesOperator::SolvePressure(Vector &y)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierStokesOperator: Solving pressure step" << std::endl;
   }

   // Solve for pressure: -D M^{-1} G p = D M^{-1} ( f - ( K + C(u) ) u ) - Hdot

   BlockVector yb(y.GetData(), offsets);
   const Vector &xu = yb.GetBlock(0);
   Vector &yp = yb.GetBlock(1);

   // Compute RHS
   z = fu_rhs;                      // z = f
   NL_form->Mult(xu, w);            // w = C(u) u
   K->AddMult(xu, w);               // w += K u
   z -= w;                          // z = f - ( K + C(u) ) u
   invM->Mult(z, w);                // w = M^{-1} ( f - ( K + C(u) ) u )
   D->Mult(w, rhs_p);               // b = D M^{-1} ( f - ( K + C(u) ) u )
   ComputeHdot(xu, Hdot, rk_order); // Compute Hdot = D Udot   TODO: implement it, we should pass a vector of previous u depending on order of extrapolation
   rhs_p += Hdot;                   // b += Hdot

   // Apply bcs
   ProjectPressureDirichletBC(yp);
   DHGc->EliminateRHS(yp, rhs_p);

   // Solve for pressure
   invS->Mult(rhs_p, yp);
}

// Compute Hdot = D u'
void NavierStokesOperator::ComputeHdot(const Vector &xu, Vector &Hdot, int order)
{
   // TODO: implement FD approximation of z = Udot, we should pass a vector of previous u depending on order of extrapolation
   z = 0.0; // This should be \approx Udot
   D->Mult(z, Hdot);
}

// Implementation of the NavierStokesOperatorIMEX class
// IMEX:
// - Implicit Residual: - K u
// - Explicit Residual: f - C(u) u - G p

// Constructor
NavierStokesOperatorIMEX::NavierStokesOperatorIMEX(std::shared_ptr<ParMesh> mesh,
                                                   ParFiniteElementSpace *vel_fes,
                                                   ParFiniteElementSpace *pres_fes,
                                                   double kin_vis,
                                                   std::shared_ptr<BCHandler> bcs,
                                                   bool verbose)
    : NavierStokesOperator(mesh, vel_fes, pres_fes, kin_vis, bcs, verbose)
{
   // Set the splitting type
   splitting_type = SplittingType::IMEX;

   // Create the correct residual operator
   ns_residual.reset(new NavierStokesResidualIMEX(*this));
}

void NavierStokesOperatorIMEX::ImplicitMult(const Vector &x, Vector &y) const
{
   // Compute Implicit Residual: - K u
   const BlockVector xb(x.GetData(), offsets);
   BlockVector yb(y.GetData(), offsets);

   const Vector &xu = xb.GetBlock(0);
   const Vector &xp = xb.GetBlock(1);
   Vector &yu = yb.GetBlock(0);
   Vector &yp = yb.GetBlock(1);

   K->Mult(xu, yu);
   yu.Neg();

   yp = 0.0;
   yu.SetSubVector(vel_ess_tdof, 0.0);
}

void NavierStokesOperatorIMEX::ExplicitMult(const Vector &x, Vector &y) const
{
   // Compute Explicit Residual: f - C(u) u - G p
   const BlockVector xb(x.GetData(), offsets);
   BlockVector yb(y.GetData(), offsets);

   const Vector &xu = xb.GetBlock(0);
   const Vector &xp = xb.GetBlock(1);
   Vector &yu = yb.GetBlock(0);
   Vector &yp = yb.GetBlock(1);

   NL_form->Mult(xu, yu);
   yu.Neg(); // - C(u) u

   G->AddMult(xp, yu, -1.0); // - G p

   if (fu_rhs.Size())
   {
      yu += fu_rhs; // + f
   }

   yp = 0.0;
   yu.SetSubVector(vel_ess_tdof, 0.0);
}

// Implement the NavierStokesOperatorImplicit class
// Implicit:
// - Implicit Residual: - K u - C(u) u
// - Explicit Residual: f - G p
// Constructor
NavierStokesOperatorImplicit::NavierStokesOperatorImplicit(std::shared_ptr<ParMesh> mesh,
                                                           ParFiniteElementSpace *ufes,
                                                           ParFiniteElementSpace *pfes,
                                                           double kin_vis,
                                                           std::shared_ptr<BCHandler> bcs,
                                                           bool verbose) : NavierStokesOperator(mesh, ufes, pfes, kin_vis, bcs, verbose)
{
   // Set the splitting type
   splitting_type = SplittingType::IMPLICIT;

   // Create the correct residual operator
   ns_residual.reset(new NavierStokesResidualImplicit(*this));
}

void NavierStokesOperatorImplicit::ImplicitMult(const Vector &x, Vector &y) const
{
   // Compute Implicit Residual: - K u - C(u) u
   const BlockVector xb(x.GetData(), offsets);
   BlockVector yb(y.GetData(), offsets);

   const Vector &xu = xb.GetBlock(0);
   const Vector &xp = xb.GetBlock(1);
   Vector &yu = yb.GetBlock(0);
   Vector &yp = yb.GetBlock(1);

   K->Mult(xu, yu);
   yu.Neg();

   NL_form->Mult(xu, z);
   yu -= z;

   yp = 0.0;
   yu.SetSubVector(vel_ess_tdof, 0.0);
}

void NavierStokesOperatorImplicit::ExplicitMult(const Vector &x, Vector &y) const
{
   // Compute Explicit Residual: f - G p
   const BlockVector xb(x.GetData(), offsets);
   BlockVector yb(y.GetData(), offsets);

   const Vector &xu = xb.GetBlock(0);
   const Vector &xp = xb.GetBlock(1);
   Vector &yu = yb.GetBlock(0);
   Vector &yp = yb.GetBlock(1);

   G->AddMult(xp, yu); // - G p

   if (fu_rhs.Size())
   {
      yu += fu_rhs; // + f
   }

   yp = 0.0;
   yu.SetSubVector(vel_ess_tdof, 0.0);
}

// Class containing computation of potential quantities of interest
QuantitiesOfInterest::QuantitiesOfInterest(ParMesh *pmesh)
{
   H1_FECollection h1fec(1);
   ParFiniteElementSpace h1fes(pmesh, &h1fec);

   onecoeff.constant = 1.0;
   mass_lf = new ParLinearForm(&h1fes);
   mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
   mass_lf->Assemble();

   ParGridFunction one_gf(&h1fes);
   one_gf.ProjectCoefficient(onecoeff);

   volume = mass_lf->operator()(one_gf);
};

double QuantitiesOfInterest::ComputeKineticEnergy(ParGridFunction &v)
{
   Vector velx, vely, velz;
   double integ = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   FiniteElementSpace *fes = v.FESpace();

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      int intorder = 2 * fe->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

      v.GetValues(i, *ir, velx, 1);
      v.GetValues(i, *ir, vely, 2);
      v.GetValues(i, *ir, velz, 3);

      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);

         double vel2 = velx(j) * velx(j) + vely(j) * vely(j) + velz(j) * velz(j);

         integ += ip.weight * T->Weight() * vel2;
      }
   }

   double global_integral = 0.0;
   MPI_Allreduce(&integ,
                 &global_integral,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   return 0.5 * global_integral / volume;
};

double QuantitiesOfInterest::ComputeCFL(ParGridFunction &u, double dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double)fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0) + uz(i) * invJ(2, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1) + uz(i) * invJ(2, 1)) * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2) + uz(i) * invJ(2, 2)) * detJinv;
         }

         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);
         if (vdim == 3)
         {
            cflz = fabs(dt * uz(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}
