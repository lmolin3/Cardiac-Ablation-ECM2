#include "schur_preconditioners.hpp"

namespace mfem {

namespace navier
{

/// PCD PRECONDITIONER
// Implementation of PCD constructor
PCD::PCD(Solver *Mp_inv, Solver *Lp_inv, Operator *Fp) : SchurComplementPC(Fp->Height()),
                                                               Mp_inv(Mp_inv),
                                                               Lp_inv(Lp_inv),
                                                               Fp(Fp)
{
   z.SetSize(Fp->Height());
   w.SetSize(Fp->Height());
}

// Implementation of PCD::Mult
void PCD::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());
   w.SetSize(y.Size());

   Lp_inv->Mult(x, z);
   Fp->Mult(z, w);
   Mp_inv->Mult(w, y);
}

// Implementation of PCDBuilder constructor
PCDBuilder::PCDBuilder(ParFiniteElementSpace *pres_fes_, Array<int> &pres_ess_tdofs_,
                       Coefficient *mass_coeff, Coefficient *diff_coeff, VectorCoefficient *conv_coeff_) : PCBuilder(),
                                                                           pres_ess_tdofs(pres_ess_tdofs_),
                                                                           pres_fes(pres_fes_),
                                                                           mp_form( pres_fes),
                                                                           lp_form( pres_fes),
                                                                           mass_coeff(mass_coeff),
                                                                           diff_coeff(diff_coeff),
                                                                           conv_coeff(conv_coeff_)
{
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.FormSystemMatrix(pres_ess_tdofs, Lp); 

   fp_form = new ParBilinearForm(pres_fes);
   fp_form->AddDomainIntegrator(new MassIntegrator(*mass_coeff));
   fp_form->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
   if ( conv_coeff != nullptr )
      fp_form->AddDomainIntegrator(new ConvectionIntegrator(*conv_coeff));
   fp_form->Assemble();
   fp_form->FormSystemMatrix(pres_ess_tdofs, Fp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim);

   pcd = new PCD(Mp_inv, Lp_inv, Fp.Ptr());
}

// Implementation of PCDBuilder constructor with default coefficients (no convection)
PCDBuilder::PCDBuilder(ParFiniteElementSpace *pres_fes_, Array<int> &pres_ess_tdofs) : PCBuilder(),
                                                                                     pres_fes(pres_fes_),
                                                                                     mp_form(pres_fes),
                                                                                     lp_form(pres_fes)
{
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.FormSystemMatrix(pres_ess_tdofs, Lp); 

   fp_form = new ParBilinearForm(pres_fes);
   fp_form->AddDomainIntegrator(new MassIntegrator);
   fp_form->AddDomainIntegrator(new DiffusionIntegrator);
   fp_form->Assemble();
   fp_form->FormSystemMatrix(pres_ess_tdofs, Fp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);

   pcd = new PCD(Mp_inv, Lp_inv, Fp.Ptr());
}

// Implementation of PCDBuilder::GetSolver
SchurComplementPC *PCDBuilder::GetSolver() { return pcd; }

// Implementation of PCDBuilder::RebuildPreconditioner
SchurComplementPC *PCDBuilder::RebuildPreconditioner()
{
   if ( conv_coeff == nullptr ) // No need to update 
      return pcd;
   
   delete fp_form;
   fp_form = nullptr;

   fp_form = new ParBilinearForm(pres_fes);
   fp_form->AddDomainIntegrator(new MassIntegrator(*mass_coeff));
   fp_form->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
   fp_form->AddDomainIntegrator(new ConvectionIntegrator(*conv_coeff));
   fp_form->Assemble();
   fp_form->FormSystemMatrix(pres_ess_tdofs, Fp);

   pcd->SetFp(Fp.Ptr()); 

   return pcd;
}

// Implementation of PCDBuilder destructor
PCDBuilder::~PCDBuilder()
{
   delete pcd;
   delete Lp_inv;
   delete Mp_inv;
   delete fp_form;
}

/// CAHOUET CHABARD PRECONDITIONER
// Implementation of CahouetChabardPC constructor
CahouetChabardPC::CahouetChabardPC(Solver &Mp_inv, Solver &Lp_inv,
                                   Array<int> &pres_ess_tdofs, real_t dt, real_t kin_vis_) : SchurComplementPC(Mp_inv.Height()),
                                                                           Mp_inv(Mp_inv),
                                                                           Lp_inv(Lp_inv),
                                                                           z(Mp_inv.Height()),
                                                                           dt(dt),
                                                                           kin_vis(kin_vis_),
                                                                           pres_ess_tdofs(pres_ess_tdofs) {}

// Implementation of CahouetChabardPC::Mult
void CahouetChabardPC::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());

   Lp_inv.Mult(x, y); 
   y /= dt;
   Mp_inv.Mult(x, z);
   z *= kin_vis;
   y += z;

   for (int i = 0; i < pres_ess_tdofs.Size(); i++)
   {
      y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
   }
}

// Implementation of CahouetChabardBuilder constructor
CahouetChabardBuilder::CahouetChabardBuilder(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t dt, real_t kin_vis) : mp_form( pres_fes),
                                                                                                                      lp_form( pres_fes)
{
   Array<int> empty;
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(empty, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(empty, Lp); 

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim); // NOTE: without this option, incorrect results when pres_ess_tdof is empty and inverting laplace operator

   cahouet_chabard = new CahouetChabardPC(*Mp_inv, *Lp_inv, pres_ess_tdofs, dt, kin_vis);
}

// Implementation of CahouetChabardBuilder::GetSolver
SchurComplementPC *CahouetChabardBuilder::GetSolver() { return cahouet_chabard; }

// Implementation of CahouetChabardBuilder destructor
CahouetChabardBuilder::~CahouetChabardBuilder()
{
   delete cahouet_chabard;
   delete Lp_inv;
   delete Mp_inv;
}

/// APPROXIMATE INVERSE PRECONDITIONER
// Implementation of SchurApproxInvPC constructor
SchurApproxInvPC::SchurApproxInvPC(Solver &Mp_inv, Solver &Lp_inv,
                                   Array<int> &pres_ess_tdofs, real_t dt) : SchurComplementPC(Mp_inv.Height()),
                                                                           Mp_inv(Mp_inv),
                                                                           Lp_inv(Lp_inv),
                                                                           z(Mp_inv.Height()),
                                                                           dt(dt),
                                                                           pres_ess_tdofs(pres_ess_tdofs) {}

// Implementation of SchurApproxInvPC::Mult
void SchurApproxInvPC::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());

   Mp_inv.Mult(x, z);
   z *= dt;
   Lp_inv.Mult(x, y);
   y += z;

   for (int i = 0; i < pres_ess_tdofs.Size(); i++)
   {
      y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
   }
}

// Implementation of SchurApproxInvBuilder constructor
SchurApproxInvBuilder::SchurApproxInvBuilder(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t dt) : mp_form( pres_fes),
                                                                                                                      lp_form( pres_fes)
{
   Array<int> empty;
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(empty, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(empty, Lp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim);

   schur_approx_inv = new SchurApproxInvPC(*Mp_inv, *Lp_inv, pres_ess_tdofs, dt);
}

// Implementation of SchurApproxInvBuilder::GetSolver
SchurComplementPC *SchurApproxInvBuilder::GetSolver() { return schur_approx_inv; }

// Implementation of SchurApproxInvBuilder destructor
SchurApproxInvBuilder::~SchurApproxInvBuilder()
{
   delete schur_approx_inv;
   delete Lp_inv;
   delete Mp_inv;
}

/// PRESSURE MASS PRECONDITIONER
// Implementation of PMassPC constructor
PMassPC::PMassPC(Solver &Mp_inv, real_t dt) : SchurComplementPC(Mp_inv.Width()), Mp_inv(Mp_inv), dt(dt) {}

// Implementation of PMassPC::Mult
void PMassPC::Mult(const Vector &x, Vector &y) const
{
   Mp_inv.Mult(x, y);
   y /= dt;
}

// Implementation of PMassBuilder constructor
PMassBuilder::PMassBuilder(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t dt) : mp_form( pres_fes)
{
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);
//#ifndef MFEM_USE_SUITESPARSE
   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
//#else
//   Mp.As<HypreParMatrix>()->GetDiag(M_local);
//   Mp_inv = new UMFPackSolver(M_local);
//   dynamic_cast<UMFPackSolver *>(Mp_inv)->SetPrintLevel(0);
//#endif
   pmass = new PMassPC(*Mp_inv, dt);
}

// Implementation of PMassBuilder::GetSolver
SchurComplementPC *PMassBuilder::GetSolver() { return pmass; }

// Implementation of PMassBuilder destructor
PMassBuilder::~PMassBuilder()
{
   delete pmass;
   delete Mp_inv;
}

/// PRESSURE LAPLACIAN PRECONDITIONER
// Implementation of PLapPC constructor
PLapPC::PLapPC(Solver &Lp_inv) : SchurComplementPC(Lp_inv.Width()), Lp_inv(Lp_inv) {}

// Implementation of PLapPC::Mult
void PLapPC::Mult(const Vector &x, Vector &y) const
{
   Lp_inv.Mult(x, y);
}

// Implementation of PLapBuilder constructor
PLapBuilder::PLapBuilder(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs) : lp_form(pres_fes)
{
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(pres_ess_tdofs, Lp);
//#ifndef MFEM_USE_SUITESPARSE
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim);
//#else
//   Lp.As<HypreParMatrix>()->GetDiag(L_local);
//   Lp_inv = new UMFPackSolver(L_local);
//   dynamic_cast<UMFPackSolver *>(Lp_inv)->SetPrintLevel(0);
//#endif
   plap = new PLapPC(*Lp_inv);
}

// Implementation of PLapBuilder::GetSolver
SchurComplementPC *PLapBuilder::GetSolver() { return plap; }

// Implementation of PLapBuilder destructor
PLapBuilder::~PLapBuilder()
{
   delete plap;
   delete Lp_inv;
}


// Implementation of ApproximateDiscreteLaplacianBuilder constructor
ApproximateDiscreteLaplacianBuilder::ApproximateDiscreteLaplacianBuilder(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs_, const HypreParMatrix *D, const  HypreParMatrix *G, const  HypreParMatrix *M, real_t sigma) : 
D(D), G(G), M(M)
{
   int sdim = pres_fes->GetMesh()->SpaceDimension();

   // Assemle operator S = D diag(M)^-1 G
   Vector *diag = new Vector(M->Height());
   M->GetDiag(*diag);

   auto MinvG = new HypreParMatrix(*G); // S = G
   MinvG->InvScaleRows(*diag);          // S = diag(M)^-1 G
   S = ParMult(D, MinvG);               // S = D diag(M)^-1 G
   delete MinvG;
   delete diag;

   invS = new HypreBoomerAMG(*S);
   dynamic_cast<HypreBoomerAMG *>(invS)->SetPrintLevel(0);
   dynamic_cast<HypreBoomerAMG *>(invS)->SetSystemsOptions(sdim);

   // Approximate Discrete Laplacian preconditioner
   adpl = new ApproximateDiscreteLaplacianPC(*invS, pres_ess_tdofs_, sigma);
}

// Implementation of ApproximateDiscreteLaplacianBuilder GetSolver
SchurComplementPC *ApproximateDiscreteLaplacianBuilder::GetSolver() { return adpl; }

// Implementation of ApproximateDiscreteLaplacianBuilder destructor
ApproximateDiscreteLaplacianBuilder::~ApproximateDiscreteLaplacianBuilder()
{
   delete adpl;
   delete invS;
   delete S;
}


// Implementation of ApproximateDiscreteLaplacian constructor
ApproximateDiscreteLaplacianPC::ApproximateDiscreteLaplacianPC(Solver &invS, Array<int> &pres_ess_tdofs_, real_t sigma) : SchurComplementPC(invS.Width()), invS(invS), sigma(sigma), pres_ess_tdofs(pres_ess_tdofs_) {}

// Implementation of ApproximateDiscreteLaplacian Mult
void ApproximateDiscreteLaplacianPC::Mult(const Vector &x, Vector &y) const
{
   invS.Mult(x, y);
   y *= sigma;

   for (int i = 0; i < pres_ess_tdofs.Size(); i++)
   {
      y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
   }
}



} // namespace navier

} // namespace mfem