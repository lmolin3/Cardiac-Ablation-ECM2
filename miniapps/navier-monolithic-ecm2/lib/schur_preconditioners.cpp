#include "schur_preconditioners.hpp"

namespace mfem
{

   namespace navier
   {

      ////////////////////////////////////////////////////////////////////////////////////////
      ///                          PRESSURE MASS PRECONDITIONER                            ///
      ////////////////////////////////////////////////////////////////////////////////////////

      // Implementation of PMass constructor
      PMass::PMass(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t kin_vis_) : SchurComplementPreconditioner(pres_fes->GetTrueVSize()), kin_vis(kin_vis_)
      {
         int sdim = pres_fes->GetMesh()->SpaceDimension();

         // Assemble pressure mass matrix Mp
         mp_form = new ParBilinearForm(pres_fes);
         mp_form->AddDomainIntegrator(new MassIntegrator);
         mp_form->Assemble();
         mp_form->FormSystemMatrix(pres_ess_tdofs, Mp);

         // Create solver for Mp
         Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
      }

      // Implementation of PMass::Mult
      void PMass::Mult(const Vector &x, Vector &y) const
      {
         Mp_inv->Mult(x, y);
         y *= kin_vis;
      }

      // Implementation of PMass destructor
      PMass::~PMass()
      {
         delete mp_form;
         delete Mp_inv;
         Mp.Clear();
      }

      ////////////////////////////////////////////////////////////////////////////////////////
      ///                          PRESSURE LAPLACIAN PRECONDITIONER                       ///
      ////////////////////////////////////////////////////////////////////////////////////////

      // Implementation of :PLap constructor
      PLap::PLap(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t sigma_) : SchurComplementPreconditioner(pres_fes->GetTrueVSize()), sigma(sigma_)
      {
         int sdim = pres_fes->GetMesh()->SpaceDimension();

         // Assemble pressure laplacian matrix Lp
         lp_form = new ParBilinearForm(pres_fes);
         lp_form->AddDomainIntegrator(new DiffusionIntegrator);
         lp_form->Assemble();
         lp_form->FormSystemMatrix(pres_ess_tdofs, Lp); // NOTE: Lp singular for fully Dirichlet (velocity) problem: should we wrap with OrthoSolver??

         // Create solver for Lp
         Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim); // NOTE: why is this working even for pres_ess_tdofs empty? i.e. fully Dirichlet problem
         Lp_inv->iterative_mode = false;
      }

      // Implementation of PLap::Mult
      void PLap::Mult(const Vector &x, Vector &y) const
      {
         Lp_inv->Mult(x, y);
         y *= sigma;
      }

      // Implementation of PLap destructor
      PLap::~PLap()
      {
         delete lp_form;
         delete Lp_inv;
         Lp.Clear();
      }

      ////////////////////////////////////////////////////////////////////////////////////////
      ///                               PCD PRECONDITIONER                                 ///
      ////////////////////////////////////////////////////////////////////////////////////////

      // Implementation of PCD constructor
      PCD::PCD(ParFiniteElementSpace *pres_fes_, Array<int> &pres_ess_tdofs_,
               Coefficient *mass_coeff_, Coefficient *diff_coeff_, VectorCoefficient *conv_coeff_) : SchurComplementPreconditioner(pres_fes_->GetTrueVSize()),
                                                                                                   pres_fes(pres_fes_),
                                                                                                   pres_ess_tdofs(pres_ess_tdofs_),
                                                                                                   mass_coeff(mass_coeff_),
                                                                                                   diff_coeff(diff_coeff_),
                                                                                                   conv_coeff(conv_coeff_)
      {
         int sdim = pres_fes->GetMesh()->SpaceDimension();

         // Assemble pressure mass matrix Mp
         mp_form = new ParBilinearForm(pres_fes);
         mp_form->AddDomainIntegrator(new MassIntegrator); // No coeff, comes from commutator
         mp_form->Assemble();
         mp_form->FormSystemMatrix(pres_ess_tdofs, Mp);

         // Assemble pressure laplacian matrix Lp
         lp_form = new ParBilinearForm(pres_fes);
         lp_form->AddDomainIntegrator(new DiffusionIntegrator); // No coeff, comes from commutator
         lp_form->Assemble();
         lp_form->FormSystemMatrix(pres_ess_tdofs, Lp); // NOTE: Lp singular for fully Dirichlet (velocity) problem: should we wrap with OrthoSolver??

         // Assemble pressure convection-diffusion matrix Fp
         fp_form = new ParBilinearForm(pres_fes);
         fp_form->AddDomainIntegrator(new MassIntegrator(*mass_coeff));
         fp_form->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
         if (conv_coeff != nullptr)
            fp_form->AddDomainIntegrator(new ConvectionIntegrator(*conv_coeff));
         fp_form->Assemble();
         fp_form->FormSystemMatrix(pres_ess_tdofs, Fp);

         // TODO: implement boundary conditions according to Elman

         // Create solver for Mp and Lp
         Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
         Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim);

         // Initialize intermediate vectors z and w
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

      // Implementation of PCD::RebuildPreconditioner
      void PCD::Rebuild()
      {
         if (conv_coeff == nullptr) // No need to update
            return;

         delete fp_form;
         fp_form = nullptr;

         fp_form = new ParBilinearForm(pres_fes);
         fp_form->AddDomainIntegrator(new MassIntegrator(*mass_coeff));
         fp_form->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
         fp_form->AddDomainIntegrator(new ConvectionIntegrator(*conv_coeff));
         fp_form->Assemble();
         fp_form->FormSystemMatrix(pres_ess_tdofs, Fp);
      }

      // Implementation of PCD destructor
      PCD::~PCD()
      {
         delete Mp_inv;
         delete Lp_inv;

         delete fp_form;
         Fp.Clear();

         delete lp_form;
         Lp.Clear();

         delete mp_form;
         Mp.Clear();
      }

      ////////////////////////////////////////////////////////////////////////////////////////
      ///                          CAHOUET-CHABARD PRECONDITIONER                          ///
      ////////////////////////////////////////////////////////////////////////////////////////

      // Implementation of CahouetChabard constructor
      CahouetChabard::CahouetChabard(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs_, real_t dt_, real_t kin_vis_) : SchurComplementPreconditioner(pres_fes->GetTrueVSize()),
                                                                                                                               dt(dt_),
                                                                                                                               kin_vis(kin_vis_),
                                                                                                                               pres_ess_tdofs(pres_ess_tdofs_)
      {
         int sdim = pres_fes->GetMesh()->SpaceDimension();

         // Assemble pressure mass matrix Mp
         mp_form = new ParBilinearForm(pres_fes);
         mp_form->AddDomainIntegrator(new MassIntegrator);
         mp_form->Assemble();
         mp_form->FormSystemMatrix(pres_ess_tdofs, Mp);

         // Assemble pressure laplacian matrix Lp
         lp_form = new ParBilinearForm(pres_fes);
         lp_form->AddDomainIntegrator(new DiffusionIntegrator);
         lp_form->Assemble();
         lp_form->FormSystemMatrix(pres_ess_tdofs, Lp); // NOTE: Lp singular for fully Dirichlet (velocity) problem: should we wrap with OrthoSolver??

         // Create solver for Mp and Lp
         Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetSystemsOptions(sdim);
         Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetSystemsOptions(sdim);

         // Initialize intermediate vector z
         z.SetSize(Mp_inv->Height());
      }

      // Implementation of CahouetChabard::Mult
      void CahouetChabard::Mult(const Vector &x, Vector &y) const
      {
         Lp_inv->Mult(x, y);
         y /= dt;
         Mp_inv->Mult(x, z);
         z *= kin_vis;
         y += z;

         for (int i = 0; i < pres_ess_tdofs.Size(); i++)
         {
            y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
         }
      }

      // Implementation of CahouetChabard destructor
      CahouetChabard::~CahouetChabard()
      {
         delete mp_form;
         Mp.Clear();

         delete lp_form;
         Lp.Clear();

         delete Lp_inv;
         delete Mp_inv;
      }

      ////////////////////////////////////////////////////////////////////////////////////////
      ///                     LEAST SQUARES COMMUTATOR PRECONDITIONER                      ///
      ////////////////////////////////////////////////////////////////////////////////////////

      // Implementation of LSC constructor
      LSC::LSC(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs_, HypreParMatrix *D_, HypreParMatrix *G_, HypreParMatrix *Mv) : SchurComplementPreconditioner(pres_fes->GetTrueVSize()),
                                                                                                                                                         D(D_), G(G_),
                                                                                                                                                         pres_ess_tdofs(pres_ess_tdofs_)
      {
         int sdim = pres_fes->GetMesh()->SpaceDimension();

         // Assemble operator Q = D diag(T)^-1 G
         diagT = new Vector(Mv->Height());
         Mv->GetDiag(*diagT);

         HypreParMatrix* MinvG = new HypreParMatrix(*G); // S = G
         MinvG->InvScaleRows(*diagT);         // S = diag(T)^-1 G
         S.Reset(ParMult(D, MinvG));          // S = D diag(T)^-1 G
         delete MinvG;

         invS = new HypreBoomerAMG(*S.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(invS)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(invS)->SetSystemsOptions(sdim);

         // Initialize intermediate vectors z1, z2, q
         z1.SetSize(Mv->Height());
         z2.SetSize(Mv->Height());
         q.SetSize(S->Height());
      }

      // Implementation of LSC::Mult    (D T^{-1} G)^{-1}  ( D T^{-1} C T^{-1} G )   (D T^{-1} G)^{-1}, T = diag(Mv)
      void LSC::Mult(const Vector &x, Vector &y) const
      {
         MFEM_ASSERT(opC != nullptr, "Operator C not set for LSC preconditioner");

         z1 = 0.0; z2 = 0.0; q = 0.0;

         invS->Mult(x, q);

         G->Mult(q, z2);
         z2 /= *diagT;
         opC->Mult(z2, z1);
         z1 /= *diagT;
         D->Mult(z1, q);

         invS->Mult(q, y);

         for (int i = 0; i < pres_ess_tdofs.Size(); i++)
         {
            y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
         }
      }

      // Implementation of LSC Destructor
      LSC::~LSC()
      {
         S.Clear();
         delete invS;
         delete diagT;
      }

      ////////////////////////////////////////////////////////////////////////////////////////
      ///                   APPROXIMATE DISCRETE LAPLACIAN PRECONDITIONER                  ///
      ////////////////////////////////////////////////////////////////////////////////////////

      // Implementation of ApproximateDiscreteLaplacian constructor
      ApproximateDiscreteLaplacian::ApproximateDiscreteLaplacian(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs_, const HypreParMatrix *D, const HypreParMatrix *G, const HypreParMatrix *Mv, real_t sigma_) : SchurComplementPreconditioner(pres_fes->GetTrueVSize()), sigma(sigma_), pres_ess_tdofs(pres_ess_tdofs_)
      {
         int sdim = pres_fes->GetMesh()->SpaceDimension();

         // Assemle operator S = D diag(M)^-1 G
         Vector *diag = new Vector(Mv->Height());
         Mv->GetDiag(*diag);

         auto MinvG = new HypreParMatrix(*G); // S = G
         MinvG->InvScaleRows(*diag);          // S = diag(M)^-1 G
         S.Reset(ParMult(D, MinvG));          // S = D diag(M)^-1 G
         delete MinvG;
         delete diag;

         invS = new HypreBoomerAMG(*S.As<HypreParMatrix>());
         dynamic_cast<HypreBoomerAMG *>(invS)->SetPrintLevel(0);
         dynamic_cast<HypreBoomerAMG *>(invS)->SetSystemsOptions(sdim);
      }

      // Implementation of ApproximateDiscreteLaplacian Mult
      void ApproximateDiscreteLaplacian::Mult(const Vector &x, Vector &y) const
      {
         invS->Mult(x, y);
         y *= sigma;

         for (int i = 0; i < pres_ess_tdofs.Size(); i++)
         {
            y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
         }
      }

      // Implementation of ApproximateDiscreteLaplacian destructor
      ApproximateDiscreteLaplacian::~ApproximateDiscreteLaplacian()
      {
         S.Clear();
         delete invS;
      }

   } // namespace navier

} // namespace mfem