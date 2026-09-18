// Configurable anatomy/benchmark driver using the existing EP solver classes.
#include "mfem.hpp"
#include "../lib/monodomain_solver.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/fiber_conductivity.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <sys/resource.h>
#ifdef MFEM_USE_CUDA
#include <cuda_runtime.h>
#endif
using namespace mfem;
using namespace mfem::electrophysiology;

namespace
{
void Sync()
{
#ifdef MFEM_USE_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   { MFEM_VERIFY(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed"); }
#endif
}
double Wall() { Sync(); return MPI_Wtime(); }
double GlobalMax(double v)
{ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); return g; }
struct Pulse
{
   double start, duration, amplitude;
   double lo[3], hi[3];
   bool Contains(const Vector &x) const
   {
      for (int k=0;k<x.Size();++k) { if (x[k]<lo[k] || x[k]>hi[k]) return false; }
      return true;
   }
};
std::vector<Pulse> ReadPulses(const char *path)
{
   std::ifstream in(path); MFEM_VERIFY(in, "Cannot open pulse file: " << path);
   std::vector<Pulse> pulses; std::string line;
   while (std::getline(in,line))
   {
      if (line.empty() || line[0]=='#') continue;
      Pulse p; std::istringstream row(line);
      MFEM_VERIFY(row >> p.start >> p.duration >> p.amplitude
                  >> p.lo[0] >> p.lo[1] >> p.lo[2] >> p.hi[0] >> p.hi[1] >> p.hi[2],
                  "Pulse rows: start_ms duration_ms amplitude_mA_cm3 xmin ymin zmin xmax ymax zmax");
      MFEM_VERIFY(p.start>=0 && p.duration>0 && std::isfinite(p.amplitude), "Invalid pulse");
      for(int k=0;k<3;k++) MFEM_VERIFY(p.lo[k]<=p.hi[k], "Inverted pulse box");
      pulses.push_back(p);
   }
   MFEM_VERIFY(!pulses.empty(), "No stimulus pulses"); return pulses;
}
struct Probe
{
   std::string name; Vector x; int element=-1; IntegrationPoint ip;
   double activation=-1, repolarization=-1, peak=-1e100, previous=0, rest=0;
   // Time of the sample held in `previous`. With strided diagnostics that is not
   // the previous time step, so linear interpolation must use this, not dt.
   double previous_time=0;
   explicit Probe(int dim) : x(dim) {}
};
std::vector<Probe> ReadProbes(const char *path, ParMesh &mesh)
{
   std::ifstream in(path); MFEM_VERIFY(in,"Cannot open probe file");
   std::vector<Probe> probes; std::string line;
   while(std::getline(in,line))
   {
      if(line.empty() || line[0]=='#') continue;
      Probe p(mesh.SpaceDimension()); double xyz[3]; std::istringstream row(line);
      MFEM_VERIFY(row >> p.name >> xyz[0] >> xyz[1] >> xyz[2],"Probe rows: name x y z [cm]");
      for(int d=0;d<p.x.Size();d++) p.x[d]=xyz[d];
      // Deterministic ownership at partition interfaces: lowest containing rank.
      for(int e=0;e<mesh.GetNE();e++)
      {
         InverseElementTransformation inv(mesh.GetElementTransformation(e));
         inv.SetPrintLevel(-1);
         if(inv.Transform(p.x,p.ip)==InverseElementTransformation::Inside)
         { p.element=e; break; }
      }
      int owner=p.element<0 ? Mpi::WorldSize() : Mpi::WorldRank(), winner;
      MPI_Allreduce(&owner,&winner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
      MFEM_VERIFY(winner<Mpi::WorldSize(), "Probe is outside myocardium: " << p.name);
      if(Mpi::WorldRank()!=winner) p.element=-1;
      probes.push_back(p);
   }
   MFEM_VERIFY(!probes.empty(),"No probes"); return probes;
}
double Sample(const ParGridFunction &u, const Probe &p)
{
   double local=p.element<0 ? 0 : u.GetValue(p.element,p.ip), global;
   MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return global;
}
void SaveNative(ParMesh &mesh, ParGridFunction &u, ReactionSolver &r, const std::string &dir)
{
   r.SyncStateGridFunctions();
   const std::string suffix="."+std::to_string(Mpi::WorldRank());
   std::ofstream ms(dir+"/mesh"+suffix+".mesh"); ms.precision(17); mesh.Print(ms);
   std::ofstream vs(dir+"/voltage"+suffix+".gf"); vs.precision(17); u.Save(vs);
   for(int k=0;k<r.GetModel()->GetNumStates();k++)
   {
      if(k==r.GetModel()->GetPotentialIndex()) continue;
      auto *gf=r.GetStateGridFunction(k);
      if(!gf) continue;
      std::ofstream ss(dir+"/state_"+std::to_string(k)+suffix+".gf"); ss.precision(17); gf->Save(ss);
   }
}
}

int main(int argc, char **argv)
{
   Mpi::Init(argc,argv); Hypre::Init();
   const char *mesh_file="", *geometry="biv", *backend="cpu", *out="results/run";
   const char *pulse_file="", *probe_file="", *fiber_file="", *sheet_file="";
   int order=4, refine=0, model=0, substeps=1, preconditioner=0, save_every=100;
   int nx=10, ny=4, nz=2, diagnostics_every=1;
   int quadrature_order=-1, ode_solver_type=21;
   bool lump_mass=false;
   double dt=0.02, tf=1, sf=0.001334, ss=0.000176, sn=0.000176;
   double chi=1400, cm=0.001, matrix_factor=1, rtol=1e-9, threshold=-30;
   bool pv=true, pa=true;
   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Native mesh in centimetres, preserving its Nodes.");
   args.AddOption(&geometry,"-g","--geometry","biv, niederer (2 x .7 x .3 cm), or sheet (4 x 4 cm).");
   args.AddOption(&backend,"-dev","--device","MFEM device backend.");
   args.AddOption(&out,"-out","--output","Output directory.");
   args.AddOption(&pulse_file,"-stim","--stimuli","Pulse file, all boxes in physical centimetres.");
   args.AddOption(&probe_file,"-probes","--probes","Named physical probes in centimetres.");
   args.AddOption(&fiber_file,"-f","--fiber","Global fiber GF paired with input mesh.");
   args.AddOption(&sheet_file,"-sheet","--sheet","Global sheet GF paired with input mesh.");
   args.AddOption(&order,"-o","--order","Solution polynomial degree, independent of geometry.");
   args.AddOption(&quadrature_order,"-q","--quadrature-order","Common integration order (-1 MFEM defaults); curved CUDA p6 needs an audited rule <=19.");
   args.AddOption(&ode_solver_type,"-ode","--ode-solver","MFEM ODESolver id for the diffusion: 1 forward Euler, 4 RK4, 21 backward Euler (default). <=20 selects an explicit solver, which is only conditionally stable.");
   args.AddOption(&lump_mass,"-lump","--lump-mass","-no-lump","--no-lump-mass","Integrate the mass form with a collocated Gauss-Lobatto rule, making it diagonal. Only useful on the explicit path, and it under-integrates the mass form.");
   args.AddOption(&refine,"-r","--refine","Serial uniform refinements.");
   args.AddOption(&nx,"-nx","--nx","Cartesian elements in x.");
   args.AddOption(&ny,"-ny","--ny","Cartesian elements in y.");
   args.AddOption(&nz,"-nz","--nz","Cartesian elements in z.");
   args.AddOption(&model,"-model","--model","0 MS, 1 FK, 2 TP06 epicardial, 3 TP06 endocardial.");
   args.AddOption(&substeps,"-sub","--substeps","Reaction substeps.");
   args.AddOption(&preconditioner,"-pc","--preconditioner","0 Jacobi, 1 scalar-surrogate LOR+AMG.");
   args.AddOption(&save_every,"-save","--save-every","Output cadence in steps.");
   args.AddOption(&diagnostics_every,"-diag","--diagnostics-every","Sample probes and the voltage excursion every N steps. Each sample copies the whole solution to the host, which is free next to an implicit solve but is ~30% of the step once the solve is explicit with a lumped mass.");
   args.AddOption(&dt,"-dt","--dt","Time step [ms].");
   args.AddOption(&tf,"-tf","--final-time","Final time [ms].");
   args.AddOption(&sf,"-sf","--sigma-f","Conductivity [S/cm], x direction without fields.");
   args.AddOption(&ss,"-ss","--sigma-s","Conductivity [S/cm], y direction without fields.");
   args.AddOption(&sn,"-sn","--sigma-n","Conductivity [S/cm], z direction without fields.");
   args.AddOption(&chi,"-chi","--chi","Surface to volume ratio [1/cm].");
   args.AddOption(&cm,"-cm","--cm","Capacitance [mF/cm^2].");
   args.AddOption(&matrix_factor,"-scale","--matrix-factor","Common positive mass/conductivity scale.");
   args.AddOption(&rtol,"-rtol","--rtol","Implicit CG relative tolerance.");
   args.AddOption(&threshold,"-threshold","--activation-threshold","Activation voltage [mV].");
   args.AddOption(&pv,"-pv","--paraview","-no-pv","--no-paraview","ParaView output.");
   args.AddOption(&pa,"-pa","--partial-assembly","-fa","--full-assembly","Partial assembly.");
   args.Parse(); if(!args.Good()) { if(Mpi::Root()) args.PrintUsage(std::cerr); return 2; }
   MFEM_VERIFY(order>=1 && order<=8 && refine>=0 && refine<=3 && dt>0 && tf>0 &&
               substeps>0 && save_every>0 && nx>0 && ny>0 && nz>0 &&
               chi>0 && cm>0 && sf>0 && ss>0 && sn>0 && matrix_factor>0 &&
               model>=0 && model<=3 && diagnostics_every>=1 &&
               (preconditioner==0 || preconditioner==1),"Invalid configuration");
   Device device(backend);
   if(Mpi::Root()) { device.Print(); args.PrintOptions(std::cout); }
   const double start=Wall();
   auto pulses=ReadPulses(pulse_file);
   std::unique_ptr<Mesh> serial;
   const std::string geom(geometry);
   if(geom=="biv") { serial=std::make_unique<Mesh>(mesh_file,1,1); }
   else if(geom=="niederer")
      serial=std::make_unique<Mesh>(Mesh::MakeCartesian3D(nx,ny,nz,Element::HEXAHEDRON,2.,.7,.3));
   else if(geom=="sheet")
      serial=std::make_unique<Mesh>(Mesh::MakeCartesian2D(nx,ny,Element::QUADRILATERAL,true,4.,4.));
   else { MFEM_ABORT("Unknown geometry"); }
   const int geometry_order=serial->GetNodes() ? serial->GetNodes()->FESpace()->GetMaxElementOrder() : 1;
   std::unique_ptr<GridFunction> global_f, global_s;
   MFEM_VERIFY((std::string(fiber_file).empty())==(std::string(sheet_file).empty()),"Supply both frame fields");
   if(std::string(fiber_file).size())
   {
      std::ifstream f(fiber_file), s(sheet_file); MFEM_VERIFY(f && s,"Cannot read fiber fields");
      global_f=std::make_unique<GridFunction>(serial.get(),f);
      global_s=std::make_unique<GridFunction>(serial.get(),s);
   }
   for(int l=0;l<refine;l++)
   {
      serial->UniformRefinement();
      for(auto *gf : {global_f.get(),global_s.get()})
         if(gf) { gf->FESpace()->Update(); gf->Update(); }
   }
   // Explicit partition is essential for the global-to-parallel field constructor.
   std::unique_ptr<int[]> partition(serial->GeneratePartitioning(Mpi::WorldSize()));
   ParMesh mesh(MPI_COMM_WORLD,*serial,partition.get());
   std::unique_ptr<ParGridFunction> fiber, sheet;
   if(global_f)
   {
      fiber=std::make_unique<ParGridFunction>(&mesh,global_f.get(),partition.get());
      sheet=std::make_unique<ParGridFunction>(&mesh,global_s.get(),partition.get());
   }
   const int dim=mesh.Dimension();
   H1_FECollection fec(order,dim); ParFiniteElementSpace fes(&mesh,&fec);
   const auto ndof=fes.GlobalTrueVSize();
   std::unique_ptr<SymmetricMatrixCoefficient> sigma;
   if(fiber) sigma=std::make_unique<FiberConductivity>(*fiber,*sheet,matrix_factor*sf,matrix_factor*ss,matrix_factor*sn);
   else sigma=std::make_unique<SymmetricMatrixFunctionCoefficient>(dim,[=](const Vector &,DenseSymmetricMatrix &K)
   { K.SetSize(dim); K=0.; K(0,0)=matrix_factor*sf; K(1,1)=matrix_factor*ss; if(dim==3) K(2,2)=matrix_factor*sn; });
   // matrix_factor scales both sides of diffusion only; physical reaction units unchanged.
   ConstantCoefficient mass_chi(matrix_factor*chi), physical_chi(chi), capacitance(cm);
   MonodomainDiffusionSolver diffusion(&fes,new BCHandler(&mesh,false),sigma.get(),&mass_chi,&capacitance,ode_solver_type,false);
   const IntegrationRule *rule=quadrature_order<0 ? nullptr : &IntRules.Get(dim==3 ? Geometry::CUBE : Geometry::SQUARE,quadrature_order);
   // Quadrature audit. Both PA operators launch one Q1D^dim thread block per
   // element, so the rule -- not the polynomial degree -- is what the device
   // refuses. MFEM advertises MAX_Q1D = 14 for CUDA, but that is a storage bound,
   // not a launch bound: measured on an RTX 4070 Laptop (sm_89, fp64),
   // PADiffusionSetup3D fails at Q1D = 10 with "too many resources requested for
   // launch" and the curved-geometry determinant kernel fails at Q1D = 10 with
   // "invalid configuration argument". Q1D = 9 launched in every case tried here.
   //
   // The MFEM defaults are what overflow, and the mass form overflows first:
   // MassIntegrator's rule is 2p + Trans.OrderW(), and a Q3 hexahedron has
   // OrderW = 3*dim - 1 = 8, so curved p=6 asks for order 20, i.e. Q1D = 11.
   // DiffusionIntegrator's rule is 2p + dim - 1, so affine p=8 asks for Q1D = 10.
   // Consequence, stated plainly: on this GPU a curved p >= 6 run cannot integrate
   // the mass form exactly, and --quadrature-order must be chosen and reported.
   int rule_order=quadrature_order;
   if (quadrature_order<0)
   {
      int local=0;
      if (mesh.GetNE()>0)
      {
         const FiniteElement &fe=*fes.GetFE(0);
         ElementTransformation *tr=mesh.GetElementTransformation(0);
         local=std::max(DiffusionIntegrator::GetRule(fe,fe).GetOrder(),
                        MassIntegrator::GetRule(fe,fe,*tr).GetOrder());
      }
      int global; MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      rule_order=global;
   }
   const int q1d=IntRules.Get(Geometry::SEGMENT,rule_order).GetNPoints();
   const int q1d_cap=9;   // measured; see the comment above
   MFEM_VERIFY(!Device::Allows(Backend::CUDA_MASK) || q1d<=q1d_cap,
               "Exact integration here needs order "<<rule_order<<" (Q1D="<<q1d
               <<"), above the measured CUDA launch limit of Q1D="<<q1d_cap<<". Pass "
               "--quadrature-order "<<(2*q1d_cap-1)<<" or lower and treat the run as "
               "under-integrated -- the deficit is "<<(rule_order-(2*q1d_cap-1))
               <<" orders, so compare it against the same case on --device cpu before "
               "using it in a convergence study.");
   if(Mpi::Root()) std::cout<<"Quadrature: order "<<rule_order<<", Q1D "<<q1d
                            <<(quadrature_order<0 ? " (MFEM default)" : " (requested)")<<"\n";
   // Collocated Gauss-Lobatto rule: one point per solution node, so the mass matrix
   // is diagonal. A GLL rule with n points is exact to degree 2n-3, so asking for
   // order 2*order-1 returns exactly order+1 points, matching the H1 node count.
   static IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
   const IntegrationRule *mass_rule = lump_mass
      ? &gll_rules.Get(dim==3 ? Geometry::CUBE : Geometry::SQUARE, 2*order-1) : nullptr;
   if (lump_mass && Mpi::Root())
   {
      const int n=IntRules.Get(Geometry::SEGMENT,1).GetNPoints();  // silence unused warnings
      (void)n;
      std::cout<<"Mass lumping: collocated Gauss-Lobatto, "<<mass_rule->GetNPoints()
               <<" points per element vs "<<(order+1)*(order+1)*(dim==3?(order+1):1)
               <<" nodes\n";
   }
   diffusion.SetIntegrationRule(rule); diffusion.SetMassIntegrationRule(mass_rule);
   diffusion.EnablePA(pa); diffusion.Setup(dt,preconditioner,rtol,false);
   if(Mpi::Root())
   { std::cout<<"Time integration: "<<(diffusion.UsesImplicitTimeIntegration() ? "implicit" : "explicit")
              <<" (ODESolver id "<<ode_solver_type<<")\n"; }
   ReactionSolver reaction(&fes,&physical_chi,&capacitance,static_cast<IonicModelType>(model),TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN,substeps);
   reaction.EnableVoltageLimiting(false); reaction.SetStimulusSampleFraction(.5);
   std::vector<double> parameters; reaction.GetDefaultParameters(parameters);
   reaction.GetModel()->DisableInternalTimeManagement(parameters.data());
   reaction.Setup({},parameters);
   // One cached device mask per pulse. The amplitude is sampled on each interval,
   // which is split at every pulse edge. TP06 expects normalized current [mV/ms].
   std::vector<std::unique_ptr<FunctionCoefficient>> masks;
   std::vector<Vector> mask_values(pulses.size());
   ParGridFunction mask(&fes);
   for(size_t i=0;i<pulses.size();i++)
   {
      masks.emplace_back(new FunctionCoefficient([&,i](const Vector &x) { return pulses[i].Contains(x) ? 1. : 0.; }));
      mask.ProjectCoefficient(*masks.back()); mask.GetTrueDofs(mask_values[i]); mask_values[i].UseDevice(true);
      double count=0; const auto *h=mask_values[i].HostRead(); for(int j=0;j<mask_values[i].Size();j++) count+=h[j];
      MFEM_VERIFY(GlobalMax(count)>0,"Stimulus box contains no solution nodes");
   }
   // Multiple spatial masks are summed into a GF only when the active pulse set changes.
   GridFunctionCoefficient stimulus(&mask);
   reaction.SetStimulation(&stimulus,true);
   Vector stimulus_values(fes.GetTrueVSize()); stimulus_values.UseDevice(true); stimulus_values=0.;
   std::vector<int> previous_active(pulses.size(),-1);
   Vector u; u.UseDevice(true); reaction.GetPotential(u);
   auto &voltage=*diffusion.GetPotentialGf(); voltage.SetFromTrueDofs(u);
   auto probes=ReadProbes(probe_file,mesh);
   std::filesystem::create_directories(out);
   std::ofstream traces, extrema;
   if(Mpi::Root())
   {
      traces.open(std::string(out)+"/traces.csv"); traces.precision(17); traces<<"time_ms";
      for(auto &p:probes) traces<<","<<p.name; traces<<"\n";
      extrema.open(std::string(out)+"/extrema.csv"); extrema.precision(17); extrema<<"time_ms,min_mV,max_mV,cg_iterations\n";
   }
   ParaViewDataCollection pvdc("EP",&mesh); pvdc.SetPrefixPath(out);
   pvdc.SetDataFormat(VTKFormat::BINARY); pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(std::max(order,geometry_order)); pvdc.RegisterField("voltage_mV",&voltage);
   reaction.RegisterFields(pvdc);
   if(fiber) { pvdc.RegisterField("fiber_reference",fiber.get()); pvdc.RegisterField("sheet_reference",sheet.get()); }
   if(pv) { reaction.SyncStateGridFunctions(); pvdc.SetCycle(0); pvdc.SetTime(0); pvdc.Save(); }
   double t=0, diffusion_seconds=0, reaction_seconds=0, diagnostic_seconds=0, io_seconds=0;
   double setup_seconds=Wall()-start; long long iterations=0; int step=0;
   double all_min=1e100, all_max=-1e100;
   for(auto &p:probes) p.rest=p.previous=p.peak=Sample(voltage,p);
   if(Mpi::Root()) { traces<<0; for(auto &p:probes) traces<<","<<p.previous; traces<<"\n"; }
   while(t<tf-1e-12)
   {
      double h=std::min(dt,tf-t);
      for(auto &p:pulses) for(double edge:{p.start,p.start+p.duration})
         if(edge>t+1e-10 && edge<t+h-1e-10) h=edge-t;
      bool changed=false; std::vector<int> active(pulses.size());
      for(size_t i=0;i<pulses.size();i++)
      { active[i]=(t+.5*h>=pulses[i].start && t+.5*h<pulses[i].start+pulses[i].duration); changed|=active[i]!=previous_active[i]; }
      if(changed)
      {
         stimulus_values=0.;
         for(size_t i=0;i<pulses.size();i++) if(active[i])
            stimulus_values.Add(pulses[i].amplitude/(model>=2 ? chi*cm : 1.),mask_values[i]);
         mask.SetFromTrueDofs(stimulus_values); reaction.SetStimulation(&stimulus,true); previous_active=active;
      }
      double tic=Wall(); diffusion.Step(u,t,h,true); diffusion_seconds+=Wall()-tic;
      MFEM_VERIFY(diffusion.GetConverged(),"Diffusion CG did not converge");
      iterations+=diffusion.GetNumIterations();
      tic=Wall(); reaction.Step(u,t,h,true); reaction_seconds+=Wall()-tic;
      t+=h; step++;
      const bool last_step = t>=tf-1e-12;
      const bool diagnose = (step%diagnostics_every==0) || last_step;
      const bool saving = pv && (step%save_every==0 || last_step);
      tic=Wall();
      if(diagnose || saving) { voltage.SetFromTrueDofs(u); }
      if(diagnose)
      {
         // No clamp: report nodal excursions and fail on non-finite values.
         double mn=1e100,mx=-1e100; int bad=0, anybad;
         const auto *host=u.HostRead();
         for(int j=0;j<u.Size();j++) { bad|=!std::isfinite(host[j]); mn=std::min(mn,host[j]); mx=std::max(mx,host[j]); }
         MPI_Allreduce(&bad,&anybad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); MFEM_VERIFY(!anybad,"Non-finite voltage");
         mn=-GlobalMax(-mn); mx=GlobalMax(mx); all_min=std::min(all_min,mn); all_max=std::max(all_max,mx);
         if(Mpi::Root()) { extrema<<t<<","<<mn<<","<<mx<<","<<diffusion.GetNumIterations()<<"\n"; traces<<t; }
         for(auto &p:probes)
         {
            // Interpolate over the interval actually sampled, which is
            // diagnostics_every steps wide, not one.
            const double span=t-p.previous_time;
            double v=Sample(voltage,p); p.peak=std::max(p.peak,v);
            if(p.activation<0 && p.previous<threshold && v>=threshold)
               p.activation=p.previous_time+span*(threshold-p.previous)/(v-p.previous);
            const double v90=p.rest+.1*(p.peak-p.rest);
            if(p.activation>=0 && p.repolarization<0 && p.previous>v90 && v<=v90)
               p.repolarization=p.previous_time+span*(v90-p.previous)/(v-p.previous);
            p.previous=v; p.previous_time=t; if(Mpi::Root()) traces<<","<<v;
         }
         if(Mpi::Root()) traces<<"\n";
      }
      diagnostic_seconds+=Wall()-tic;
      if(saving)
      { tic=Wall(); reaction.SyncStateGridFunctions(); pvdc.SetCycle(step); pvdc.SetTime(t); pvdc.Save(); io_seconds+=Wall()-tic; }
   }
   double tic=Wall(); SaveNative(mesh,voltage,reaction,out); io_seconds+=Wall()-tic;
   // Check all ionic states, including slow gates/calcium (not just voltage).
   int bad_state=0;
   for(int k=0;k<reaction.GetModel()->GetNumStates();k++)
   { if(k==reaction.GetModel()->GetPotentialIndex()) continue; auto *gf=reaction.GetStateGridFunction(k); if(gf) { const auto *h=gf->HostRead(); for(int j=0;j<gf->Size();j++) bad_state|=!std::isfinite(h[j]); } }
   MFEM_VERIFY(GlobalMax(bad_state)==0,"Non-finite ionic state");
   diffusion_seconds=GlobalMax(diffusion_seconds); reaction_seconds=GlobalMax(reaction_seconds);
   diagnostic_seconds=GlobalMax(diagnostic_seconds); io_seconds=GlobalMax(io_seconds);
   setup_seconds=GlobalMax(setup_seconds); double total=GlobalMax(Wall()-start);
   struct rusage usage; getrusage(RUSAGE_SELF,&usage); double rss=GlobalMax(usage.ru_maxrss*1024.);
   if(Mpi::Root())
   {
      std::ofstream metrics(std::string(out)+"/metrics.json"); metrics<<std::setprecision(17);
      metrics<<"{\n\"mfem_version\":\""<<MFEM_VERSION_STRING<<"\",\n\"device\":\""<<backend<<"\",\n"
             <<"\"precision_bytes\":"<<sizeof(real_t)<<",\"mpi_ranks\":"<<Mpi::WorldSize()<<",\n"
             <<"\"geometry_order\":"<<geometry_order<<",\"solution_order\":"<<order<<",\"refinements\":"<<refine<<",\n"
             <<"\"quadrature_order\":"<<quadrature_order<<",\"effective_quadrature_order\":"<<rule_order
             <<",\"quadrature_1d_points\":"<<q1d<<",\n"
             <<"\"preconditioner\":"<<preconditioner<<",\"ode_solver\":"<<ode_solver_type
             <<",\"implicit\":"<<(diffusion.UsesImplicitTimeIntegration()?1:0)
             <<",\"lumped_mass\":"<<(lump_mass?1:0)<<",\n"
             <<"\"true_dofs\":"<<ndof<<",\"dt_ms\":"<<dt<<",\"final_ms\":"<<t<<",\"steps\":"<<step<<",\n"
             <<"\"cg_iterations\":"<<iterations<<",\"peak_rank_rss_bytes\":"<<rss<<",\n"
             <<"\"voltage_min_mV\":"<<all_min<<",\"voltage_max_mV\":"<<all_max<<",\n"
             <<"\"setup_seconds\":"<<setup_seconds<<",\"diffusion_seconds\":"<<diffusion_seconds<<",\n"
             <<"\"reaction_seconds\":"<<reaction_seconds<<",\"diagnostics_transfer_seconds\":"<<diagnostic_seconds<<",\n"
             <<"\"io_seconds\":"<<io_seconds<<",\"total_seconds\":"<<total<<",\n\"probes\":[";
      for(size_t i=0;i<probes.size();i++)
      { auto &p=probes[i]; if(i) metrics<<","; metrics<<"{\"name\":\""<<p.name<<"\",\"activation_ms\":"<<p.activation
         <<",\"apd90_ms\":"<<(p.repolarization<0 ? -1 : p.repolarization-p.activation)<<",\"peak_mV\":"<<p.peak<<"}"; }
      metrics<<"]}\n";
      std::cout<<"Completed "<<step<<" steps, "<<ndof<<" DOFs, voltage ["<<all_min<<","<<all_max<<"] mV\n";
   }
}
