// Filename: main.cpp

#include <ibamr/config.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFECentroidPostProcessor.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/IBFESurfaceMethod.h>
#include <ibamr/IBStrategySet.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/HierarchyAveragedDataManager.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_CHKERRQ.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/IndexUtilities.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserRobinBcCoefs.h>
#include <ibtk/snapshot_utilities.h>

#include <ADS/CutCellVolumeMeshMapping.h>
#include <ADS/LSCutCellLaplaceOperator.h>
#include <ADS/LSFromMesh.h>
#include <ADS/SBAdvDiffIntegrator.h>
#include <ADS/SBBoundaryConditions.h>
#include <ADS/SBIntegrator.h>
#include <ADS/VolumeBoundaryMeshMapping.h>
#include <ADS/app_namespaces.h>

#include <libmesh/analytic_function.h>
#include <libmesh/boundary_info.h>
#include <libmesh/boundary_mesh.h>
#include <libmesh/dense_matrix.h>
#include <libmesh/dense_vector.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/dof_map.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/fe.h>
#include <libmesh/fe_interface.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_function.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/mesh_tools.h>
#include <libmesh/parallel.h>
#include <libmesh/quadrature.h>
#include <libmesh/sparse_matrix.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

// Local includes
#include "CBFinder.h"
#include "CirculationModel.h"
#include "FeedbackForcer.h"
#include "QFcn.h"
#include "RBFOneSidedReconstructions.h"
#include "RBFReconstructCacheOS.h"
#include "VelocityBcCoefs.h"

using namespace LS;

static double dy = std::numeric_limits<double>::quiet_NaN();
void
bdry_fcn(const IBTK::VectorNd& x, double& ls_val)
{
    if (x[1] < -3.84 + dy)
        ls_val = std::max((-3.476 - x[0]), (x(0) - 0.392));
    else if (x[1] > (3.84 - dy))
        ls_val = std::max((-1.4166 - x[0]), (x[0] - 1.4174));
}

static double k_on = 1.0;
static double k_off = 1.0;
static double sf_max = 1.0;
static double fl_scale = 1.0;
static double sf_scale = 1.0;
static double time_to_start = 0.0;
static double D_coef = 0.0;
static double sf_init_val = 0.0;

double
sf_ode(const double sf_val,
       const std::vector<double>& fl_vals,
       const std::vector<double>& sf_vals,
       const double time,
       void* /*ctx*/)
{
    if (time < time_to_start) return 0.0;
    // Convert fl platelets
    double fl_pl = fl_vals[0] * fl_scale;
    // Convert sf platelets
    double sf_pl = sf_val * sf_scale;
    // Flux
    double flux = k_on * (sf_max - sf_pl) * fl_pl - k_off * sf_pl;
    // Return flux given in thousands of platelets
    return flux / sf_scale;
}

double
sf_init(const VectorNd& /*X*/, const Node* const /*node*/)
{
    return sf_init_val;
}

double
a_fcn(const double q_val,
      const std::vector<double>& fl_vals,
      const std::vector<double>& sf_vals,
      const double time,
      void* /*ctx*/)
{
    if (time < time_to_start) return 0.0;
    // Convert fl concentration to amount per cm^3
    // Note currently in ten million per cm^3
    double fl_pl = q_val * fl_scale;
    // Convert sf concentration to amount per cm^2
    // Note currently in thousands per cm^2
    double sf_pl = sf_vals[0] * sf_scale;
    // Flux
    double flux = k_on * (sf_max - sf_pl) * fl_pl;
    // Return flux given in 10 millions of platelets
    return flux / fl_scale;
}

double
g_fcn(const double q_val,
      const std::vector<double>& fl_vals,
      const std::vector<double>& sf_vals,
      const double time,
      void* /*ctx*/)
{
    if (time < time_to_start) return 0.0;
    // Convert sf platelets
    double sf_pl = sf_vals[0] * sf_scale;
    // Flux
    double flux = k_off * sf_pl;
    // Return flux given in 10 millions of platelets
    return flux / fl_scale;
}

namespace
{
static const unsigned int NUM_PARTS = 2;
static const unsigned int LEAFLET_PART = 0;
static const unsigned int HOUSING_PART = 1;

double J_min_leaflets = std::numeric_limits<double>::max();
double J_max_leaflets = std::numeric_limits<double>::min();
double I1_min_leaflets = std::numeric_limits<double>::max();
double I1_max_leaflets = std::numeric_limits<double>::min();
bool use_feedback = true;
bool stiff_right = false;
// double I4_min_leaflets = std::numeric_limits<double>::max();
// double I4_max_leaflets = std::numeric_limits<double>::min();

inline TensorValue<double>
DEV(const TensorValue<double>& FF, const TensorValue<double>& PP)
{
    // P_dev = P - (tr(PP FF^T) / 3) FF^-T
    return PP - ((1.0 / 3.0) * (PP * FF.transpose()).tr()) * tensor_inverse_transpose(FF);
}

inline TensorValue<double>
dI1_dFF(const TensorValue<double>& FF)
{
    // I1 = I1(CC) = tr(FF^T FF)
    return 2.0 * FF;
}

inline TensorValue<double>
dI1_bar_dFF(const TensorValue<double>& FF)
{
    // I1_bar = I1(CC_bar) = tr(FF_bar^T FF_bar)
    // FF_bar = J^(-1/3) FF ===> det(FF_bar) = 1, I1_bar = J^(-2/3) I1
    const double J = FF.det();
    const double J_n13 = 1.0 / std::cbrt(J);
    double I1 = (FF.transpose() * FF).tr();
    return 2.0 * J_n13 * J_n13 * (FF - (1.0 / 3.0) * I1 * tensor_inverse_transpose(FF));
}

inline TensorValue<double>
dI4f_dFF(const TensorValue<double>& FF, const VectorValue<double>& f0)
{
    // I4f = f0 * CC * f0 = f0 * FF^T FF * f0 = (FF f0) * (FF f0)
    const VectorValue<double> f = FF * f0;
    return 2.0 * outer_product(f, f0);
}

inline TensorValue<double>
dI4f_bar_dFF(const TensorValue<double>& FF, const VectorValue<double>& f0)
{
    // I4f_bar = f0 * CC_bar * f0 = f0 * FF_bar^T FF_bar * f0 = (FF_bar f0) * (FF_bar f0)
    // FF_bar = J^(-1/3) FF ===> det(FF_bar) = 1, I4f_bar = J^(-2/3) I4f
    const double J = FF.det();
    const VectorValue<double> f = FF * f0;
    const double I4f = f * f;
    const double J_n13 = 1.0 / std::cbrt(J);
    return 2.0 * J_n13 * J_n13 * (outer_product(f, f0) - (1.0 / 3.0) * I4f * tensor_inverse_transpose(FF));
}

inline TensorValue<double>
dJ_dFF(const TensorValue<double>& FF)
{
    const double J = FF.det();
    return J * tensor_inverse_transpose(FF);
}

struct PenaltyStressParams
{
    double c1_s;
    double c1_p;
};

void
penalty_stress_fcn(TensorValue<double>& PP,
                   const TensorValue<double>& FF,
                   const libMesh::Point& /*X*/,
                   const libMesh::Point& /*s*/,
                   Elem* const elem,
                   const vector<const vector<double>*>& /*var_data*/,
                   const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                   double /*time*/,
                   void* ctx)
{
    auto params = static_cast<PenaltyStressParams*>(ctx);
    const double c1_s = params->c1_s;
    const double c1_p = params->c1_p;
    const TensorValue<double> FF_inv_trans = tensor_inverse_transpose(FF, NDIM);
    if (elem->subdomain_id() == 1)
    {
        PP = 2.0 * c1_p * (FF - FF_inv_trans);
    }
    else
    {
        PP = 2.0 * c1_s * (FF - FF_inv_trans);
    }
    return;
}

struct PenaltyForceParams
{
    double kappa_s;
    double kappa_p;
};

void
penalty_body_force_fcn(VectorValue<double>& F,
                       const TensorValue<double>& /*FF*/,
                       const libMesh::Point& x,
                       const libMesh::Point& X,
                       Elem* const elem,
                       const vector<const vector<double>*>& /*var_data*/,
                       const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                       double /*time*/,
                       void* ctx)
{
    auto params = static_cast<PenaltyForceParams*>(ctx);
    const double kappa_s = params->kappa_s;
    const double kappa_p = params->kappa_p;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        if (elem->subdomain_id() == 1)
        {
            F(d) = kappa_p * (X(d) - x(d));
        }
        else
        {
            F(d) = kappa_s * (X(d) - x(d));
        }
    }
    return;
}

struct LeafletStressParams
{
    double C10;    // dyne/cm^2
    double C01;    // dimensionless
    double k1;     // dyne/cm^2
    double k2;     // dimensionless
    double a_disp; // dimensionless
    double beta_s;
    double C10_min; // dyne/cm^2
    double C10_max; // dyne/cm^2
};

double
findStiffness(const double C10_min, const double C10_max, const double CB, const double CB_max)
{
    return 0.5 * (C10_min + C10_max) - 0.5 * (C10_max - C10_min) * std::cos(M_PI * CB / CB_max);
}

void
leaflet_stress_fcn(TensorValue<double>& PP,
                   const TensorValue<double>& FF,
                   const libMesh::Point& X, // current location
                   const libMesh::Point& s, // reference location
                   Elem* const elem,
                   const vector<const vector<double>*>& var_data,
                   const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                   double time,
                   void* ctx)
{
    auto params = static_cast<std::pair<LeafletStressParams*, std::shared_ptr<CBFinder>*>*>(ctx)->first;
    auto cb_finder = *(static_cast<std::pair<LeafletStressParams*, std::shared_ptr<CBFinder>*>*>(ctx)->second);
    const vector<double>& v1_vec = *var_data[0];
    const vector<double>& v2_vec = *var_data[1];
#if (NDIM == 2)
    const VectorValue<double> v1(v1_vec[0], v1_vec[1], 0.0);
    const VectorValue<double> v2(v2_vec[0], v2_vec[1], 0.0);
#else
    const VectorValue<double> v1(v1_vec[0], v1_vec[1], v1_vec[2]);
    const VectorValue<double> v2(v2_vec[0], v2_vec[1], v2_vec[2]);
#endif

    double C10_min = params->C10_min;
    double C10_max = params->C10_max;
    const double C01 = params->C01;

    const double J = FF.det();
    const double J_n13 = 1.0 / std::cbrt(J);
    const double I1 = (FF.transpose() * FF).tr();
    const double I1_bar = J_n13 * J_n13 * I1;

    // BHV model following Murdock et al., J Mech Behav Biomed Mat, 2018

    double C10 = 0.0;
    if (use_feedback)
    {
        double CB = cb_finder->findCB(FF, X, s, elem, time);
        C10 = findStiffness(C10_min, C10_max, CB, sf_max);
    }
    else if (stiff_right)
    {
        double CB = cb_finder->findCB(FF, X, s, elem, time);
        C10 = CB > 0.0 ? C10_max : C10_min;
    }
    else
    {
        C10 = C10_min;
    }

    // Isotropic contribution.
    PP = C10 * exp(C01 * (I1_bar - 3.0)) * C01 * dI1_bar_dFF(FF);
#if 0
    // Fiber contributions.
    const VectorValue<double> f0 = v1;

    // f = FF*f0 is the stretched and rotated fiber direction in the current
    // configuration.
    const VectorValue<double> f = FF * f0;

    // f_bar = FF_bar*f0 is the stretched and rotated fiber direction in the
    // current configuration, but using the modified deformation gradient
    // tensor.
    const double I4f = f * f;
    const double I_disp = a_disp*I1_bar + (1.0-3.0*a_disp)*I4f;
    const TensorValue<double> dI_disp_dFF = a_disp*dI1_bar_dFF(FF) + (1.0-3.0*a_disp)*dI4f_dFF(FF, f0);

    // Only include fiber stresses when the fibers are under extension:
    if (I4f > 1.0)
    {
        PP += k1 * exp(k2 * pow(I_disp-1.0, 2.0)) * (I_disp - 1.0) * dI_disp_dFF;
    }
#endif

    J_min_leaflets = std::min(J_min_leaflets, J);
    J_max_leaflets = std::max(J_max_leaflets, J);
    I1_min_leaflets = std::min(I1_min_leaflets, I1);
    I1_max_leaflets = std::max(I1_max_leaflets, I1);
    // I4_min_leaflets = std::min(I4_min_leaflets, I4f);
    // I4_max_leaflets = std::max(I4_max_leaflets, I4f);
    return;
}

void
leaflet_penalty_stress_fcn(TensorValue<double>& PP,
                           const TensorValue<double>& FF,
                           const libMesh::Point& /*X*/,
                           const libMesh::Point& /*s*/,
                           Elem* const /*elem*/,
                           const vector<const vector<double>*>& /*var_data*/,
                           const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                           double /*time*/,
                           void* ctx)
{
    LeafletStressParams* params = static_cast<LeafletStressParams*>(ctx);
    const double beta_s = params->beta_s;
    const TensorValue<double> FF_inv_trans = tensor_inverse_transpose(FF, NDIM);
    // PP = (beta_s == 0.0 ? 0.0 : beta_s * log(pow(FF.det(), 2.0))) * FF_inv_trans;
    // PP = (beta_s == 0.0 ? 0.0 : beta_s * 0.5 * FF.det()) * FF_inv_trans; //Nandini model
    double J = FF.det();
    PP = beta_s * J * log(J) * FF_inv_trans;
    return;
}

struct LeafletPenaltyForceParams
{
    BoundaryInfo* boundary_info;
    double kappa_s;
    double kappa_p;
};

void
leaflet_penalty_surface_force_fcn(VectorValue<double>& F,
                                  const VectorValue<double>& /*n*/,
                                  const VectorValue<double>& /*N*/,
                                  const TensorValue<double>& /*FF*/,
                                  const libMesh::Point& x,
                                  const libMesh::Point& X,
                                  Elem* const elem,
                                  const unsigned short side,
                                  const vector<const vector<double>*>& /*var_data*/,
                                  const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                                  double /*time*/,
                                  void* ctx)
{
    LeafletPenaltyForceParams* params = static_cast<LeafletPenaltyForceParams*>(ctx);
    BoundaryInfo* boundary_info = params->boundary_info;
    if (boundary_info->has_boundary_id(elem, side, 4))
    {
        const double kappa_s = params->kappa_s;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            F(d) = kappa_s * (X(d) - x(d));
        }
    }
    else
    {
        F.zero();
    }
    return;
}

void
zero_boundary_condition_fcn(DenseVector<Real>& output, const libMesh::Point& /*p*/, Real /*time*/)
{
    output(0) = 0.0;
    return;
}

void
one_boundary_condition_fcn(DenseVector<Real>& output, const libMesh::Point& /*p*/, Real /*time*/)
{
    output(0) = 1.0;
    return;
}

void
assemble_poisson(EquationSystems& es, const std::string& system_name)
{
    const MeshBase& mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();
    LinearImplicitSystem& system = es.get_system<LinearImplicitSystem>(system_name);
    const DofMap& dof_map = system.get_dof_map();
    FEType fe_type = dof_map.variable_type(0);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    QGauss qrule(dim, FIFTH);
    fe->attach_quadrature_rule(&qrule);
    const std::vector<Real>& JxW = fe->get_JxW();
    const std::vector<std::vector<Real>>& phi = fe->get_phi();
    const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();
    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;
    std::vector<dof_id_type> dof_indices;
    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
    for (; el != end_el; ++el)
    {
        const Elem* elem = *el;
        dof_map.dof_indices(elem, dof_indices, 0);
        fe->reinit(elem);
        Ke.resize(dof_indices.size(), dof_indices.size());
        Fe.resize(dof_indices.size());
        for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
        {
            for (unsigned int i = 0; i < phi.size(); i++)
            {
                for (unsigned int j = 0; j < phi.size(); j++)
                {
                    Ke(i, j) += (dphi[i][qp] * dphi[j][qp]) * JxW[qp];
                }
            }
        }
        dof_map.heterogenously_constrain_element_matrix_and_vector(Ke, Fe, dof_indices);
        system.matrix->add_matrix(Ke, dof_indices);
        system.rhs->add_vector(Fe, dof_indices);
    }
    return;
}

double
kappa_fcn(double x, double xhalf, double tau)
{
    return (tanh((x - xhalf) / tau) + tanh(xhalf / tau)) / (1.0 + tanh(xhalf / tau));
}

static double tau = 0.015, xhalf = 0.05;
void
buttress_force(VectorValue<double>& F,
               const TensorValue<double>& /*FF*/,
               const libMesh::Point& x,
               const libMesh::Point& X,
               Elem* const elem,
               const std::vector<const vector<double>*>& /*var_data*/,
               const std::vector<const std::vector<VectorValue<double>>*>& /*grad_var_data*/,
               double /*time*/,
               void* ctx)
{
#if (1)
    LeafletPenaltyForceParams* params = static_cast<LeafletPenaltyForceParams*>(ctx);
    double kappa_p = params->kappa_p;
    VectorValue<double> com_pt;
    if (X(0) > -0.74)
        com_pt = { 0.1, -0.6 };
    else
        com_pt = { -1.7, -0.5 };
    double rest_length = (X - com_pt).norm();
    double cur_dist = (x - com_pt).norm();
    if (cur_dist - rest_length > 0.0)
    {
        kappa_p *= kappa_fcn(cur_dist - rest_length, xhalf, tau);
        // We have stretched, apply a restoring force
        F = kappa_p * (1 - rest_length / cur_dist) * (com_pt - x);
    }
    else
    {
        F.zero();
    }
#else
    if (X(1) - x(1) > 0)
    {
        LeafletPenaltyForceParams* params = static_cast<LeafletPenaltyForceParams*>(ctx);
        const double kappa_p = params->kappa_p;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            F(d) = kappa_p * (X(d) - x(d));
        }
    }
    else
    {
        F.zero();
    }
#endif
    return;
}

static double dt_const = 0.0;

double
find_dt(const std::shared_ptr<CBFinder>& cb_finder, const LeafletStressParams& leaflet_params)
{
    const double C10_min = leaflet_params.C10_min;
    const double C10_max = leaflet_params.C10_max;
    const double cb = cb_finder->maxCB();
    const double c10 = findStiffness(C10_min, C10_max, cb, sf_max);
    plog << "  c10         : " << c10 << "\n";
    plog << "  dt          : " << dt_const / std::sqrt(c10) << "\n";
    return dt_const / std::sqrt(c10);
}
} // namespace

void compute_variance(const int u_idx, const int ubar_idx, const int uvar_idx, Pointer<PatchHierarchy<NDIM>> hierarchy);

int
main(int argc, char* argv[])
{
    // Initialize libMesh, PETSc, MPI, and SAMRAI.
    IBTKInit ibtk_init(argc, argv);
    LibMeshInit& init = ibtk_init.getLibMeshInit();

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && !app_initializer->getVisItDataWriter().isNull();
        const bool uses_exodus = dump_viz_data && !app_initializer->getExodusIIFilename().empty();
        const string viz_dump_dirname = app_initializer->getVizDumpDirectory();
        const string leaflet_filename = viz_dump_dirname + "/leaflet.ex2";
        const string housing_filename = viz_dump_dirname + "/housing.ex2";
        const string leaflet_bdry_filename = viz_dump_dirname + "/leaflet_bdry.ex2";
        const string housing_bdry_filename = viz_dump_dirname + "/housing_bdry.ex2";
        const string reaction_bdry_filename = viz_dump_dirname + "/reaction_bdry.ex2";

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();
        const string restart_read_dirname = app_initializer->getRestartReadDirectory();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        dy = input_db->getDouble("DY");

        // Load the FE meshes.
        pout << "Loading the meshes...\n";
        const bool housing_second_order_mesh = (input_db->getString("housing_elem_order") == "SECOND");
        const bool leaflet_second_order_mesh = (input_db->getString("leaflet_elem_order") == "SECOND");

        ReplicatedMesh leaflet_mesh(init.comm(), NDIM);
        leaflet_mesh.read(input_db->getString("LEAFLET_MESH_FILENAME"));
        if (leaflet_second_order_mesh)
        {
            leaflet_mesh.all_second_order(true);
        }
        else
        {
            leaflet_mesh.all_first_order();
        }

        ReplicatedMesh housing_solid_mesh(init.comm(), NDIM);
        housing_solid_mesh.read(input_db->getString("HOUSING_MESH_FILENAME"));
        if (housing_second_order_mesh)
        {
            housing_solid_mesh.all_second_order(true);
        }
        else
        {
            housing_solid_mesh.all_first_order();
        }
        MeshBase& housing_mesh = housing_solid_mesh;

        vector<MeshBase*> vol_meshes(NUM_PARTS);
        vol_meshes[LEAFLET_PART] = &leaflet_mesh;
        vol_meshes[HOUSING_PART] = &housing_mesh;

        // Pull in some libMesh helper functions.
        using MeshTools::Modification::rotate;
        using MeshTools::Modification::translate;

        // Setup data for imposing constraints.
        Pointer<Database> housing_params_db = app_initializer->getComponentDatabase("HousingParams");
        PenaltyStressParams housing_stress_params;
        PenaltyForceParams housing_body_force_params;
        housing_stress_params.c1_s = housing_params_db->getDoubleWithDefault("C1_S", 0.0);
        housing_stress_params.c1_p = housing_params_db->getDoubleWithDefault("C1_P", 0.0);
        housing_body_force_params.kappa_s = housing_params_db->getDoubleWithDefault("KAPPA_S_BODY", 0.0);
        housing_body_force_params.kappa_p = housing_params_db->getDoubleWithDefault("KAPPA_P_BODY", 0.0);

        Pointer<Database> leaflet_params_db = app_initializer->getComponentDatabase("LeafletParams");
        LeafletStressParams leaflet_stress_params;
        LeafletPenaltyForceParams leaflet_penalty_surface_force_params;
        LeafletPenaltyForceParams buttress_force_params;
        leaflet_stress_params.C10 = leaflet_params_db->getDoubleWithDefault("C10", 83850);
        leaflet_stress_params.C10_min = leaflet_params_db->getDoubleWithDefault("C10_min", leaflet_stress_params.C10);
        leaflet_stress_params.C10_max =
            leaflet_params_db->getDoubleWithDefault("C10_max", leaflet_stress_params.C10 * 10);
        leaflet_stress_params.C01 = leaflet_params_db->getDoubleWithDefault("C01", 11.163);
        leaflet_stress_params.k1 = leaflet_params_db->getDoubleWithDefault("K1", 103719.1);
        leaflet_stress_params.k2 = leaflet_params_db->getDoubleWithDefault("K2", 37.1714);
        leaflet_stress_params.a_disp = leaflet_params_db->getDoubleWithDefault("a_disp", 0.0);
        leaflet_stress_params.beta_s = leaflet_params_db->getDoubleWithDefault("BETA_S", 0.0);
        leaflet_penalty_surface_force_params.boundary_info = &leaflet_mesh.get_boundary_info();
        leaflet_penalty_surface_force_params.kappa_s = leaflet_params_db->getDoubleWithDefault("KAPPA_S_SURFACE", 0.0);
        leaflet_penalty_surface_force_params.kappa_p = leaflet_params_db->getDoubleWithDefault("KAPPA_P_BODY", 0.0);
        buttress_force_params.kappa_p = input_db->getDouble("BUTTRESS_KAPPA");

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<INSHierarchyIntegrator> navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
            "INSStaggeredHierarchyIntegrator",
            app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
        Pointer<IBFEMethod> ibfe_method_ops =
            new IBFEMethod("IBFEMethod",
                           app_initializer->getComponentDatabase("IBFEMethod"),
                           vol_meshes,
                           app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"),
                           /*register_for_restart*/ true,
                           app_initializer->getRestartReadDirectory(),
                           app_initializer->getRestartRestoreNumber());
        vector<Pointer<IBStrategy>> ib_ops_vec;
        ib_ops_vec.push_back(ibfe_method_ops);
        Pointer<IBStrategySet> ib_ops_set = new IBStrategySet(ib_ops_vec.begin(), ib_ops_vec.end());
        Pointer<IBExplicitHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_ops_set,
                                              navier_stokes_integrator);
        //        Pointer<SBAdvDiffIntegrator> adv_diff_integrator = new SBAdvDiffIntegrator(
        //            "AdvDiffIntegrator", app_initializer->getComponentDatabase("AdvDiffIntegrator"), time_integrator,
        //            true);
        Pointer<LSAdvDiffIntegrator> adv_diff_integrator = new LSAdvDiffIntegrator(
            "AdvDiffIntegrator", app_initializer->getComponentDatabase("AdvDiffIntegrator"), true);
        //        navier_stokes_integrator->registerAdvDiffHierarchyIntegrator(adv_diff_integrator);
        adv_diff_integrator->registerAdvectionVelocity(navier_stokes_integrator->getAdvectionVelocityVariable());
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Configure the IBFE solver.
        pout << "\nConfiguring the solver...\n";
        string leaflet_kernel_fcn = input_db->getStringWithDefault("LEAFLET_KERNEL_FCN", "IB_3");
        FEDataManager::InterpSpec leaflet_interp_spec = ibfe_method_ops->getDefaultInterpSpec();
        leaflet_interp_spec.kernel_fcn = leaflet_kernel_fcn;
        ibfe_method_ops->setInterpSpec(leaflet_interp_spec, LEAFLET_PART);
        FEDataManager::SpreadSpec leaflet_spread_spec = ibfe_method_ops->getDefaultSpreadSpec();
        leaflet_spread_spec.kernel_fcn = leaflet_kernel_fcn;
        ibfe_method_ops->setSpreadSpec(leaflet_spread_spec, LEAFLET_PART);

        string housing_kernel_fcn = input_db->getStringWithDefault("HOUSING_KERNEL_FCN", "PIECEWISE_LINEAR");
        FEDataManager::InterpSpec housing_interp_spec = ibfe_method_ops->getDefaultInterpSpec();
        housing_interp_spec.kernel_fcn = housing_kernel_fcn;
        ibfe_method_ops->setInterpSpec(housing_interp_spec, HOUSING_PART);
        FEDataManager::SpreadSpec housing_spread_spec = ibfe_method_ops->getDefaultSpreadSpec();
        housing_spread_spec.kernel_fcn = housing_kernel_fcn;
        ibfe_method_ops->setSpreadSpec(housing_spread_spec, HOUSING_PART);

        ibfe_method_ops->initializeFEEquationSystems();
        pout << "\nSolver configured.\n";

        EquationSystems* leaflet_systems = ibfe_method_ops->getFEDataManager(LEAFLET_PART)->getEquationSystems();
        EquationSystems* housing_systems = ibfe_method_ops->getFEDataManager(HOUSING_PART)->getEquationSystems();

        Pointer<IBFEPostProcessor> ib_post_processor =
            new IBFECentroidPostProcessor("IBFEPostProcessor", ibfe_method_ops->getFEDataManager(LEAFLET_PART));

        pout << "\nSetting up body variables...\n";
        std::vector<int> vars(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d) vars[d] = d;
        vector<SystemData> leaflet_sys_data(2);
        leaflet_sys_data[0] = SystemData("v1_0", vars);
        leaflet_sys_data[1] = SystemData("v2_0", vars);
        vector<SystemData> v1_sys_data(1);
        v1_sys_data[0] = SystemData("v1_0", vars);
        vector<SystemData> v2_sys_data(1);
        v2_sys_data[0] = SystemData("v2_0", vars);
        bool from_restart = RestartManager::getManager()->isFromRestart();

        dt_const = input_db->getDouble("DT_CONST");
        auto cb_finder = std::make_shared<CBFinder>();
        std::pair<LeafletStressParams*, std::shared_ptr<CBFinder>*> param_cb_finder_pair(&leaflet_stress_params,
                                                                                         &cb_finder);

        for (unsigned int part = 0; part < NUM_PARTS; ++part)
        {
            if (part == LEAFLET_PART)
            {
                EquationSystems* equation_systems = ibfe_method_ops->getFEDataManager(part)->getEquationSystems();
                if (!from_restart)
                {
                    System& u_poisson_system = equation_systems->add_system<LinearImplicitSystem>("u system");
                    System& v_poisson_system = equation_systems->add_system<LinearImplicitSystem>("v system");
                    FEFamily family = LAGRANGE;
                    Order order = FIRST;
                    u_poisson_system.add_variable("u", order, family);
                    v_poisson_system.add_variable("v", order, family);
                    u_poisson_system.attach_assemble_function(assemble_poisson);
                    v_poisson_system.attach_assemble_function(assemble_poisson);

                    // Set up boundary conditions.
                    std::vector<unsigned int> variables(1);
                    variables[0] = 0;

                    AnalyticFunction<Real> zero_boundary_condition_mesh_fcn(zero_boundary_condition_fcn);
                    zero_boundary_condition_mesh_fcn.init();
                    AnalyticFunction<Real> one_boundary_condition_mesh_fcn(one_boundary_condition_fcn);
                    one_boundary_condition_mesh_fcn.init();

                    std::set<boundary_id_type> u_zero_boundary_ids;
                    u_zero_boundary_ids.insert(1);
                    std::set<boundary_id_type> v_zero_boundary_ids;
                    v_zero_boundary_ids.insert(2);

                    std::set<boundary_id_type> u_one_boundary_ids;
                    u_one_boundary_ids.insert(4);
                    std::set<boundary_id_type> v_one_boundary_ids;
                    v_one_boundary_ids.insert(3);

                    DirichletBoundary u_zero_dirichlet_bc(
                        u_zero_boundary_ids, variables, &zero_boundary_condition_mesh_fcn);
                    DirichletBoundary v_zero_dirichlet_bc(
                        v_zero_boundary_ids, variables, &zero_boundary_condition_mesh_fcn);
                    DirichletBoundary u_one_dirichlet_bc(
                        u_one_boundary_ids, variables, &one_boundary_condition_mesh_fcn);
                    DirichletBoundary v_one_dirichlet_bc(
                        v_one_boundary_ids, variables, &one_boundary_condition_mesh_fcn);
                    u_poisson_system.get_dof_map().add_dirichlet_boundary(u_zero_dirichlet_bc);
                    v_poisson_system.get_dof_map().add_dirichlet_boundary(v_zero_dirichlet_bc);
                    u_poisson_system.get_dof_map().add_dirichlet_boundary(u_one_dirichlet_bc);
                    v_poisson_system.get_dof_map().add_dirichlet_boundary(v_one_dirichlet_bc);

                    System& v1_system = equation_systems->add_system<System>("v1_0");
                    System& v2_system = equation_systems->add_system<System>("v2_0");
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        ostringstream os;
                        os << "v1_0_" << d;
                        v1_system.add_variable(os.str(), CONSTANT, MONOMIAL);
                    }
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        ostringstream os;
                        os << "v2_0_" << d;
                        v2_system.add_variable(os.str(), CONSTANT, MONOMIAL);
                    }
                    v1_system.assemble_before_solve = false;
                    v2_system.assemble();
                }

                IBFEMethod::PK1StressFcnData* PK1_stress_data = new IBFEMethod::PK1StressFcnData(); // memory leak!
                PK1_stress_data->fcn = leaflet_stress_fcn;
                PK1_stress_data->system_data = leaflet_sys_data;
                PK1_stress_data->ctx = &param_cb_finder_pair;
                PK1_stress_data->quad_order = Utility::string_to_enum<libMesh::Order>(
                    input_db->getStringWithDefault("PK1_QUAD_ORDER", leaflet_second_order_mesh ? "FIFTH" : "THIRD"));
                ibfe_method_ops->registerPK1StressFunction(*PK1_stress_data, part);

                IBFEMethod::PK1StressFcnData* PK1_penalty_stress_data =
                    new IBFEMethod::PK1StressFcnData(); // memory leak!
                PK1_penalty_stress_data->fcn = leaflet_penalty_stress_fcn;
                PK1_penalty_stress_data->ctx = &leaflet_stress_params;
                PK1_penalty_stress_data->quad_order =
                    Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault(
                        "PK1_PENALTY_QUAD_ORDER", leaflet_second_order_mesh ? "THIRD" : "FIRST"));
                ibfe_method_ops->registerPK1StressFunction(*PK1_penalty_stress_data, part);

                IBFEMethod::LagSurfaceForceFcnData surface_fcn_data;
                surface_fcn_data.fcn = leaflet_penalty_surface_force_fcn;
                surface_fcn_data.ctx = &leaflet_penalty_surface_force_params;
                ibfe_method_ops->registerLagSurfaceForceFunction(surface_fcn_data, part);

                IBFEMethod::LagBodyForceFcnData body_fcn_data;
                body_fcn_data.fcn = buttress_force;
                body_fcn_data.ctx = &buttress_force_params;
                ibfe_method_ops->registerLagBodyForceFunction(body_fcn_data, part);

                if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
                {
                    ibfe_method_ops->registerStressNormalizationPart(part);
                }

                // Setup post processing.
                ib_post_processor->registerTensorVariable("FF", MONOMIAL, CONSTANT, IBFEPostProcessor::FF_fcn);

                ib_post_processor->registerVectorVariable(
                    "v1", MONOMIAL, CONSTANT, IBFEPostProcessor::deformed_material_axis_fcn, v1_sys_data);

                ib_post_processor->registerVectorVariable(
                    "v2", MONOMIAL, CONSTANT, IBFEPostProcessor::deformed_material_axis_fcn, v2_sys_data);

                ib_post_processor->registerScalarVariable(
                    "lambda_v1", MONOMIAL, CONSTANT, IBFEPostProcessor::material_axis_stretch_fcn, v1_sys_data);

                ib_post_processor->registerScalarVariable(
                    "lambda_v2", MONOMIAL, CONSTANT, IBFEPostProcessor::material_axis_stretch_fcn, v2_sys_data);

                ib_post_processor->registerTensorVariable("sigma_dev",
                                                          MONOMIAL,
                                                          CONSTANT,
                                                          IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                          PK1_stress_data->system_data,
                                                          PK1_stress_data);

                ib_post_processor->registerTensorVariable("sigma_dil",
                                                          MONOMIAL,
                                                          CONSTANT,
                                                          IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                          PK1_penalty_stress_data->system_data,
                                                          PK1_penalty_stress_data);

                Pointer<hier::Variable<NDIM>> p_var = navier_stokes_integrator->getPressureVariable();
                Pointer<VariableContext> p_current_ctx = navier_stokes_integrator->getCurrentContext();
                HierarchyGhostCellInterpolation::InterpolationTransactionComponent p_ghostfill(
                    /*data_idx*/ -1,
                    "LINEAR_REFINE",
                    /*use_cf_bdry_interpolation*/ false,
                    "CONSERVATIVE_COARSEN",
                    "LINEAR");
                FEDataManager::InterpSpec p_interp_spec("PIECEWISE_LINEAR",
                                                        QGAUSS,
                                                        FIFTH,
                                                        /*use_adaptive_quadrature*/ false,
                                                        /*point_density*/ 2.0,
                                                        /*use_consistent_mass_matrix*/ true,
                                                        /*use_nodal_quadrature*/ false);
                ib_post_processor->registerInterpolatedScalarEulerianVariable(
                    "p_f", LAGRANGE, FIRST, p_var, p_current_ctx, p_ghostfill, p_interp_spec);
            }
            if (part == HOUSING_PART)
            {
                IBFEMethod::PK1StressFcnData PK1_stress_data;
                PK1_stress_data.fcn = penalty_stress_fcn;
                PK1_stress_data.ctx = &housing_stress_params;
                PK1_stress_data.quad_order = Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault(
                    "PK1_QUAD_ORDER_HOUSING", housing_second_order_mesh ? "FIFTH" : "THIRD"));
                ibfe_method_ops->registerPK1StressFunction(PK1_stress_data, part);

                IBFEMethod::LagBodyForceFcnData body_fcn_data;
                body_fcn_data.fcn = penalty_body_force_fcn;
                body_fcn_data.ctx = &housing_body_force_params;
                ibfe_method_ops->registerLagBodyForceFunction(body_fcn_data, part);

                if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
                {
                    pout << "ELIMINATE_PRESSURE_JUMPS is DISABLED for the housing mesh!\n";
                }
            }
        }

        pout << "Setting up level set\n";
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");
        auto bdry_id_for_rcn = static_cast<boundary_id_type>(input_db->getInteger("BDRY_ID_FOR_RCN"));
        adv_diff_integrator->registerLevelSetVariable(ls_var);
        std::vector<std::set<boundary_id_type>> bdry_id_vec = { { 1, 2, 3 }, { 3 }, { bdry_id_for_rcn } };
        std::vector<FEDataManager*> fe_data_managers = { ibfe_method_ops->getFEDataManager(LEAFLET_PART),
                                                         ibfe_method_ops->getFEDataManager(HOUSING_PART) };
        std::vector<unsigned int> parts = { 0, 1, 0 };
        auto vol_bdry_mesh_mapping =
            std::make_shared<VolumeBoundaryMeshMapping>("VolBdryMeshMap",
                                                        app_initializer->getComponentDatabase("VolBdryMeshMap"),
                                                        vol_meshes,
                                                        fe_data_managers,
                                                        bdry_id_vec,
                                                        parts,
                                                        app_initializer->getRestartReadDirectory(),
                                                        app_initializer->getRestartRestoreNumber());
        vol_bdry_mesh_mapping->initializeEquationSystems();
        Pointer<CutCellVolumeMeshMapping> cut_cell_mapping =
            new CutCellVolumeMeshMapping("CutCellMeshMapping",
                                         app_initializer->getComponentDatabase("CutCellMapping"),
                                         vol_bdry_mesh_mapping->getMeshPartitioners({ 0, 1 }));
        Pointer<CutCellVolumeMeshMapping> cut_cell_rcn_mapping =
            new CutCellVolumeMeshMapping("CutCellRcnMeshMapping",
                                         app_initializer->getComponentDatabase("CutCellMapping"),
                                         vol_bdry_mesh_mapping->getMeshPartitioner(2));
        Pointer<LSFromMesh> ls_fcn = new LSFromMesh("LSFcn", patch_hierarchy, cut_cell_mapping, false);
        ls_fcn->registerBdryFcn(bdry_fcn);
        ls_fcn->registerNormalReverseDomainId({ 5, 6, 9, 12, 11 });
        ls_fcn->registerNormalReverseElemId({ 632, 633, 634 });
        adv_diff_integrator->registerLevelSetVolFunction(ls_var, ls_fcn);
        //        adv_diff_integrator->registerGeneralBoundaryMeshMapping(vol_bdry_mesh_mapping);

        Pointer<RBFReconstructCacheOS> reconstruct_cache_from_centroids = new RBFReconstructCacheOS(1);
        Pointer<RBFReconstructCacheOS> reconstruct_cache_to_centroids = new RBFReconstructCacheOS(1);
        Pointer<RBFReconstructCacheOS> reconstruct_cache = new RBFReconstructCacheOS(1);
        adv_diff_integrator->registerReconstructionCacheToCentroids(reconstruct_cache_to_centroids, ls_var);
        adv_diff_integrator->registerReconstructionCacheFromCentroids(reconstruct_cache_from_centroids, ls_var);

        EquationSystems* leaflet_bdry_eq = cut_cell_mapping->getMeshPartitioner(LEAFLET_PART)->getEquationSystems();
        EquationSystems* housing_bdry_eq = cut_cell_mapping->getMeshPartitioner(HOUSING_PART)->getEquationSystems();
        EquationSystems* reaction_bdry_eq = vol_bdry_mesh_mapping->getMeshPartitioner(2)->getEquationSystems();

        pout << "Setting up transported quantity\n";
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<CartGridFunction> Q_init = new QFcn("QInit", app_initializer->getComponentDatabase("QInit"));

        SAMRAI::tbox::Pointer<RobinBcCoefStrategy<NDIM>> Q_bcs;
        if (grid_geometry->getPeriodicShift().min() == 0)
            Q_bcs = new muParserRobinBcCoefs("Q_bcs", app_initializer->getComponentDatabase("Q_bcs"), grid_geometry);

        adv_diff_integrator->registerTransportedQuantity(Q_var);
        adv_diff_integrator->setInitialConditions(Q_var, Q_init);
        adv_diff_integrator->setPhysicalBcCoef(Q_var, Q_bcs.getPointer());
        D_coef = input_db->getDouble("D_COEF");
        adv_diff_integrator->setDiffusionCoefficient(Q_var, D_coef);
        adv_diff_integrator->restrictToLevelSet(Q_var, ls_var);
        adv_diff_integrator->setAdvectionVelocity(Q_var, navier_stokes_integrator->getAdvectionVelocityVariable());
        auto convective_reconstruct =
            std::make_shared<RBFOneSidedReconstructions>("OneSided", Reconstruct::RBFPolyOrder::QUADRATIC, 7);
        adv_diff_integrator->registerAdvectionReconstruction(Q_var, convective_reconstruct);

        // Setup reactions
        auto sb_coupling_manager =
            std::make_shared<SBSurfaceFluidCouplingManager>("CouplingManager",
                                                            app_initializer->getComponentDatabase("CouplingManager"),
                                                            vol_bdry_mesh_mapping->getMeshPartitioner(2));
        sb_coupling_manager->registerReconstructCache(reconstruct_cache);
        sb_coupling_manager->registerFluidConcentration(Q_var);
        std::string sf_name = "SurfaceConcentration";
        sb_coupling_manager->registerSurfaceConcentration(sf_name);
        sb_coupling_manager->registerSurfaceReactionFunction(sf_name, sf_ode);
        sb_coupling_manager->registerFluidBoundaryCondition(Q_var, a_fcn, g_fcn);
        sb_coupling_manager->registerFluidSurfaceDependence(sf_name, Q_var);
        sf_init_val = input_db->getDouble("SF_INIT");
        sb_coupling_manager->registerInitialConditions(sf_name, sf_init);
        sb_coupling_manager->initializeFEData();
        Pointer<SBIntegrator> sb_integrator = new SBIntegrator("SBIntegrator", sb_coupling_manager);
        Pointer<SBBoundaryConditions> bdry_conds = new SBBoundaryConditions(
            "SBBoundaryConditions", sb_coupling_manager->getFLName(Q_var), sb_coupling_manager, cut_cell_rcn_mapping);
        bdry_conds->setFluidContext(adv_diff_integrator->getCurrentContext());
        //        adv_diff_integrator->registerSBIntegrator(sb_integrator, ls_var);
        //        adv_diff_integrator->registerLevelSetSBDataManager(ls_var, sb_coupling_manager);
        k_on = input_db->getDouble("K_ON");
        k_off = input_db->getDouble("K_OFF");
        sf_max = input_db->getDouble("SF_MAX");
        sf_scale = input_db->getDoubleWithDefault("SF_SCALE", sf_scale);
        fl_scale = input_db->getDoubleWithDefault("FL_SCALE", fl_scale);
        use_feedback = input_db->getBool("USE_FEEDBACK");
        stiff_right = input_db->getBool("STIFF_RIGHT");
        time_to_start = input_db->getDouble("TIME_TO_START_RCNS");

        // Set up diffusion operators
        Pointer<LSCutCellLaplaceOperator> rhs_oper = new LSCutCellLaplaceOperator(
            "LSCutCellRHSOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        Pointer<LSCutCellLaplaceOperator> sol_oper = new LSCutCellLaplaceOperator(
            "LSCutCellSolOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        Pointer<PETScKrylovPoissonSolver> Q_helmholtz_solver = new PETScKrylovPoissonSolver(
            "PoissonSolver", app_initializer->getComponentDatabase("PoissonSolver"), "poisson_solve_");
        sol_oper->setBoundaryConditionOperator(bdry_conds);
        rhs_oper->setBoundaryConditionOperator(bdry_conds);
        Q_helmholtz_solver->setOperator(sol_oper);
        adv_diff_integrator->setHelmholtzSolver(Q_var, Q_helmholtz_solver);
        adv_diff_integrator->setHelmholtzRHSOperator(Q_var, rhs_oper);

        // Create Eulerian boundary condition specification objects.
        CirculationModel circ_model("circ_model", input_db->getDatabase("BcCoefs"));
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        for (int d = 0; d < NDIM; ++d) u_bc_coefs[d] = new VelocityBcCoefs(&circ_model, d);
        navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        Pointer<FeedbackForcer> feedback_forcer =
            new FeedbackForcer(&circ_model, navier_stokes_integrator, patch_hierarchy);
        time_integrator->registerBodyForceFunction(feedback_forcer);

        pout << "Registering visit writers...\n";
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            time_integrator->registerVisItDataWriter(visit_data_writer);
            adv_diff_integrator->registerVisItDataWriter(visit_data_writer);
        }
        std::unique_ptr<ExodusII_IO> leaflet_io(uses_exodus ? new ExodusII_IO(leaflet_mesh) : nullptr);
        std::unique_ptr<ExodusII_IO> housing_io(uses_exodus ? new ExodusII_IO(housing_mesh) : nullptr);
        std::unique_ptr<ExodusII_IO> leaflet_bdry_io(uses_exodus ? new ExodusII_IO(leaflet_bdry_eq->get_mesh()) :
                                                                   nullptr);
        std::unique_ptr<ExodusII_IO> housing_bdry_io(uses_exodus ? new ExodusII_IO(housing_bdry_eq->get_mesh()) :
                                                                   nullptr);
        std::unique_ptr<ExodusII_IO> reaction_bdry_io(uses_exodus ? new ExodusII_IO(reaction_bdry_eq->get_mesh()) :
                                                                    nullptr);

        if (leaflet_io) leaflet_io->append(from_restart);
        if (housing_io) housing_io->append(from_restart);
        if (leaflet_bdry_io) leaflet_bdry_io->append(from_restart);
        if (housing_bdry_io) housing_bdry_io->append(from_restart);
        if (reaction_bdry_io) reaction_bdry_io->append(from_restart);

        // Initialize FE data.
        pout << "\nInitializing FE data...\n";
        ibfe_method_ops->initializeFEData();
        if (ib_post_processor) ib_post_processor->initializeFEData();
        vol_bdry_mesh_mapping->initializeFEData();
        if (!from_restart) sb_coupling_manager->fillInitialConditions();

        // Setup CBFinder. Note that this must be done here so that the FEDataManager is already set up.
        cb_finder = std::make_shared<CBFinder>(
            sf_name, vol_meshes[LEAFLET_PART], sb_coupling_manager, ibfe_method_ops->getFEDataManager(LEAFLET_PART));

        // Initialize hierarchy configuration and data on all patches.
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy, coarsest_ln, finest_ln);

#if 0
        // Set up fiber structure.
        {
            EquationSystems* equation_systems = ibfe_method_ops->getFEDataManager(LEAFLET_PART)->getEquationSystems();
            System& u_poisson_system = equation_systems->get_system<LinearImplicitSystem>("u system");
            System& v_poisson_system = equation_systems->get_system<LinearImplicitSystem>("v system");
            u_poisson_system.solve();
            v_poisson_system.solve();
            MeshFunction u_fcn(*equation_systems, *u_poisson_system.current_local_solution,
                               u_poisson_system.get_dof_map(), vector<unsigned int>(1, 0));
            MeshFunction v_fcn(*equation_systems, *v_poisson_system.current_local_solution,
                               v_poisson_system.get_dof_map(), vector<unsigned int>(1, 0));
            u_fcn.init();
            v_fcn.init();
            System& v1_system = equation_systems->get_system<System>("v1_0");
            System& v2_system = equation_systems->get_system<System>("v2_0");
            const int v1_sys_num = v1_system.number();
            const int v2_sys_num = v2_system.number();
            MeshBase::const_element_iterator el = leaflet_mesh.active_local_elements_begin();
            const MeshBase::const_element_iterator end_el = leaflet_mesh.active_local_elements_end();
            for (; el != end_el; ++el)
            {
                const Elem* elem = *el;
                const libMesh::Point& X = elem->centroid();
                const Gradient& grad_u = u_fcn.gradient(X).unit();
                const Gradient& grad_v = v_fcn.gradient(X).unit();
                const VectorValue<double> v1 = grad_u.cross(grad_v).unit();
                const VectorValue<double> v2 = (v1.cross(grad_v)).unit();
                for (int d = 0; d < NDIM; ++d)
                {
                    v1_system.solution->set(elem->dof_number(v1_sys_num, d, 0), v1(d));
                    v2_system.solution->set(elem->dof_number(v2_sys_num, d, 0), v2(d));
                }
            }
	     //    v1_system.solution->close();
         //    v1_system.solution->localize(*v1_system.current_local_solution);
         //    v2_system.solution->close();
         //    v2_system.solution->localize(*v2_system.current_local_solution);
            v1_system.solution->close();
            copy_and_synch(*v1_system.solution, *v1_system.current_local_solution);
            v2_system.solution->close();
            copy_and_synch(*v2_system.solution, *v2_system.current_local_solution);
        }
#endif

        Pointer<Database> avg_manager_u_db = app_initializer->getComponentDatabase("AvgManagerU");
        HierarchyAveragedDataManager u_avg_manager("UAvgManager",
                                                   navier_stokes_integrator->getVelocityVariable(),
                                                   avg_manager_u_db,
                                                   patch_hierarchy,
                                                   grid_geometry,
                                                   false);
        Pointer<Database> avg_manager_tke_db = app_initializer->getComponentDatabase("AvgManagerTKE");
        HierarchyAveragedDataManager tke_avg_manager("TKEAvgManager",
                                                     navier_stokes_integrator->getVelocityVariable(),
                                                     avg_manager_tke_db,
                                                     patch_hierarchy,
                                                     grid_geometry,
                                                     false);
        const std::set<double>& time_pts = u_avg_manager.getSnapshotTimePts();
        const double t_start = avg_manager_u_db->getDouble("t_start");
        const double avg_freq = avg_manager_u_db->getDouble("avg_freq");
        double next_save_time = t_start;
        bool finding_tke = false;
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int ubar_idx = var_db->registerVariableAndContext(
            navier_stokes_integrator->getVelocityVariable(), var_db->getContext("TKE"), 1);

        // Deallocate initialization objects.
        app_initializer.setNull();
        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        double viz_dump_time_interval = input_db->getDouble("VIZ_DUMP_TIME_INTERVAL");
        double next_viz_dump_time = 0.0;
        int viz_dump_iteration_num = 1;
        while (loop_time > 0.0 &&
               (next_viz_dump_time < loop_time || MathUtilities<double>::equalEps(loop_time, next_viz_dump_time)))
        {
            next_viz_dump_time += viz_dump_time_interval;
            viz_dump_iteration_num += 1;
        }

        // Main time step loop.
        pout << "Entering main time step loop...\n";
        const double loop_time_end = time_integrator->getEndTime();
        while (!MathUtilities<double>::equalEps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            if (dump_viz_data &&
                (MathUtilities<double>::equalEps(loop_time, next_viz_dump_time) || loop_time >= next_viz_dump_time))
            {
                pout << "\n\nWriting visualization files...\n\n";
                if (uses_visit)
                {
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                }
                if (uses_exodus)
                {
                    if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                    if (leaflet_io)
                        leaflet_io->write_timestep(
                            leaflet_filename, *leaflet_systems, viz_dump_iteration_num, loop_time);
                    if (housing_io)
                        housing_io->write_timestep(
                            housing_filename, *housing_systems, viz_dump_iteration_num, loop_time);
                    if (leaflet_bdry_io)
                        leaflet_bdry_io->write_timestep(
                            leaflet_bdry_filename, *leaflet_bdry_eq, viz_dump_iteration_num, loop_time);
                    if (housing_bdry_io)
                        housing_bdry_io->write_timestep(
                            housing_bdry_filename, *housing_bdry_eq, viz_dump_iteration_num, loop_time);
                    if (reaction_bdry_io)
                        reaction_bdry_io->write_timestep(
                            reaction_bdry_filename, *reaction_bdry_eq, viz_dump_iteration_num, loop_time);
                }
                next_viz_dump_time += viz_dump_time_interval;
                viz_dump_iteration_num += 1;
            }

            // Determine if we need to update the average
            if (loop_time > next_save_time && !finding_tke)
            {
                auto var_db = VariableDatabase<NDIM>::getDatabase();
                const int u_idx = var_db->mapVariableAndContextToIndex(navier_stokes_integrator->getVelocityVariable(),
                                                                       navier_stokes_integrator->getCurrentContext());
                HierarchyMathOps hier_math_ops("HierarchyMathOps", patch_hierarchy);
                hier_math_ops.resetLevels(0, patch_hierarchy->getFinestLevelNumber());
                const int wgt_sc_idx = hier_math_ops.getSideWeightPatchDescriptorIndex();
                bool at_steady_state = u_avg_manager.updateTimeAveragedSnapshot(
                    u_idx, next_save_time, patch_hierarchy, "CONSERVATIVE_LINEAR_REFINE", wgt_sc_idx, 1.0e-8);
                if (at_steady_state)
                    pout << "Mean U at steady state!\n";
                else
                    pout << "Mean U not at steady state!\n";
                next_save_time += avg_freq;
                if (at_steady_state) finding_tke = true;
            }
            else if (loop_time > next_save_time && finding_tke)
            {
                // Time to find TKE
                auto var_db = VariableDatabase<NDIM>::getDatabase();
                const int u_idx = var_db->mapVariableAndContextToIndex(navier_stokes_integrator->getVelocityVariable(),
                                                                       navier_stokes_integrator->getCurrentContext());
                HierarchyMathOps hier_math_ops("HierarchyMathOps", patch_hierarchy);
                hier_math_ops.resetLevels(0, patch_hierarchy->getFinestLevelNumber());
                const int wgt_sc_idx = hier_math_ops.getSideWeightPatchDescriptorIndex();
                navier_stokes_integrator->allocatePatchData(
                    ubar_idx, loop_time, 0, patch_hierarchy->getFinestLevelNumber());
                // Get the average velocity field
                fill_snapshot_on_hierarchy(*u_avg_manager.getSnapshotCache(),
                                           ubar_idx,
                                           t_start,
                                           patch_hierarchy,
                                           "CONSERVATIVE_LINEAR_REFINE",
                                           1.0e-8);
                // Now compute variance
                compute_variance(u_idx, ubar_idx, ubar_idx, patch_hierarchy);
                // Now store variance
                bool at_steady_state = tke_avg_manager.updateTimeAveragedSnapshot(
                    ubar_idx, next_save_time, patch_hierarchy, "CONSERVATIVE_LINEAR_REFINE", wgt_sc_idx, 1.0e-8);
                if (at_steady_state)
                    pout << "TKE at steady state!\n";
                else
                    pout << "TKE not at steady state!\n";
                next_save_time += avg_freq;
                navier_stokes_integrator->deallocatePatchData(ubar_idx, 0, patch_hierarchy->getFinestLevelNumber());
                if (at_steady_state) break;
            }

            iteration_num = time_integrator->getIntegratorStep();

            pout << endl;
            pout << "++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            pout << "At beginning of timestep # " << iteration_num << endl;
            pout << "Simulation time is " << loop_time << endl;

            double dt = find_dt(cb_finder, leaflet_stress_params);
            dt = std::min(time_integrator->getMaximumTimeStepSize(), dt);

            Pointer<hier::Variable<NDIM>> U_var = navier_stokes_integrator->getVelocityVariable();
            Pointer<hier::Variable<NDIM>> P_var = navier_stokes_integrator->getPressureVariable();
            Pointer<VariableContext> current_ctx = navier_stokes_integrator->getCurrentContext();
            VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
            const int U_current_idx = var_db->mapVariableAndContextToIndex(U_var, current_ctx);
            const int P_current_idx = var_db->mapVariableAndContextToIndex(P_var, current_ctx);
            Pointer<HierarchyMathOps> hier_math_ops = navier_stokes_integrator->getHierarchyMathOps();
            const int wgt_cc_idx = hier_math_ops->getCellWeightPatchDescriptorIndex();
            const int wgt_sc_idx = hier_math_ops->getSideWeightPatchDescriptorIndex();
            circ_model.advanceTimeDependentData(
                dt, patch_hierarchy, U_current_idx, P_current_idx, wgt_cc_idx, wgt_sc_idx);

            time_integrator->advanceHierarchy(dt);

            pout << endl;
            pout << "J_min_leaflets is " << SAMRAI_MPI::minReduction(J_min_leaflets) << endl;
            J_min_leaflets = std::numeric_limits<double>::max();
            pout << "J_max_leaflets is " << SAMRAI_MPI::maxReduction(J_max_leaflets) << endl;
            J_max_leaflets = std::numeric_limits<double>::min();
            pout << "I1_min_leaflets is " << SAMRAI_MPI::minReduction(I1_min_leaflets) << endl;
            I1_min_leaflets = std::numeric_limits<double>::max();
            pout << "I1_max_leaflets is " << SAMRAI_MPI::maxReduction(I1_max_leaflets) << endl;
            I1_max_leaflets = std::numeric_limits<double>::min();
            // pout << "I4_min_leaflets is " << SAMRAI_MPI::minReduction(I4_min_leaflets) << endl;
            // I4_min_leaflets = std::numeric_limits<double>::max();
            // pout << "I4_max_leaflets is " << SAMRAI_MPI::maxReduction(I4_max_leaflets) << endl;
            // I4_max_leaflets = std::numeric_limits<double>::min();

            loop_time += dt;

            pout << endl;
            pout << "At end       of timestep # " << iteration_num << endl;
            pout << "Simulation time is " << loop_time << endl;
            pout << "++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            pout << endl;

            iteration_num += 1;

            if (dump_restart_data && (iteration_num % restart_dump_interval == 0))
            {
                pout << "\nWriting restart files...\n\n";
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
                ibfe_method_ops->writeFEDataToRestartFile(restart_dump_dirname, iteration_num);
                vol_bdry_mesh_mapping->writeFEDataToRestartFile(restart_dump_dirname, iteration_num);
            }

            if (dump_timer_data && (iteration_num % timer_dump_interval == 0))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
        }
        pout << "Deleting bc coefs\n";
        for (int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];
        pout << "Done deleting bc coefs\n";
    }
    pout << "Finished cleaning up\n";

    return 0;
} // main

void
compute_tke(const int u_idx, const int ubar_idx, const int uvar_idx, Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<SideData<NDIM, double>> ubar_data = patch->getPatchData(ubar_idx);
            Pointer<CellData<NDIM, double>> uvar_data = patch->getPatchData(uvar_idx);
            for (int axis = 0; axis < NDIM; ++axis)
            {
                for (SideIterator<NDIM> si(patch->getBox(), axis); si; si++)
                {
                    const SideIndex<NDIM>& idx = si();
                    const double u = (*u_data)(idx);
                    const double ubar = (*ubar_data)(idx);
                    (*uvar_data)(idx) = (u - ubar) * (u - ubar);
                }
            }
        }
    }
}
