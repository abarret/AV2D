#include "ADS/app_namespaces.h"

#include "CBFinder.h"

#include "libmesh/enum_preconditioner_type.h"
#include "libmesh/enum_solver_type.h"
#include "libmesh/fe_interface.h"
#include "libmesh/petsc_linear_solver.h"
#include "libmesh/petsc_matrix.h"

#include "petscmat.h"

namespace IBAMR
{
CBFinder::CBFinder(std::string sf_name,
                   MeshBase* vol_mesh,
                   std::shared_ptr<SBSurfaceFluidCouplingManager> sb_data_manager,
                   FEDataManager* fe_data_manager)
    : d_sf_name(std::move(sf_name)),
      d_sb_data_manager(std::move(sb_data_manager)),
      d_fe_data_manager(fe_data_manager),
      d_phi_sys_name(d_sf_name + "_PHI")
{
    BoundaryMesh* bdry_mesh = d_sb_data_manager->getMesh();
    std::map<dof_id_type, unsigned char> side_id_map;
    vol_mesh->boundary_info->get_side_and_node_maps(*bdry_mesh, d_node_id_map, side_id_map);
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
    const System& X_sys = eq_sys->get_system(fe_data_manager->COORDINATES_SYSTEM_NAME);
    d_elem_type = X_sys.get_dof_map().variable_type(0);

    // Set up harmonic interpolation variable
    ExplicitSystem& phi_sys = eq_sys->add_system<ExplicitSystem>(d_phi_sys_name);
    phi_sys.add_variable(d_phi_sys_name, d_elem_type);
    eq_sys->reinit();
    return;
}

void
CBFinder::setFEDataManager(FEDataManager* fe_data_manager)
{
    d_fe_data_manager = fe_data_manager;
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
    const System& X_sys = eq_sys->get_system(fe_data_manager->COORDINATES_SYSTEM_NAME);
    d_elem_type = X_sys.get_dof_map().variable_type(0);
}

void
CBFinder::registerSFName(std::string sf_name)
{
    d_sf_name = std::move(sf_name);
}

void
CBFinder::registerSBData(MeshBase* vol_mesh, std::shared_ptr<SBSurfaceFluidCouplingManager> sb_data_manager)
{
    d_sb_data_manager = std::move(sb_data_manager);
    std::map<dof_id_type, unsigned char> side_id_map;
    vol_mesh->boundary_info->get_side_and_node_maps(*d_sb_data_manager->getMesh(), d_node_id_map, side_id_map);
}

double
CBFinder::findCB(const TensorValue<double>& FF,
                 const libMesh::Point& X, // current location
                 const libMesh::Point& s, // reference location
                 Elem* const elem,
                 double time)
{
    if (!IBTK::abs_equal_eps(d_time, time))
    {
        plog << "Updating harmonic interpolant\n";
        formHarmonicInterpolant();
        d_time = time;
    }
    EquationSystems* eq_sys = d_fe_data_manager->getEquationSystems();
    System& phi_sys = eq_sys->get_system(d_phi_sys_name);
    NumericVector<double>* phi_vec = phi_sys.solution.get();

    std::unique_ptr<FEBase> fe = FEBase::build(NDIM, FEType());
    const std::vector<std::vector<double>>& phi = fe->get_phi();

    const libMesh::Point mapped_pt(FEInterface::inverse_map(NDIM, d_elem_type, elem, s));
    std::vector<libMesh::Point> pts = { mapped_pt };
    fe->reinit(elem, &pts);
    double sf = 0.0;
    int num_nodes = 0;
    // Note harmonic interpolant can make small negative values.
    // If value is negative, make it zero.
    for (unsigned int n_num = 0; n_num < elem->n_nodes(); ++n_num)
    {
        double phi_val = (*phi_vec)(elem->node_id(n_num));
        phi_val = phi_val < 0.0 ? 0.0 : phi_val;
        sf += phi_val * phi[n_num][0];
    }
    d_max_CB = std::max(d_max_CB, std::abs(sf));
    d_min_CB = std::min(d_min_CB, std::abs(sf));
    d_avg_CB += std::abs(sf);
    d_tot_CB += 1;
    return sf;
}

double
CBFinder::maxCB() const
{
    return d_max_CB;
}

void
CBFinder::printStatistics() const
{
    plog << "  max CB used : " << d_max_CB << "\n";
    plog << "  min CB used : " << d_min_CB << "\n";
    plog << "  avg CB used : " << d_avg_CB / static_cast<double>(d_tot_CB) << "\n";
}

void
CBFinder::formHarmonicInterpolant()
{
    d_max_CB = 0.0;
    d_min_CB = 1.0;
    d_avg_CB = 0.0;
    d_tot_CB = 0;
    EquationSystems* vol_eq_sys = d_fe_data_manager->getEquationSystems();
    const MeshBase& vol_mesh = vol_eq_sys->get_mesh();
    EquationSystems* bdry_eq_sys = d_sb_data_manager->getFEMeshPartitioner()->getEquationSystems();

    System& phi_sys = vol_eq_sys->get_system(d_phi_sys_name);
    DofMap& phi_dof_map = phi_sys.get_dof_map();
    phi_dof_map.compute_sparsity(vol_eq_sys->get_mesh());
    System& sf_sys = bdry_eq_sys->get_system(d_sf_name);

    NumericVector<double>* phi_vec = phi_sys.solution.get();
    std::unique_ptr<NumericVector<double>> phi_F_vec(phi_vec->zero_clone());
    NumericVector<double>* sf_vec = sf_sys.solution.get();

    std::unique_ptr<FEBase> fe = FEBase::build(vol_mesh.mesh_dimension(), d_elem_type);
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, vol_mesh.mesh_dimension(), FIFTH);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<std::vector<double>>& phi = fe->get_phi();
    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();

    // Also get face qrule for boundary conditions
    std::unique_ptr<FEBase> fe_face = FEBase::build(vol_mesh.mesh_dimension(), d_elem_type);
    std::unique_ptr<QBase> qrule_face = QBase::build(QGAUSS, vol_mesh.mesh_dimension() - 1, FIFTH);
    fe_face->attach_quadrature_rule(qrule_face.get());
    const std::vector<libMesh::Point>& xyz_face = fe_face->get_xyz();
    const std::vector<double>& JxW_face = fe_face->get_JxW();
    const std::vector<std::vector<double>>& phi_face = fe_face->get_phi();

    // We need a fe base for sf as well
    std::unique_ptr<FEBase> fe_sf =
        FEBase::build(bdry_eq_sys->get_mesh().mesh_dimension(), sf_sys.get_dof_map().variable_type(0));
    std::unique_ptr<QBase> sf_qrule = QBase::build(QGAUSS, bdry_eq_sys->get_mesh().mesh_dimension(), FIFTH);
    fe_sf->attach_quadrature_rule(sf_qrule.get());
    const std::vector<std::vector<double>>& sf_phi = fe_sf->get_phi();
    const std::vector<libMesh::Point>& sf_xyz = fe_sf->get_xyz();

    std::unique_ptr<PetscLinearSolver<double>> solver(new PetscLinearSolver<double>(vol_eq_sys->get_mesh().comm()));
    std::unique_ptr<PetscMatrix<double>> mat(new PetscMatrix<double>(vol_eq_sys->get_mesh().comm()));
    mat->attach_dof_map(phi_dof_map);
    mat->init();

    DenseMatrix<double> M_e;
    DenseVector<double> F_e;
    std::vector<dof_id_type> phi_dofs;

    const MeshBase::const_element_iterator el_begin = vol_eq_sys->get_mesh().local_elements_begin();
    const MeshBase::const_element_iterator el_end = vol_eq_sys->get_mesh().local_elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        Elem* const elem = *el_it;
        phi_dof_map.dof_indices(elem, phi_dofs);
        const auto n_dofs = static_cast<unsigned int>(phi_dofs.size());
        fe->reinit(elem);
        M_e.resize(n_dofs, n_dofs);
        F_e.resize(n_dofs);

        for (unsigned int qp = 0; qp < phi[0].size(); ++qp)
        {
            for (unsigned int i = 0; i < n_dofs; ++i)
            {
                for (unsigned int j = 0; j < n_dofs; ++j)
                {
                    M_e(i, j) += JxW[qp] * (dphi[i][qp] * dphi[j][qp]);
                }
            }
        }

        // Determine if we need to add in contribution from boundary
        for (auto side : elem->side_index_range())
        {
            if (elem->neighbor_ptr(side)) continue;
            const std::unique_ptr<Elem>& side_elem = elem->build_side_ptr(side);
            bool side_is_on_react = true;
            for (unsigned int n_num = 0; n_num < side_elem->n_nodes(); ++n_num)
            {
                auto node_id_iter = d_node_id_map.find(side_elem->node_id(n_num));
                if (node_id_iter == d_node_id_map.end())
                {
                    side_is_on_react = false;
                }
            }
            fe_face->reinit(elem, side);
            fe_sf->reinit(side_elem.get());
            const double penalty = 1.0e10;

            for (unsigned int qp = 0; qp < JxW_face.size(); ++qp)
            {
                double sf_val = 0.0;
                for (unsigned int i = 0; i < side_elem->n_nodes(); ++i)
                {
                    auto node_id_iter = d_node_id_map.find(side_elem->node_id(i));
                    if (node_id_iter != d_node_id_map.end())
                    {
                        sf_val += sf_phi[i][qp] * (*sf_vec)((*node_id_iter).second);
                    }
                }
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        M_e(i, j) += JxW_face[qp] * penalty * (phi_face[i][qp] * phi_face[j][qp]);
                    }
                    F_e(i) += JxW_face[qp] * penalty * phi_face[i][qp] * sf_val;
                }
            }
        }
        phi_dof_map.constrain_element_matrix_and_vector(M_e, F_e, phi_dofs);
        mat->add_matrix(M_e, phi_dofs);
        phi_F_vec->add_vector(F_e, phi_dofs);
    }

    MatSetOption(mat->mat(), MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    mat->close();

    solver->reuse_preconditioner(true);
    solver->set_preconditioner_type(JACOBI_PRECOND);
    solver->set_solver_type(MINRES);
    solver->init();

    solver->solve(*mat, *phi_vec, *phi_F_vec, 1.0e-12, 1000);
    KSPConvergedReason reason;
    int ierr = KSPGetConvergedReason(solver->ksp(), &reason);
    IBTK_CHKERRQ(ierr);
    reportPETScKSPConvergedReason("CBFinder", reason, plog);

    phi_vec->close();
    phi_sys.update();
    return;
}

void
CBFinder::reportPETScKSPConvergedReason(const std::string& object_name,
                                        const KSPConvergedReason& reason,
                                        std::ostream& os)
{
    switch (static_cast<int>(reason))
    {
    case KSP_CONVERGED_RTOL:
        os << object_name
           << ": converged: |Ax-b| <= rtol*|b| --- residual norm is less than specified relative tolerance.\n";
        break;
    case KSP_CONVERGED_ATOL:
        os << object_name
           << ": converged: |Ax-b| <= atol --- residual norm is less than specified absolute tolerance.\n";
        break;
    case KSP_CONVERGED_ITS:
        os << object_name << ": converged: maximum number of iterations reached.\n";
        break;
    case KSP_CONVERGED_STEP_LENGTH:
        os << object_name << ": converged: step size less than specified tolerance.\n";
        break;
    case KSP_DIVERGED_NULL:
        os << object_name << ": diverged: null.\n";
        break;
    case KSP_DIVERGED_ITS:
        os << object_name
           << ": diverged: reached maximum number of iterations before any convergence criteria were satisfied.\n";
        break;
    case KSP_DIVERGED_DTOL:
        os << object_name
           << ": diverged: |Ax-b| >= dtol*|b| --- residual is greater than specified divergence tolerance.\n";
        break;
    case KSP_DIVERGED_BREAKDOWN:
        os << object_name << ": diverged: breakdown in the Krylov method.\n";
        break;
    case KSP_DIVERGED_BREAKDOWN_BICG:
        os << object_name << ": diverged: breakdown in the bi-congugate gradient method.\n";
        break;
    case KSP_DIVERGED_NONSYMMETRIC:
        os << object_name
           << ": diverged: it appears the operator or preconditioner is not symmetric, but this Krylov method (KSPCG, "
              "KSPMINRES, KSPCR) requires symmetry\n";
        break;
    case KSP_DIVERGED_INDEFINITE_PC:
        os << object_name
           << ": diverged: it appears the preconditioner is indefinite (has both positive and negative eigenvalues), "
              "but this Krylov method (KSPCG) requires it to be positive definite.\n";
        break;
    case KSP_CONVERGED_ITERATING:
        os << object_name << ": iterating: KSPSolve() is still running.\n";
        break;
    default:
        os << object_name << ": unknown completion code " << static_cast<int>(reason) << " reported.\n";
        break;
    }
    return;
} // reportPETScKSPConvergedReason
} // namespace IBAMR
