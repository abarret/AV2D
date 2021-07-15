// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2020 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/app_namespaces.h" // IWYU pragma: keep
#include "CCAD/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"

#include "RBFOneSidedReconstructions.h"
#include "SAMRAIVectorReal.h"

#include <Eigen/Dense>

#include <utility>

RBFOneSidedReconstructions::RBFOneSidedReconstructions(std::string object_name,
                                                       Reconstruct::RBFPolyOrder rbf_order,
                                                       int stencil_size)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_rbf_order(rbf_order),
      d_rbf_stencil_size(stencil_size),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), 2);
    return;
} // RBFOneSidedReconstructions

RBFOneSidedReconstructions::~RBFOneSidedReconstructions()
{
    deallocateOperatorState();
    return;
} // ~RBFOneSidedReconstructions

void
RBFOneSidedReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_vol_idx > 0);
    TBOX_ASSERT(d_new_vol_idx > 0);
#endif

    // TODO: What kind of physical boundary conditions should we use for
    // advection?
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps(2);
    ghost_cell_comps[0] =
        ITC(d_Q_scr_idx, Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
    ghost_cell_comps[1] = ITC(d_cur_ls_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    HierarchyGhostCellInterpolation hier_ghost_cells;
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(d_current_time);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(d_Q_scr_idx);
            Pointer<CellData<NDIM, double>> vol_cur_data = patch->getPatchData(d_cur_vol_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);
            Pointer<CellData<NDIM, double>> vol_new_data = patch->getPatchData(d_new_vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);
            Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(d_new_ls_idx);

            Q_new_data->fillAll(0.0);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_new_data)(idx) > 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    // Find Node index closest to point. Note that this location
                    // corresponds to the Cell index that is the "lower left" of the box
                    // centered at nodal index.
                    CellIndex<NDIM> idx_ll;
                    VectorNd x_ll;
                    for (int d = 0; d < NDIM; ++d) x_ll[d] = std::round(x_loc[d]) - 0.5;
                    for (int d = 0; d < NDIM; ++d) idx_ll(d) = static_cast<int>(x_ll[d]);
                    // Check if we can use bi-linear interpolation, i.e. check if all
                    // neighboring cells are "full" cells
                    bool use_bilinear = true;
                    for (int x = -1; x <= 2; ++x)
                        for (int y = -1; y <= 2; ++y)
                            use_bilinear = use_bilinear && (*vol_cur_data)(idx_ll + IntVector<NDIM>(x, y)) == 1.0;
                    std::vector<double> temp_dx = { 1.0, 1.0 };

                    // Check if we can use z-spline
                    if (use_bilinear)
                        (*Q_new_data)(idx) =
                            Reconstruct::bilinearReconstruction(x_loc, x_ll, idx_ll, *Q_cur_data, temp_dx.data());
                    else
                        (*Q_new_data)(idx) = oneSidedRBFReconstruct(
                            x_loc, idx, patch, *Q_cur_data, *ls_data, *vol_cur_data, *vol_new_data, *ls_new_data);
                }
                else
                {
                    (*Q_new_data)(idx) = 0.0;
                }
            }
        }
    }
}

void
RBFOneSidedReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                  double current_time,
                                                  double new_time)
{
    AdvectiveReconstructionOperator::allocateOperatorState(hierarchy, current_time, new_time);
    d_hierarchy = hierarchy;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_Q_scr_idx)) level->allocatePatchData(d_Q_scr_idx);
    }
    d_is_allocated = true;
}

void
RBFOneSidedReconstructions::deallocateOperatorState()
{
    AdvectiveReconstructionOperator::deallocateOperatorState();

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_Q_scr_idx)) level->deallocatePatchData(d_Q_scr_idx);
    }
}

double
RBFOneSidedReconstructions::oneSidedRBFReconstruct(VectorNd x_loc,
                                                   const hier::Index<NDIM>& idx,
                                                   Pointer<Patch<NDIM>> patch,
                                                   const CellData<NDIM, double>& Q_data,
                                                   const NodeData<NDIM, double>& ls_data,
                                                   const CellData<NDIM, double>& vol_data,
                                                   const CellData<NDIM, double>& vol_new_data,
                                                   const NodeData<NDIM, double>& ls_new_data)
{
    std::vector<VectorNd> X_vals;
    std::vector<CellIndex<NDIM>> final_idxs;
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    // We are on a cut cell. We need to interpolate to cell center.
    // Use a flooding algorithm where we only select indices that are on the same
    // side of the boundary as the cell centroid.
    std::vector<CellIndex<NDIM>> new_idxs = { idx };
    VectorNd normal_vec, midpt;
    double sgn = 0.0;
    if (vol_new_data(idx) < 1.0 && vol_new_data(idx) > 0.0)
    {
        // Find tangent line
        std::vector<VectorNd> intersect_pts;
        double ls_ll = ls_new_data(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerLeft));
        double ls_lr = ls_new_data(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerRight));
        double ls_ur = ls_new_data(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperRight));
        double ls_ul = ls_new_data(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperLeft));
        if (ls_ll * ls_lr < 0.0)
        {
            intersect_pts.push_back(
                CCAD::midpoint_value(VectorNd(idx(0), idx(1)), ls_ll, VectorNd(idx(0) + 1.0, idx(1)), ls_lr));
        }
        if (ls_lr * ls_ur < 0.0)
        {
            intersect_pts.push_back(CCAD::midpoint_value(
                VectorNd(idx(0) + 1.0, idx(1)), ls_lr, VectorNd(idx(0) + 1.0, idx(1) + 1.0), ls_ur));
        }
        if (ls_ur * ls_ul < 0.0)
        {
            intersect_pts.push_back(CCAD::midpoint_value(
                VectorNd(idx(0) + 1.0, idx(1) + 1.0), ls_ur, VectorNd(idx(0), idx(1) + 1.0), ls_ul));
        }
        if (ls_ul * ls_ll < 0.0)
        {
            intersect_pts.push_back(
                CCAD::midpoint_value(VectorNd(idx(0), idx(1) + 1.0), ls_ul, VectorNd(idx(0), idx(1)), ls_ll));
        }
#if (NDIM == 2)
        IBTK::Vector3d tangent_vec(
            intersect_pts[0](0) - intersect_pts[1](0), intersect_pts[0](1) - intersect_pts[1](1), 0.0);
        IBTK::Vector3d normal_vec_3d = tangent_vec.cross(IBTK::Vector3d::UnitZ());
        for (int d = 0; d < NDIM; ++d) normal_vec[d] = normal_vec_3d[d];
        // To find proper sign of normal, we just need to check one node. Choose the
        // upper left.
        // TODO: We should probably choose the largest value to avoid finite
        // precision issues.
        VectorNd idx_ul = VectorNd(idx(0), idx(1) + 1.0);
        midpt = 0.5 * (intersect_pts[0] + intersect_pts[1]);
        if (normal_vec.dot(idx_ul - midpt) * ls_ul > 0.0)
        {
            // Normal vec is pointing the wrong way.
            normal_vec *= -1;
        }
#endif
        normal_vec.normalize();
        sgn = normal_vec.dot(find_cell_centroid(idx, ls_new_data) + 0.5 * std::min(dx[0], dx[1]) * normal_vec - midpt);
    }
    else
    {
        for (int d = 0; d < NDIM; ++d) normal_vec[d] = std::numeric_limits<double>::quiet_NaN();
    }
    unsigned int i = 0;
    while (final_idxs.size() < 5)
    {
#ifndef NDEBUG
        TBOX_ASSERT(i < new_idxs.size());
#endif
        CellIndex<NDIM> new_idx = new_idxs[i];
        // Add new_idx to idx_map
        if (useIdx(new_idx, ls_data, vol_data, normal_vec, midpt, sgn))
        {
            final_idxs.push_back(new_idx);
            X_vals.push_back(find_cell_centroid(new_idx, ls_data));
        }
        // Add Neighboring points to new_idxs
        IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
        CellIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
        CellIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
        appendIndexToList(new_idxs, idx_l, new_idx, idx, vol_data);
        appendIndexToList(new_idxs, idx_r, new_idx, idx, vol_data);
        appendIndexToList(new_idxs, idx_u, new_idx, idx, vol_data);
        appendIndexToList(new_idxs, idx_b, new_idx, idx, vol_data);
        ++i;
    }
    // We have all the points. Now find the PHS.
    const int m = X_vals.size();
    IBTK::MatrixXd A(MatrixXd::Zero(m, m));
    IBTK::MatrixXd B(MatrixXd::Zero(m, NDIM + 1));
    IBTK::VectorXd U(VectorXd::Zero(m + NDIM + 1));
    for (size_t i = 0; i < X_vals.size(); ++i)
    {
        for (size_t j = 0; j < X_vals.size(); ++j)
        {
            const VectorNd X = X_vals[i] - X_vals[j];
            const double phi = Reconstruct::rbf(X.norm());
            A(i, j) = phi;
        }
        B(i, 0) = 1.0;
        for (int d = 0; d < NDIM; ++d) B(i, d + 1) = X_vals[i](d);
        U(i) = Q_data(final_idxs[i]);
    }

    IBTK::MatrixXd final_mat(MatrixXd::Zero(m + NDIM + 1, m + NDIM + 1));
    final_mat.block(0, 0, m, m) = A;
    final_mat.block(0, m, m, NDIM + 1) = B;
    final_mat.block(m, 0, NDIM + 1, m) = B.transpose();
    IBTK::VectorXd x1 = FullPivHouseholderQR<IBTK::MatrixXd>(final_mat).solve(U);
    IBTK::VectorXd rbf_coefs = x1.block(0, 0, m, 1);
    IBTK::VectorXd poly_coefs = x1.block(m, 0, NDIM + 1, 1);
    IBTK::VectorXd poly_vec = IBTK::VectorXd::Ones(NDIM + 1);
    for (int d = 0; d < NDIM; ++d) poly_vec(d + 1) = x_loc(d);
    double val = 0.0;
    for (size_t i = 0; i < X_vals.size(); ++i) val += rbf_coefs[i] * Reconstruct::rbf((X_vals[i] - x_loc).norm());
    val += poly_coefs.dot(poly_vec);
    return val;
}

bool
RBFOneSidedReconstructions::useIdx(const hier::Index<NDIM>& idx,
                                   const NodeData<NDIM, double>& ls_data,
                                   const CellData<NDIM, double>& vol_data,
                                   const VectorNd& normal_vec,
                                   const VectorNd& midpt,
                                   double sgn)
{
    if (vol_data(idx) == 0.0) return false;
    // This point is inside the domain. If normal_vec contains NaNs, then we
    // return true
    for (int d = 0; d < NDIM; ++d)
    {
        if (normal_vec(d) != normal_vec(d)) return true;
    }
    // We are on a cut cell. Find the cell centroid, then make sure the dot
    // product is the same sign as sgn.
    VectorNd cell_centroid = find_cell_centroid(idx, ls_data);
    if (normal_vec.dot(cell_centroid - midpt) * sgn > 0.0) return true;
    // If at this point, we haven't returned true, then return false
    return false;
}

bool
RBFOneSidedReconstructions::appendIndexToList(std::vector<CellIndex<NDIM>>& new_idxs,
                                              const CellIndex<NDIM>& idx,
                                              const CellIndex<NDIM>& orig_idx,
                                              const CellIndex<NDIM>& base_idx,
                                              const CellData<NDIM, double>& vol_data)
{
    // If index is not inside domain, return false
    if (vol_data(idx) == 0.0) return false;
    // If index has already been used, return false
    if (std::find(new_idxs.begin(), new_idxs.end(), idx) != new_idxs.end()) return false;
    new_idxs.push_back(idx);
    return true;
}

//////////////////////////////////////////////////////////////////////////////
