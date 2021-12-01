#include "ADS/app_namespaces.h"
#include "ADS/reconstructions.h"

#include "RBFReconstructCacheOS.h"

namespace LS
{
RBFReconstructCacheOS::RBFReconstructCacheOS(const unsigned int stencil_width) : ReconstructCache(stencil_width)
{
    // intentionally blank
}

RBFReconstructCacheOS::RBFReconstructCacheOS(int ls_idx,
                                             int vol_idx,
                                             Pointer<PatchHierarchy<NDIM>> hierarchy,
                                             bool use_centroids)
    : ReconstructCache(ls_idx, vol_idx, hierarchy, use_centroids)
{
    // intentionally blank
}

void
RBFReconstructCacheOS::cacheData()
{
    // intentionally blank
    return;
}

void
RBFReconstructCacheOS::clearCache()
{
    ReconstructCache::clearCache();
    d_reconstruct_idxs_map_vec.clear();
}

double
RBFReconstructCacheOS::reconstructOnIndex(VectorNd x_loc,
                                          const hier::Index<NDIM>& idx,
                                          const CellData<NDIM, double>& Q_data,
                                          Pointer<Patch<NDIM>> patch)
{
    if (d_update_weights) cacheData();
    const int ln = patch->getPatchLevelNumber();
    IndexList pi_pair(patch, idx);
    Box<NDIM> box(idx, idx);
    box.grow(d_stencil_size);
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
    Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
#ifndef NDEBUG
    TBOX_ASSERT(ls_data->getGhostBox().contains(box));
    TBOX_ASSERT(Q_data.getGhostBox().contains(box));
    TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();
    for (int d = 0; d < NDIM; ++d) x_loc[d] = idx_low(d) + (x_loc[d] - xlow[d]) / dx[d];

    std::vector<double> Q_vals;
    std::vector<VectorNd> X_vals;

    if (d_reconstruct_idxs_map_vec.size() == 0 ||
        d_reconstruct_idxs_map_vec[ln].find(pi_pair) == d_reconstruct_idxs_map_vec[ln].end())
    {
        // We need to reconstruct. Let's cache the data
        cacheData(x_loc, idx, Q_data, patch);
    }
    const std::vector<hier::Index<NDIM>>& idx_vec = d_reconstruct_idxs_map_vec[ln][pi_pair];

    for (const auto& idx_c : idx_vec)
    {
        VectorNd x_cent_c;
        if (d_use_centroids)
        {
            x_cent_c = ADS::find_cell_centroid(idx_c, *ls_data);
        }
        else
        {
            for (int d = 0; d < NDIM; ++d) x_cent_c[d] = static_cast<double>(idx_c[d]) + 0.5;
        }
        Q_vals.push_back(Q_data(idx_c));
        X_vals.push_back(x_cent_c);
    }

    const int m = Q_vals.size();
    IBTK::VectorXd U(VectorXd::Zero(m + NDIM + 1));
    for (size_t i = 0; i < Q_vals.size(); ++i) U(i) = Q_vals[i];

    IBTK::VectorXd x1 = d_qr_matrix_vec[ln][patch->getPatchNumber()][IndexList(patch, idx)].solve(U);
    IBTK::VectorXd rbf_coefs = x1.block(0, 0, m, 1);
    IBTK::VectorXd poly_coefs = x1.block(m, 0, NDIM + 1, 1);
    IBTK::VectorXd poly_vec = IBTK::VectorXd::Ones(NDIM + 1);
    for (int d = 0; d < NDIM; ++d) poly_vec(d + 1) = x_loc(d);
    double val = 0.0;
    for (size_t i = 0; i < X_vals.size(); ++i)
    {
        val += rbf_coefs[i] * Reconstruct::rbf((X_vals[i] - x_loc).norm());
    }
    val += poly_coefs.dot(poly_vec);
    return val;
}

void
RBFReconstructCacheOS::cacheData(VectorNd x_loc,
                                 const hier::Index<NDIM>& idx,
                                 const CellData<NDIM, double>& Q_data,
                                 Pointer<Patch<NDIM>> patch)
{
    const int ln = patch->getPatchLevelNumber();
    const int finest_ln = d_hierarchy->getFinestLevelNumber();

    // allocate matrix data
    if (d_qr_matrix_vec.size() == 0) d_qr_matrix_vec.resize(finest_ln + 1);
    if (d_reconstruct_idxs_map_vec.size() == 0) d_reconstruct_idxs_map_vec.resize(finest_ln + 1);

    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
    std::vector<std::map<IndexList, FullPivHouseholderQR<IBTK::MatrixXd>>>& qr_map_vec = d_qr_matrix_vec[ln];
    if (qr_map_vec.size() == 0) qr_map_vec.resize(level->getNumberOfPatches());
    std::map<IndexList, std::vector<hier::Index<NDIM>>>& idx_map = d_reconstruct_idxs_map_vec[ln];
    int local_patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        Pointer<Patch<NDIM>> cur_patch = level->getPatch(p());
        if (patch != cur_patch) continue;
        IndexList p_idx = IndexList(patch, idx);
        std::map<IndexList, FullPivHouseholderQR<IBTK::MatrixXd>>& qr_map = qr_map_vec[local_patch_num];

        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
        Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

        if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
        {
            // We are on a cut cell. We need to interpolate to cell center
            VectorNd x_loc;
            for (int d = 0; d < NDIM; ++d) x_loc(d) = static_cast<double>(idx(d)) + 0.5;
            Box<NDIM> box(idx, idx);
            box.grow(d_stencil_size);
#ifndef NDEBUG
            TBOX_ASSERT(ls_data->getGhostBox().contains(box));
            TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif
            // Find tangent line
            std::vector<VectorNd> intersect_pts;
            double ls_ll = (*ls_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerLeft));
            double ls_lr = (*ls_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerRight));
            double ls_ur = (*ls_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperRight));
            double ls_ul = (*ls_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperLeft));
            if (ls_ll * ls_lr < 0.0)
            {
                intersect_pts.push_back(
                    ADS::midpoint_value(VectorNd(idx(0), idx(1)), ls_ll, VectorNd(idx(0) + 1.0, idx(1)), ls_lr));
            }
            if (ls_lr * ls_ur < 0.0)
            {
                intersect_pts.push_back(ADS::midpoint_value(
                    VectorNd(idx(0) + 1.0, idx(1)), ls_lr, VectorNd(idx(0) + 1.0, idx(1) + 1.0), ls_ur));
            }
            if (ls_ur * ls_ul < 0.0)
            {
                intersect_pts.push_back(ADS::midpoint_value(
                    VectorNd(idx(0) + 1.0, idx(1) + 1.0), ls_ur, VectorNd(idx(0), idx(1) + 1.0), ls_ul));
            }
            if (ls_ul * ls_ll < 0.0)
            {
                intersect_pts.push_back(
                    ADS::midpoint_value(VectorNd(idx(0), idx(1) + 1.0), ls_ul, VectorNd(idx(0), idx(1)), ls_ll));
            }
            IBTK::Vector3d tangent_vec(
                intersect_pts[0](0) - intersect_pts[1](0), intersect_pts[0](1) - intersect_pts[1](1), 0.0);
            IBTK::Vector3d normal_vec = tangent_vec.cross(IBTK::Vector3d::UnitZ());
            VectorNd normal_vec_2d(normal_vec(0), normal_vec(1));
            // We know that the current idx is on the "interior" of the structure.
            // Use this to determine the sign
            VectorNd cur_vec = ADS::find_cell_centroid(idx, *ls_data) - 0.5 * (intersect_pts[0] + intersect_pts[1]);
            double sgn = normal_vec_2d.dot(cur_vec);
            // Now we need to check against this sign to determine if we can use the point.

            const CellIndex<NDIM>& idx_low = patch->getBox().lower();
            std::vector<VectorNd> X_vals;

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx_c = ci();
                if ((*vol_data)(idx_c) > 0.0)
                {
                    // Use this point to calculate least squares reconstruction.
                    VectorNd x_cent_c;
                    if (d_use_centroids)
                    {
                        x_cent_c = ADS::find_cell_centroid(idx_c, *ls_data);
                    }
                    else
                    {
                        for (int d = 0; d < NDIM; ++d) x_cent_c[d] = static_cast<double>(idx_c[d]) + 0.5;
                    }
                    VectorNd dist_to_plane =
                        ADS::find_cell_centroid(idx_c, *ls_data) - 0.5 * (intersect_pts[0] + intersect_pts[1]);

                    // Check to see if it's the same sign as sgn
                    if ((normal_vec_2d.dot(dist_to_plane) * sgn) > 0.0)
                    {
                        X_vals.push_back(x_cent_c);
                        idx_map[p_idx].push_back(idx_c);
                    }
                }
            }
            const int m = X_vals.size();
            IBTK::MatrixXd A(MatrixXd::Zero(m, m));
            IBTK::MatrixXd B(MatrixXd::Zero(m, NDIM + 1));
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
            }

            IBTK::MatrixXd final_mat(MatrixXd::Zero(m + NDIM + 1, m + NDIM + 1));
            final_mat.block(0, 0, m, m) = A;
            final_mat.block(0, m, m, NDIM + 1) = B;
            final_mat.block(m, 0, NDIM + 1, m) = B.transpose();

#ifndef NDEBUG
            if (qr_map.find(p_idx) == qr_map.end())
                qr_map[p_idx] = FullPivHouseholderQR<IBTK::MatrixXd>(final_mat);
            else
                TBOX_WARNING("Already had a QR decomposition in place");
#else
            qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(final_mat);
#endif
        }
        else
        {
            // We must be on a full cell (or something has gone terribly wrong and we are on an empty cell)
            // We can be on a full cell if the reconstruction point has two intersections on the same side of the cell.
            // We'll use flooding to do the reconstruction since finding "normals" doesn't make sense here
            std::vector<CellIndex<NDIM>> new_idxs = { idx };
            std::vector<VectorNd> X_vals;
            unsigned int i = 0;
            while (idx_map[p_idx].size() < 5)
            {
#ifndef NDEBUG
                TBOX_ASSERT(i < new_idxs.size());
#endif
                CellIndex<NDIM> new_idx = new_idxs[i];
                // Add new_idx to idx_map
                if ((*vol_data)(new_idx) > 0.0)
                {
                    idx_map[p_idx].push_back(new_idx);
                    VectorNd x_cent;
                    if (d_use_centroids)
                        x_cent = ADS::find_cell_centroid(new_idx, *ls_data);
                    else
                        for (int d = 0; d < NDIM; ++d) x_cent[d] = static_cast<double>(new_idx(d)) + 0.5;
                    X_vals.push_back(x_cent);
                }
                // Add Neighboring points to new_idxs
                IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
                CellIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
                CellIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
                if ((*vol_data)(idx_l) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_l) == new_idxs.end()))
                    new_idxs.push_back(idx_l);
                if ((*vol_data)(idx_r) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_r) == new_idxs.end()))
                    new_idxs.push_back(idx_r);
                if ((*vol_data)(idx_u) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_u) == new_idxs.end()))
                    new_idxs.push_back(idx_u);
                if ((*vol_data)(idx_b) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_b) == new_idxs.end()))
                    new_idxs.push_back(idx_b);
                ++i;
            }
            const int m = X_vals.size();
            IBTK::MatrixXd A(MatrixXd::Zero(m, m));
            IBTK::MatrixXd B(MatrixXd::Zero(m, NDIM + 1));
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
            }

            IBTK::MatrixXd final_mat(MatrixXd::Zero(m + NDIM + 1, m + NDIM + 1));
            final_mat.block(0, 0, m, m) = A;
            final_mat.block(0, m, m, NDIM + 1) = B;
            final_mat.block(m, 0, NDIM + 1, m) = B.transpose();

#ifndef NDEBUG
            if (qr_map.find(p_idx) == qr_map.end())
                qr_map[p_idx] = FullPivHouseholderQR<IBTK::MatrixXd>(final_mat);
            else
                TBOX_WARNING("Already had a QR decomposition in place");
#else
            qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(final_mat);
#endif
        }
    }
}
} // namespace LS
