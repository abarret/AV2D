// Filename: CirculationModel.cpp
// Created on 20 Aug 2007 by Boyce Griffith

#include <ibamr/config.h>

#include "ibamr/app_namespaces.h"

#include "CirculationModel.h"
#include <tbox/RestartManager.h>
#include <tbox/SAMRAI_MPI.h>
#include <tbox/Utilities.h>

#include <Eigen/Dense>

#include <CartesianGridGeometry.h>
#include <CartesianPatchGeometry.h>
#include <PatchLevel.h>
#include <SAMRAI_config.h>
#include <SideData.h>

#include <cassert>
using namespace Eigen;

/////////////////////////////// NAMESPACE ////////////////////////////////////

/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Name of output file.
static string DATA_FILE_NAME = "bc_data.txt";

// Conversion factors.
static const double flconv = 1 / 0.06;  // l/min ===> ml/sec
static const double prconv = 1333.2239; // mmHg  ===> dyne/cm^2

// Parameters for upstream LA/LV Model
static double R_LVOT = 0.0150;               // LVOT resistance (mmHg ml^-1 s)
static double R_MV = 0.0050;                 // mitral valve resistance (mmHg ml^-1 s)
static double T_per = 0.8512;                // period length (time) for cardiac cycle (s);
static double Q_vein = 6.2;                  // pulmonary veinous flow rate into LA (L/min);
static double Q_vein_conv = Q_vein * flconv; // pulmonary veinous flow rate into LA (ml/sec);
static double t_sys = 0.2;                   // sec
static double m1_LA = 1.32;
static double m2_LA = 13.1;
static double m1_LV = 2.4039;
static double m2_LV = 20.9518;
static double tau1_LA = 0.1150 * T_per;  // sec
static double tau2_LA = 0.1882 * T_per;  // sec
static double t_shift_LA = 0.85 * T_per; // sec
static double tau1_LV = 0.0887 * T_per;  // sec
static double tau2_LV = 0.4461 * T_per;  // sec
static double E_min_LA = 0.08;           // elastance (mmHg ml^-1)
static double E_max_LA = 0.17;           // elastance (mmHg ml^-1)
static double E_min_LV = 0.0265;         // elastance (mmHg ml^-1)
static double E_max_LV = 0.3631;         // elastance (mmHg ml^-1)
static double k_denom_LA = 0.558482;
static double k_denom_LV = 0.9588;
static double k_LA = (E_max_LA - E_min_LA) / k_denom_LA;
static double k_LV = (E_max_LV - E_min_LV) / k_denom_LV;

// Three-element windkessel model fit to Larry Scotten's experimental data for porcine BHV:
static double R_P = 0.9046; // peripheral resistance (mmHg ml^-1 s)
static double R_C = 0.0420; // characteristic resistance (mmHg ml^-1 s)
static double C = 1.9504;   // total arterial compliance (ml mmHg^-1)

// Time required to "ramp up" the pressure in the LA/LV and Windkessel models.
//
// The values of P_LV_init and P_Wk_init are essentially the initial conditions for the
// upstream/downstream flow circuits, for which the pressure is a state variable.
//
// We *actually* start from zero pressure, however, to allow the leaflets the opportunity
// to gradually "load up".
static double P_LA_init = 10.38;   // mmHg
static double P_LV_init = 10.2892; // mmHg
static double P_Wk_init = 76.6591; // mmHg
static double t_ramp = 0.15;       // sec

// Backward Euler update for windkessel model.
inline void
windkessel_be_update(double& P_Wk, double& P_Ao, const double& Q_Ao, const double& t, const double& dt)
{
    if (t < t_ramp)
    {
        P_Wk = (t + dt) * P_Wk_init / t_ramp;
    }
    else
    {
        P_Wk = (C * P_Wk + Q_Ao * dt) * R_P / (C * R_P + dt);
    }
    P_Ao = P_Wk + R_C * Q_Ao;
    return;
} // windkessel_be_update

// Backward Euler update for the LA/LV model
inline void
Upstream_be_update(double& P_LA,
                   double& P_LV,
                   double& P_LVOT,
                   double& C_LA_old,
                   double& C_LA_new,
                   double& C_LV_old,
                   double& C_LV_new,
                   const double& Q_LV,
                   const double& t,
                   const double& dt)
{
    if (t < t_ramp)
    {
        P_LA = (t + dt) * P_LA_init / t_ramp;
        P_LV = (t + dt) * P_LV_init / t_ramp;
    }
    else
    {
        double S = (P_LV >= P_LA) ? 0.0 : 1.0; // valve closed?
        double denom = S * dt * (C_LA_new + C_LV_new) + R_MV * C_LV_new * C_LA_new;
        double P_LA_tmp = -(S * dt * dt * (Q_LV - Q_vein_conv) - S * dt * (C_LA_old * P_LA + C_LV_old * P_LV) -
                            R_MV * C_LV_new * (dt * Q_vein_conv + C_LA_old * P_LA));
        double P_LV_tmp = -(S * dt * dt * (Q_LV - Q_vein_conv) - S * dt * (C_LA_old * P_LA + C_LV_old * P_LV) +
                            R_MV * C_LA_new * (dt * Q_LV - C_LV_old * P_LV));
        P_LA = P_LA_tmp / denom;
        P_LV = P_LV_tmp / denom;
    }
    P_LVOT = P_LV - R_LVOT * Q_LV;
    return;
} // Upstream_be_update

// Time varying compliances for the LA/LV model
inline void
Upstream_compliances(double& C,
                     const double E_min,
                     const double tau1,
                     const double tau2,
                     const double m1,
                     const double m2,
                     const double k,
                     const double& t,
                     const double& dt)
{
    const double t_mod = fmod(t + T_per - t_sys, T_per);
    double g1 = pow(t_mod / tau1, m1);
    double g2 = pow(t_mod / tau2, m2);
    double E = k * (g1 / (1.0 + g1)) * (1.0 / (1.0 + g2)) + E_min;
    C = 1.0 / E;
    return;
} // Upstream_compliances
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

CirculationModel::CirculationModel(const string& object_name, Pointer<Database> input_db, bool register_for_restart)
    : d_object_name(object_name),
      d_registered_for_restart(register_for_restart),
      d_time(0.0),
      d_nsrc(2),
      d_qsrc(d_nsrc, 0.0),
      d_psrc(d_nsrc, 0.0),
      d_rsrc(d_nsrc, 0.0),
      d_posn(d_nsrc),
      d_srcname(d_nsrc),
      d_P_Wk(0.0),
      d_P_LA(0.0),
      d_P_LV(0.0),
      d_bdry_interface_level_number(numeric_limits<int>::max())
{
#if !defined(NDEBUG)
    assert(!object_name.empty());
#endif
    if (d_registered_for_restart)
    {
        RestartManager::getManager()->registerRestartItem(d_object_name, this);
    }
    d_posn[0](0) = -1.541837105;
    d_posn[0](1) = -3.84;
    // d_posn[0](2) = -0.214172;
    d_posn[1](0) = 0.00039225;
    d_posn[1](1) = 3.84;
    // d_posn[1](2) = 0.999614;
    if (input_db)
    {
        DATA_FILE_NAME = input_db->getStringWithDefault("DATA_FILE_NAME", DATA_FILE_NAME);
        d_bdry_interface_level_number =
            input_db->getIntegerWithDefault("bdry_interface_level_number", d_bdry_interface_level_number);
        R_LVOT = input_db->getDoubleWithDefault("R_LVOT", R_LVOT); // LVOT resistance (mmHg ml^-1 s)
        R_MV = input_db->getDoubleWithDefault("R_MV", R_MV);       // mitral valve resistance (mmHg ml^-1 s)
        T_per = input_db->getDoubleWithDefault("T_per", T_per);    // period length (time) for cardiac cycle (s);
        Q_vein = input_db->getDoubleWithDefault("Q_vein", Q_vein); // pulmonary veinous flow rate into LA (L/min);
        Q_vein_conv = Q_vein * flconv;                             // pulmonary veinous flow rate into LA (ml/sec);
        t_sys = input_db->getDoubleWithDefault("t_sys", t_sys);    // time to begin systole (s);
        m1_LA = input_db->getDoubleWithDefault("m1_LA", m1_LA);
        m2_LA = input_db->getDoubleWithDefault("m2_LA", m2_LA);
        m1_LV = input_db->getDoubleWithDefault("m1_LV", m1_LV);
        m2_LV = input_db->getDoubleWithDefault("m2_LV", m2_LV);
        tau1_LA = input_db->getDoubleWithDefault("tau1_LA", tau1_LA);          // sec
        tau2_LA = input_db->getDoubleWithDefault("tau2_LA", tau2_LA);          // sec
        t_shift_LA = input_db->getDoubleWithDefault("t_shift_LA", t_shift_LA); // time to shift pump in LA (s);
        tau1_LV = input_db->getDoubleWithDefault("tau1_LV", tau1_LV);          // sec
        tau2_LV = input_db->getDoubleWithDefault("tau2_LV", tau2_LV);          // sec
        E_min_LA = input_db->getDoubleWithDefault("E_min_LA", E_min_LA);       // elastance (mmHg ml^-1)
        E_max_LA = input_db->getDoubleWithDefault("E_max_LA", E_max_LA);       // elastance (mmHg ml^-1)
        E_min_LV = input_db->getDoubleWithDefault("E_min_LV", E_min_LV);       // elastance (mmHg ml^-1)
        E_max_LV = input_db->getDoubleWithDefault("E_max_LV", E_max_LV);       // elastance (mmHg ml^-1)
        k_denom_LA = input_db->getDoubleWithDefault("k_denom_LA", k_denom_LA);
        k_denom_LV = input_db->getDoubleWithDefault("k_denom_LV", k_denom_LV);
        k_LA = (E_max_LA - E_min_LA) / k_denom_LA;
        k_LV = (E_max_LV - E_min_LV) / k_denom_LV;
        R_P = input_db->getDoubleWithDefault("R_P", R_P);                   // peripheral resistance (mmHg ml^-1 s)
        R_C = input_db->getDoubleWithDefault("R_C", R_C);                   // characteristic resistance (mmHg ml^-1 s)
        C = input_db->getDoubleWithDefault("C", C);                         // total arterial compliance (ml mmHg^-1)
        P_LA_init = input_db->getDoubleWithDefault("P_LA_init", P_LA_init); // mmHg
        P_LV_init = input_db->getDoubleWithDefault("P_LV_init", P_LV_init); // mmHg
        P_Wk_init = input_db->getDoubleWithDefault("P_Wk_init", P_Wk_init); // mmHg
        t_ramp = input_db->getDoubleWithDefault("t_ramp", t_ramp);          // sec
        d_vol_conversion_fac = input_db->getDouble("vol_conversion_fac");
    }

    // Initialize object with data read from the input and restart databases.
    const bool from_restart = RestartManager::getManager()->isFromRestart();
    if (from_restart)
    {
        getFromRestart();
    }
    else
    {
        // Set the initial values for the source/sink data.
        if (input_db)
        {
            d_rsrc[0] = input_db->getDoubleWithDefault("radius_lv", d_rsrc[0]);
            d_rsrc[1] = input_db->getDoubleWithDefault("radius_ao", d_rsrc[1]);
        }

        //   nsrcs = the number of sources in the model:
        //           (1) left ventricle
        //           (2) aorta (normally thought of as a sink)
        d_srcname[0] = "left ventricle    ";
        d_srcname[1] = "aorta             ";
    }
    return;
} // CirculationModel

CirculationModel::~CirculationModel()
{
    return;
} // ~CirculationModel

void
CirculationModel::advanceTimeDependentData(const double dt,
                                           const Pointer<PatchHierarchy<NDIM>> hierarchy,
                                           const int U_idx,
                                           const int /*P_idx*/,
                                           const int /*wgt_cc_idx*/,
                                           const int wgt_sc_idx)
{
    // Compute the mean flow rates in the vicinity of the inflow and outflow
    // boundaries.
    std::fill(d_qsrc.begin(), d_qsrc.end(), 0.0);
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            if (pgeom->getTouchesRegularBoundary())
            {
                Pointer<SideData<NDIM, double>> U_data = patch->getPatchData(U_idx);
                Pointer<SideData<NDIM, double>> wgt_sc_data = patch->getPatchData(wgt_sc_idx);
                const Box<NDIM>& patch_box = patch->getBox();
                const double* const x_lower = pgeom->getXLower();
                const double* const dx = pgeom->getDx();
                double dV = 1.0;
                for (int d = 0; d < NDIM; ++d)
                {
                    dV *= dx[d];
                }
                double X[NDIM];
                static const int axis = 1;
                for (int side = 0; side <= 1; ++side)
                {
                    const bool is_lower = side == 0;
                    if (pgeom->getTouchesRegularBoundary(axis, side))
                    {
                        const double rsrc = d_rsrc[side];
                        const Point& posn = d_posn[side];
                        IBTK::Vector n;
                        for (int d = 0; d < NDIM; ++d)
                        {
                            n[d] = axis == d ? (is_lower ? -1.0 : +1.0) : 0.0;
                        }
                        Box<NDIM> side_box = patch_box;
                        if (is_lower)
                        {
                            side_box.lower(axis) = patch_box.lower(axis);
                            side_box.upper(axis) = patch_box.lower(axis);
                        }
                        else
                        {
                            side_box.lower(axis) = patch_box.upper(axis) + 1;
                            side_box.upper(axis) = patch_box.upper(axis) + 1;
                        }
                        for (Box<NDIM>::Iterator b(side_box); b; b++)
                        {
                            const hier::Index<NDIM>& i = b();
                            double r_sq = 0.0;
                            for (int d = 0; d < NDIM; ++d)
                            {
                                X[d] =
                                    x_lower[d] + dx[d] * (double(i(d) - patch_box.lower(d)) + (d == axis ? 0.0 : 0.5));
                                if (d != axis) r_sq += pow(X[d] - posn[d], 2.0);
                            }
                            const double r = sqrt(r_sq);
                            if (r <= rsrc)
                            {
                                const SideIndex<NDIM> i_s(i, axis, SideIndex<NDIM>::Lower);
                                if ((*wgt_sc_data)(i_s) > std::numeric_limits<double>::epsilon())
                                {
                                    double dA = n[axis] * dV / dx[axis];
                                    d_qsrc[side] += (*U_data)(i_s)*dA;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Convert from 2D to 3D flow rates. Conversion factor is chosen to give the rectangle a similar area as that of the
    // disk.
    d_qsrc[0] *= d_vol_conversion_fac;
    d_qsrc[1] *= d_vol_conversion_fac;
    SAMRAI_MPI::sumReduction(&d_qsrc[0], d_nsrc);

    const double t = d_time;

    // The upstream pressure is determined by the LA/LV Model
    double P_LVOT;
    double& P_LV = d_P_LV;
    double& P_LA = d_P_LA;
    double C_LA_old;
    double C_LV_old;
    double C_LA_new;
    double C_LV_new;
    const double Q_LV = -d_qsrc[0];
    Upstream_compliances(C_LA_old, E_min_LA, tau1_LA, tau2_LA, m1_LA, m2_LA, k_LA, t + T_per - t_shift_LA, dt);
    Upstream_compliances(C_LA_new, E_min_LA, tau1_LA, tau2_LA, m1_LA, m2_LA, k_LA, t + T_per - t_shift_LA + dt, dt);
    Upstream_compliances(C_LV_old, E_min_LV, tau1_LV, tau2_LV, m1_LV, m2_LV, k_LV, t, dt);
    Upstream_compliances(C_LV_new, E_min_LV, tau1_LV, tau2_LV, m1_LV, m2_LV, k_LV, t + dt, dt);
    Upstream_be_update(P_LA, P_LV, P_LVOT, C_LA_old, C_LA_new, C_LV_old, C_LV_new, Q_LV, t, dt);
    d_psrc[0] = P_LVOT * prconv;

    // The downstream pressure is determined by a three-element Windkessel
    // model.
    double& P_Wk = d_P_Wk;
    double P_Ao;
    const double Q_Ao = d_qsrc[1];
    windkessel_be_update(P_Wk, P_Ao, Q_Ao, t, dt);
    d_psrc[1] = P_Ao * prconv;

    // Update the current time.
    d_time += dt;

    // Output the updated values.
    const long precision = plog.precision();
    plog.unsetf(ios_base::showpos);
    plog.unsetf(ios_base::scientific);

    plog << "============================================================================\n"
         << "Circulation model variables at time " << d_time << ":\n";

    plog << "Valve is ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    if (P_LV < P_LA)
    {
        plog << "open    ";
    }
    else
    {
        plog << "closed    ";
    }

    plog << "\n";

    plog << "Q_LV   = ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    plog << -d_qsrc[0] / flconv << "    ";

    plog << "P_LVOT = ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    plog << d_psrc[0] / prconv << "    ";

    plog << "P_LV   = ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    plog << P_LV << "    ";

    plog << "\n";

    plog << "Q_Ao   = ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    plog << d_qsrc[1] / flconv << "    ";

    plog << "P_Ao   = ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    plog << d_psrc[1] / prconv << "    ";

    plog << "P_Wk   = ";
    plog.setf(ios_base::showpos);
    plog.setf(ios_base::scientific);
    plog.precision(12);
    plog << P_Wk << "    ";

    plog << "\n";

    plog << "flow units: liter/min    ";
    plog << "pressure units: mmHg\n";

    plog << "============================================================================\n";

    plog.unsetf(ios_base::showpos);
    plog.unsetf(ios_base::scientific);
    plog.precision(precision);

    // Write the current state to disk.
    writeDataFile();
    return;
} // advanceTimeDependentData

void
CirculationModel::putToDatabase(Pointer<Database> db)
{
    db->putDouble("d_time", d_time);
    db->putInteger("d_nsrc", d_nsrc);
    db->putDoubleArray("d_qsrc", &d_qsrc[0], d_nsrc);
    db->putDoubleArray("d_psrc", &d_psrc[0], d_nsrc);
    db->putDoubleArray("d_rsrc", &d_rsrc[0], d_nsrc);
    db->putStringArray("d_srcname", &d_srcname[0], d_nsrc);
    db->putDouble("d_P_Wk", d_P_Wk);
    db->putDouble("d_P_LA", d_P_LA);
    db->putDouble("d_P_LV", d_P_LV);
    db->putInteger("d_bdry_interface_level_number", d_bdry_interface_level_number);
    return;
} // putToDatabase

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

void
CirculationModel::writeDataFile() const
{
    static const int mpi_root = 0;
    if (SAMRAI_MPI::getRank() == mpi_root)
    {
        static bool file_initialized = false;
        const bool from_restart = RestartManager::getManager()->isFromRestart();
        if (!from_restart && !file_initialized)
        {
            ofstream fout(DATA_FILE_NAME.c_str(), ios::out);
            fout << "time,"
                 << "p_LVOT_mmHg,"
                 << "p_aorta_mmHg,"
                 << "q_LV_l/min,"
                 << "q_aorta_l/min,"
                 << "p_LA_mmHg"
                 << "p_LV_mmHg"
                 << "p_Wk_mmHg"
                 << "\n";
            file_initialized = true;
        }

        ofstream fout(DATA_FILE_NAME.c_str(), ios::app);
        fout.unsetf(ios_base::showpos);
        fout.setf(ios_base::scientific);
        fout.precision(12);
        fout << d_time;
        fout.setf(ios_base::scientific);
        fout.setf(ios_base::showpos);
        fout.precision(12);
        fout << "," << d_psrc[0] / prconv;
        fout << "," << d_psrc[1] / prconv;
        fout.setf(ios_base::scientific);
        fout.setf(ios_base::showpos);
        fout.precision(12);
        fout << "," << -d_qsrc[0] / flconv;
        fout << "," << d_qsrc[1] / flconv;
        fout.setf(ios_base::scientific);
        fout.setf(ios_base::showpos);
        fout.precision(12);
        fout << "," << d_P_LA;
        fout.setf(ios_base::scientific);
        fout.setf(ios_base::showpos);
        fout.precision(12);
        fout << "," << d_P_LV;
        fout.setf(ios_base::scientific);
        fout.setf(ios_base::showpos);
        fout.precision(12);
        fout << "," << d_P_Wk;
        fout << "\n";
    }
    return;
} // writeDataFile

void
CirculationModel::getFromRestart()
{
    Pointer<Database> restart_db = RestartManager::getManager()->getRootDatabase();
    Pointer<Database> db;
    if (restart_db->isDatabase(d_object_name))
    {
        db = restart_db->getDatabase(d_object_name);
    }
    else
    {
        TBOX_ERROR("Restart database corresponding to " << d_object_name << " not found in restart file.");
    }

    d_time = db->getDouble("d_time");
    d_nsrc = db->getInteger("d_nsrc");
    d_qsrc.resize(d_nsrc);
    d_psrc.resize(d_nsrc);
    d_rsrc.resize(d_nsrc);
    d_srcname.resize(d_nsrc);
    db->getDoubleArray("d_qsrc", &d_qsrc[0], d_nsrc);
    db->getDoubleArray("d_psrc", &d_psrc[0], d_nsrc);
    db->getDoubleArray("d_rsrc", &d_rsrc[0], d_nsrc);
    db->getStringArray("d_srcname", &d_srcname[0], d_nsrc);
    d_P_Wk = db->getDouble("d_P_Wk");
    d_P_LA = db->getDouble("d_P_LA");
    d_P_LV = db->getDouble("d_P_LV");
    d_bdry_interface_level_number = db->getInteger("d_bdry_interface_level_number");
    return;
} // getFromRestart

/////////////////////////////// NAMESPACE ////////////////////////////////////

/////////////////////////////// TEMPLATE INSTANTIATION ///////////////////////

//////////////////////////////////////////////////////////////////////////////
