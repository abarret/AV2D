// Filename: CirculationModel.h
// Created on 04 May 2007 by Boyce Griffith

#ifndef included_CirculationModel
#define included_CirculationModel

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/ibtk_utilities.h>

#include <tbox/Database.h>
#include <tbox/Serializable.h>

#include <CellVariable.h>

#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Class CirculationModel
 */
class CirculationModel : public SAMRAI::tbox::Serializable
{
public:
    /*!
     * \brief The object name.
     */
    std::string d_object_name;

    /*!
     * \brief Whether the object is registered with the restart manager.
     */
    bool d_registered_for_restart;

    /*!
     * \brief Windkessel model data.
     */
    double d_time;
    int d_nsrc;
    std::vector<double> d_qsrc, d_psrc, d_rsrc;
    std::vector<IBTK::Point> d_posn;
    std::vector<std::string> d_srcname;
    double d_P_Wk, d_P_LA, d_P_LV;
    double d_vol_conversion_fac = std::numeric_limits<double>::quiet_NaN();

    /*!
     * \brief The level of the patch hierarchy on which the Lagrangian
     * structures that interface the boundary are located.
     */
    int d_bdry_interface_level_number;

    /*!
     * \brief Constructor
     */
    CirculationModel(const std::string& object_name,
                     SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                     bool register_for_restart = true);

    /*!
     * \brief Destructor.
     */
    virtual ~CirculationModel();

    /*!
     * \brief Advance time-dependent data.
     */
    void advanceTimeDependentData(const double dt,
                                  const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                  const int U_idx,
                                  const int P_idx,
                                  const int wgt_cc_idx,
                                  const int wgt_sc_idx);

    /*!
     * \name Implementation of Serializable interface.
     */

    /*!
     * Write out object state to the given database.
     *
     * When assertion checking is active, database point must be non-null.
     */
    void putToDatabase(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db);

private:
    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    CirculationModel(const CirculationModel& from);

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    CirculationModel& operator=(const CirculationModel& that);

    /*!
     * Write out source/sink state data to disk.
     */
    void writeDataFile() const;

    /*!
     * Read object state from the restart file and initialize class data
     * members.  The database from which the restart data is read is determined
     * by the object_name specified in the constructor.
     *
     * Unrecoverable Errors:
     *
     *    -   The database corresponding to object_name is not found in the
     *        restart file.
     *
     *    -   The class version number and restart version number do not match.
     *
     */
    void getFromRestart();
};
#endif //#ifndef included_CirculationModel
