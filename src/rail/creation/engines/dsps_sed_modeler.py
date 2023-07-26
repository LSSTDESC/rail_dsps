import os
from rail.creation.engine import Modeler
from rail.core.stage import RailStage
from rail.core.utils import find_rail_file
from rail.core.data import Hdf5Handle
from ceci.config import StageParameter as Param
import numpy as np
from jax import vmap
from jax import jit as jjit
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import load_ssp_templates
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_sed_sfh_table_met_table


class DSPSSingleSedModeler(Modeler):
    r"""
    Derived class of Modeler for creating a single galaxy rest-frame SED model using DSPS v3.* (Hearin+21).
    SPS calculations are based on a set of template SEDs of simple stellar populations (SSPs).
    Supplying such templates is outside the planned scope of the DSPS package, and so they
    will need to be retrieved from some other library. For example, the FSPS library supplies
    such templates in a convenient form.

    The input galaxy properties, such as star-formation histories and metallicities, need to be supplied via an
    hdf5 table.

    Notes
    -----
    The user-provided metallicity grid should be consistently defined with the metallicity of the templates SEDs.
    Users should be cautious in the use of the cosmic time grid. The time resolution strongly depends on the
    user scientific aim.

    """

    name = "DSPSSingleSedModeler"
    default_files_folder = find_rail_file(os.path.join('examples_data', 'creation_data', 'data', 'dsps_default_data'))
    config_options = RailStage.config_options.copy()
    config_options.update(ssp_templates_file=Param(str, os.path.join(default_files_folder,
                                                                     'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                   msg='hdf5 file storing the SSP libraries used to create SEDs'),
                          redshift_key=Param(str, 'redshifts',
                                             msg='Redshift keyword name of the hdf5 dataset'),
                          cosmic_time_grid_key=Param(str, 'cosmic_time_grid',
                                                     msg='Cosmic time grid keyword name of the hdf5 dataset, '
                                                         'this is the grid of Universe age over which the stellar mass'
                                                         ' build-up takes place in units of Gyr'),
                          star_formation_history_key=Param(str, 'star_formation_history',
                                                           msg='Star-formation history keyword name of the hdf5 '
                                                               'dataset, this is the star-formation history of the '
                                                               'galaxy in units of Msun/yr'),
                          stellar_metallicity_key=Param(str, 'stellar_metallicity',
                                                        msg='Stellar metallicity keyword name of the hdf5 dataset, '
                                                            'this is the stellar metallicity in units of log10(Z)'),
                          stellar_metallicity_scatter_key=Param(str, 'stellar_metallicity_scatter',
                                                                msg='Stellar metallicity scatter keyword name of the '
                                                                    'hdf5 dataset, this is lognormal scatter in the'
                                                                    ' metallicity distribution function'),
                          restframe_sed_key=Param(str, 'restframe_sed', msg='Rest-frame SED keyword name of the '
                                                                            'output hdf5 dataset'),
                          default_cosmology=Param(bool, True, msg='True to use default DSPS cosmology. If False,'
                                                                  'Om0, w0, wa, h need to be supplied in the '
                                                                  'fit_model function'))

    inputs = [("input", Hdf5Handle)]
    outputs = [("model", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        Initialize SedModeler class. If the SSP templates are not provided by the user, they are automatically
        downloaded from the public NERSC directory. These default templates are created with default FSPS values,
        with gas emission at fixed gas solar metallicity value.

        Parameters
        ----------
        args:
        comm:

        """
        RailStage.__init__(self, args, comm=comm)

        if not os.path.isfile(self.config.ssp_templates_file):
            default_files_folder = find_rail_file(os.path.join('examples_data', 'creation_data', 'data',
                                                  'dsps_default_data'))
            os.system('curl -O https://portal.nersc.gov/cfs/lsst/schmidt9/ssp_data_fsps_v3.2_lgmet_age.h5 '
                      '--output-dir {}'.format(default_files_folder))

    def _get_rest_frame_seds(self, ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                             stellar_metallicities, stellar_metallicities_scatter):
        """
        Computes the rest-frame SED with DSPS based on user-supplied input galaxy population properties.
        The functions calc_rest_sed_sfh_table_lognormal_mdf and calc_rest_sed_sfh_table_met_table
        return a RestSED object composed of
        rest_sedndarray of shape (n_wave, )
            Restframe SED of the galaxy in units of Lsun/Hz
        weightsndarray of shape (n_met, n_ages, 1)
            SSP weights of the joint distribution of stellar age and metallicity
        lgmet_weightsndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity
        age_weightsndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

        Parameters
        ----------
        ssp_data
            SSP templates created with FSPS
        redshifts
            Array of redshifts for each galaxy
        cosmic_time_grids
            Array of ages of the universe in Gyr at which the input galaxy SFH and metallicity have been tabulated
        star_formation_histories
            Star formation history in Msun/yr evaluated at the input cosmic_time_grids
        stellar_metallicities
            If a scalar value, log10(Z) of the galaxy at the time of observation, else Metallicity history evaluated
            at the input cosmic_time_grids
        stellar_metallicities_scatter
            Lognormal scatter in metallicity

        Returns
        -------
        restframe_seds
            Array of rest-frame SEDs generated with DSPS for each input galaxy.
        """

        restframe_seds = {}

        for i in self.split_tasks_by_rank(range(len(redshifts))):
            t_obs = age_at_z(redshifts[i], self.config.Om0, self.config.w0, self.config.wa, self.config.h)
            if t_obs[0] > cosmic_time_grids[i][-1]:
                t_obs = cosmic_time_grids[i][-1]
            else:
                t_obs = t_obs[0]

            if np.isscalar(stellar_metallicities[i]):
                restframe_sed = calc_rest_sed_sfh_table_lognormal_mdf(cosmic_time_grids[i], star_formation_histories[i],
                                                                      stellar_metallicities[i],
                                                                      stellar_metallicities_scatter[i],
                                                                      ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr,
                                                                      ssp_data.ssp_flux, t_obs)
            elif len(stellar_metallicities[i]) > 1:
                restframe_sed = calc_rest_sed_sfh_table_met_table(cosmic_time_grids[i], star_formation_histories[i],
                                                                  stellar_metallicities[i],
                                                                  stellar_metallicities_scatter[i], ssp_data.ssp_lgmet,
                                                                  ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)
            else:
                raise ValueError

            restframe_seds[i] = restframe_sed.rest_sed

        if self.comm is not None:  # pragma: no cover
            restframe_seds = self.comm.gather(restframe_seds)

            if self.rank != 0:  # pragma: no cover
                return None, None

            restframe_seds = {k: v for a in restframe_seds for k, v in a.items()}

        restframe_seds = np.array([restframe_seds[i] for i in range(len(redshifts))])

        return restframe_seds

    def fit_model(self, input_data=os.path.join(default_files_folder, 'input_galaxy_properties_dsps.hdf5'),
                  Om0=DEFAULT_COSMOLOGY.Om0, w0=DEFAULT_COSMOLOGY.w0, wa=DEFAULT_COSMOLOGY.wa,
                  h=DEFAULT_COSMOLOGY.h):
        """
        This function generates the rest-frame SEDs and stores them into the Hdf5Handle.

        Parameters
        ----------
        input_data: str
            Filepath to the hdf5 table containing galaxy properties.
        Om0: float
            Omega matter: density of non-relativistic matter in units of the critical density at z=0.
        w0: float
            Dark energy equation of state at z=0 (a=1). This is pressure/density for dark energy in units where c=1.
        wa: float
            Negative derivative of the dark energy equation of state with respect to the scale factor.
            A cosmological constant has w0=-1.0 and wa=0.0.
        h: float
            dimensionless Hubble constant at z=0.

        Returns
        -------
        model: Hdf5Handle
            Hdf5Handle storing the rest-frame SED model
        """
        if self.config.default_cosmology:
            self.config.Om0 = DEFAULT_COSMOLOGY.Om0
            self.config.w0 = DEFAULT_COSMOLOGY.w0
            self.config.wa = DEFAULT_COSMOLOGY.wa
            self.config.h = DEFAULT_COSMOLOGY.h
        else:
            self.config.Om0 = Om0
            self.config.w0 = w0
            self.config.wa = wa
            self.config.h = h
        self.set_data('input', input_data)
        self.run()
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self):
        """
        Run method. It Calls `_get_rest_frame_seds` from DSPS to create a galaxy rest-frame SED.
        The load_ssp_templates function loads the SSP templates created with FSPS. The resulting NamedTuple has
        4 entries:
        ssp_lgmetndarray of shape (n_met, )
            Array of log10(Z) of the SSP templates where dimensionless Z is the mass fraction of elements heavier than He
        ssp_lg_age_gyrndarray of shape (n_ages, )
            Array of log10(age/Gyr) of the SSP templates
        ssp_wave : ndarray of shape (n_wave, )
        ssp_fluxndarray of shape (n_met, n_ages, n_wave)
            SED of the SSP in units of Lsun/Hz/Msun

        Notes
        -----
        The initial stellar mass of the galaxy is 0.
        The definition of the stellar mass table as cumulative sum refers to the total stellar mass formed.
        DSPS conveniently provides IMF-dependent fitting functions to compute the surviving mass
        (see surviving_mstar.py).
        The units of the resulting rest-frame SED is solar luminosity per Hertz. The luminosity refers to that
        emitted by the formed mass at the time of observation.

        Returns
        -------

        """
        input_galaxy_properties = self.get_data('input')
        ssp_data = load_ssp_templates(fn=self.config.ssp_templates_file)

        redshifts = input_galaxy_properties[self.config.redshift_key][()]
        cosmic_time_grids = input_galaxy_properties[self.config.cosmic_time_grid_key][()]
        star_formation_histories = input_galaxy_properties[self.config.star_formation_history_key][()]
        stellar_metallicities = input_galaxy_properties[self.config.stellar_metallicity_key][()]
        stellar_metallicities_scatter = input_galaxy_properties[self.config.stellar_metallicity_scatter_key][()]

        restframe_seds = self._get_rest_frame_seds(ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                                                   stellar_metallicities, stellar_metallicities_scatter)

        if self.rank == 0:
            rest_frame_sed_models = {self.config.restframe_sed_key: restframe_seds,
                                     self.config.redshift_key: redshifts}
            self.add_data('model', rest_frame_sed_models)


class DSPSPopulationSedModeler(Modeler):
    r"""
    Derived class of Modeler for creating a population of galaxy rest-frame SED models using DSPS v3.* (Hearin+21).
    SPS calculations are based on a set of template SEDs of simple stellar populations (SSPs).
    Supplying such templates is outside the planned scope of the DSPS package, and so they
    will need to be retrieved from some other library. For example, the FSPS library supplies
    such templates in a convenient form.

    The input galaxy properties, such as star-formation histories and metallicities, need to be supplied via an
    hdf5 table.

    Notes
    -----
    The user-provided metallicity grid should be consistently defined with the metallicity of the templates SEDs.
    Users should be cautious in the use of the cosmic time grid. The time resolution strongly depends on the
    user scientific aim.
    jax serially execute the computations on CPU on single core, for CPU parallelization you need MPI.
    If GPU is used, jax natively and automatically parallelize the execution.
    """

    name = "DSPSPopulationSedModeler"
    default_files_folder = find_rail_file(os.path.join('examples_data', 'creation_data', 'data', 'dsps_default_data'))
    config_options = RailStage.config_options.copy()
    config_options.update(ssp_templates_file=Param(str, os.path.join(default_files_folder,
                                                                     'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                   msg='hdf5 file storing the SSP libraries used to create SEDs'),
                          redshift_key=Param(str, 'redshift',
                                             msg='Redshift keyword name of the hdf5 dataset'),
                          cosmic_time_grid_key=Param(str, 'cosmic_time_grid',
                                                     msg='Cosmic time grid keyword name of the hdf5 dataset, '
                                                         'this is the grid of Universe age over which the stellar mass'
                                                         ' build-up takes place in units of Gyr'),
                          star_formation_history_key=Param(str, 'star_formation_history',
                                                           msg='Star-formation history keyword name of the hdf5 '
                                                               'dataset, this is the star-formation history of the'
                                                               ' galaxy in units of Msun/yr'),
                          stellar_metallicity_key=Param(str, 'stellar_metallicity',
                                                        msg='Stellar metallicity keyword name of the hdf5 dataset, '
                                                            'this is the stellar metallicity in units of log10(Z)'),
                          stellar_metallicity_scatter_key=Param(str, 'stellar_metallicity_scatter',
                                                                msg='Stellar metallicity scatter keyword name of the '
                                                                    'hdf5 dataset, this is lognormal scatter in the '
                                                                    'metallicity distribution function'),
                          restframe_sed_key=Param(str, 'restframe_seds', msg='Rest-frame SED keyword name of the '
                                                                             'output hdf5 dataset'),
                          default_cosmology = Param(bool, True, msg='True to use default DSPS cosmology. If False,'
                                                                    'Om0, w0, wa, h need to be supplied in the '
                                                                    'fit_model function'))

    inputs = [("input", Hdf5Handle)]
    outputs = [("model", Hdf5Handle)]

    def __init__(self, args, comm=None):
        r"""
        Initialize SedModeler class. If the SSP templates are not provided by the user, they are automatically
        downloaded from the public NERSC directory. These default templates are created with default FSPS values,
        with gas emission at fixed gas solar metallicity value.
        The _a tuple for jax is composed of None or 0, depending on whether you don't or do want the
        array axis to map over for all arguments.

        Parameters
        ----------
        args:
        comm:
        """

        RailStage.__init__(self, args, comm=comm)

        if not os.path.isfile(self.config.ssp_templates_file):
            default_files_folder = find_rail_file(os.path.join('examples_data', 'creation_data', 'data',
                                                  'dsps_default_data'))
            os.system('curl -O https://portal.nersc.gov/cfs/lsst/schmidt9/ssp_data_fsps_v3.2_lgmet_age.h5 '
                      '--output-dir {}'.format(default_files_folder))

    def _get_rest_frame_seds(self, ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                             stellar_metallicities, stellar_metallicities_scatter):
        """
        Computes the rest-frame SED with DSPS based on user-supplied input galaxy population properties.
        The functions calc_rest_sed_sfh_table_lognormal_mdf and calc_rest_sed_sfh_table_met_table
        return a RestSED object composed of
        rest_sedndarray of shape (n_wave, )
            Restframe SED of the galaxy in units of Lsun/Hz
        weightsndarray of shape (n_met, n_ages, 1)
            SSP weights of the joint distribution of stellar age and metallicity
        lgmet_weightsndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity
        age_weightsndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

        Parameters
        ----------
        ssp_data
            SSP templates created with FSPS
        redshifts
            Array of redshifts for each galaxy
        cosmic_time_grids
            Array of ages of the universe in Gyr at which the input galaxy SFH and metallicity have been tabulated
        star_formation_histories
            Star formation history in Msun/yr evaluated at the input cosmic_time_grids
        stellar_metallicities
            If a scalar value, log10(Z) of the galaxy at the time of observation, else Metallicity history evaluated
            at the input cosmic_time_grids
        stellar_metallicities_scatter
            Lognormal scatter in metallicity

        Returns
        -------
        restframe_seds_galpop.rest_sed
            Array of rest-frame SEDs generated with DSPS for each input galaxy.
        """

        # consider the whole chunk
        self._a = (0, 0, 0, 0, None, None, None, 0)

        if np.isscalar(stellar_metallicities[0]):
            self._calc_sed_vmap = jjit(vmap(calc_rest_sed_sfh_table_lognormal_mdf, in_axes=self._a))
        elif len(stellar_metallicities[0]) > 1:
            self._calc_sed_vmap = jjit(vmap(calc_rest_sed_sfh_table_met_table, in_axes=self._a))
        else:
            raise ValueError

        self._b = (0, None, None, None, None)
        self._calc_age_at_z_vmap = jjit(vmap(age_at_z, in_axes=self._b))
        args_pop_z = (redshifts, self.config.Om0, self.config.w0, self.config.wa, self.config.h)
        t_obs = self._calc_age_at_z_vmap(*args_pop_z)[:, 0]

        args_pop = (cosmic_time_grids, star_formation_histories, stellar_metallicities,
                    stellar_metallicities_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux,
                    t_obs)

        restframe_seds_galpop = self._calc_sed_vmap(*args_pop)

        return restframe_seds_galpop.rest_sed

    def fit_model(self, input_data=os.path.join(default_files_folder, 'input_galaxy_properties_dsps.hdf5'),
                  Om0=DEFAULT_COSMOLOGY.Om0, w0=DEFAULT_COSMOLOGY.w0, wa=DEFAULT_COSMOLOGY.wa,
                  h=DEFAULT_COSMOLOGY.h):
        """
        This function generates the rest-frame SEDs and stores them into the Hdf5Handle.

        Parameters
        ----------
        input_data: str
            Filepath to the hdf5 table containing galaxy properties.
        Om0: float
            Omega matter: density of non-relativistic matter in units of the critical density at z=0.
        w0: float
            Dark energy equation of state at z=0 (a=1). This is pressure/density for dark energy in units where c=1.
        wa: float
            Negative derivative of the dark energy equation of state with respect to the scale factor.
            A cosmological constant has w0=-1.0 and wa=0.0.
        h: float
            dimensionless Hubble constant at z=0.

        Returns
        -------
        model: Hdf5Handle
            Hdf5Handle storing the rest-frame SED model
        """
        if self.config.default_cosmology:
            self.config.Om0 = DEFAULT_COSMOLOGY.Om0
            self.config.w0 = DEFAULT_COSMOLOGY.w0
            self.config.wa = DEFAULT_COSMOLOGY.wa
            self.config.h = DEFAULT_COSMOLOGY.h
        else:
            self.config.Om0 = Om0
            self.config.w0 = w0
            self.config.wa = wa
            self.config.h = h
        self.set_data('input', input_data)
        self.run()
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self):
        """
        Run method. It Calls `_get_rest_frame_seds` from DSPS to create rest-frame SEDs for a population of galaxies.
        The load_ssp_templates function loads the SSP templates created with FSPS. The resulting NamedTuple has
        4 entries:
        ssp_lgmetndarray of shape (n_met, )
            Array of log10(Z) of the SSP templates where dimensionless Z is the mass fraction of elements heavier than He
        ssp_lg_age_gyrndarray of shape (n_ages, )
            Array of log10(age/Gyr) of the SSP templates
        ssp_wave : ndarray of shape (n_wave, )
        ssp_fluxndarray of shape (n_met, n_ages, n_wave)
            SED of the SSP in units of Lsun/Hz/Msun

        Notes
        -----
        The initial stellar mass of the galaxy is 0.
        The definition of the stellar mass table as cumulative sum refers to the total stellar mass formed.
        DSPS conveniently provides IMF-dependent fitting functions to compute the surviving mass
        (see surviving_mstar.py).
        The units of the resulting rest-frame SED is solar luminosity per Hertz. The luminosity refers to that
        emitted by the formed mass at the time of observation.

        Returns
        -------
        """
        input_galaxy_properties = self.get_data('input')
        ssp_data = load_ssp_templates(fn=self.config.ssp_templates_file)

        redshifts = input_galaxy_properties[self.config.redshift_key][()]
        cosmic_time_grids = input_galaxy_properties[self.config.cosmic_time_grid_key][()]
        star_formation_histories = input_galaxy_properties[self.config.star_formation_history_key][()]
        stellar_metallicities = input_galaxy_properties[self.config.stellar_metallicity_key][()]
        stellar_metallicities_scatter = input_galaxy_properties[self.config.stellar_metallicity_scatter_key][()]

        restframe_seds = self._get_rest_frame_seds(ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                                                   stellar_metallicities, stellar_metallicities_scatter)

        rest_frame_sed_models = {self.config.restframe_sed_key: restframe_seds,
                                 self.config.redshift_key: redshifts}
        self.add_data('model', rest_frame_sed_models)
