import os
from rail.core.utils import find_rail_file
from rail.creation.engine import Creator
from rail.core.stage import RailStage
from rail.core.data import Hdf5Handle
from ceci.config import StageParameter as Param
import numpy as np
from jax import vmap
from jax import jit as jjit
from dsps import calc_rest_mag
from dsps import calc_obs_mag
from dsps import load_ssp_templates
from dsps import load_transmission_curve
from dsps.cosmology import DEFAULT_COSMOLOGY


class DSPSPhotometryCreator(Creator):
    """
    Derived class of Creator that generate synthetic absolute and apparent magnitudes from one or more SED models
    generated with the DSPSSingleSedModeler or DSPSPopulationSedModeler classes.
    It accepts as input Hdf5Handles containing the rest-frame SEDs in units of Lsun/Hz and outputs an Hdf5Handle
    containing sequential indices, absolute and apparent magnitudes for each galaxy.
    Photometric quantities are computed for the filters defined in the configuration file.

    Notes
    -----
    jax serially execute the computations on CPU on single core, for CPU parallelization you need MPI.
    If GPU is used, jax natively and automatically parallelize the execution.

    """

    name = "DSPSPhotometryCreator"
    default_files_folder = find_rail_file(os.path.join('examples_data', 'creation_data', 'data', 'dsps_default_data'))
    config_options = RailStage.config_options.copy()
    config_options.update(redshift_key=Param(str, 'redshifts', msg='Redshift keyword name of the hdf5 dataset '
                                                                   'containing rest-frame SEDs'),
                          restframe_sed_key=Param(str, 'restframe_seds', msg='Rest-frame SED keyword name of the '
                                                                             'hdf5 dataset containing rest-frame SEDs'),
                          absolute_mags_key=Param(str, 'rest_frame_absolute_mags', msg='Absolute magnitudes keyword'
                                                                                       ' name of the output hdf5 '
                                                                                       'dataset'),
                          apparent_mags_key=Param(str, 'apparent_mags', msg='Apparent magnitudes keyword name of the '
                                                                            'output hdf5 dataset'),
                          filter_folder=Param(str, os.path.join(default_files_folder, 'filters'),
                                              msg='Folder containing filter transmissions'),
                          instrument_name=Param(str, 'lsst', msg='Instrument name as prefix to filter transmission'
                                                                 ' files'),
                          wavebands=Param(str, 'u,g,r,i,z,y', msg='Comma-separated list of wavebands'),
                          ssp_templates_file=Param(str, os.path.join(default_files_folder,
                                                                     'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                   msg='hdf5 file storing the SSP libraries used to create SEDs'),
                          default_cosmology = Param(bool, True, msg='True to use default DSPS cosmology. If False,'
                                                                    'Om0, w0, wa, h need to be supplied in the '
                                                                    'sample function'))

    inputs = [("model", Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        Initialize DSPSPhotometryCreator class. If the SSP templates are not provided by the user, they are automatically
        downloaded from the public NERSC directory. These default templates are created with default FSPS values,
        with gas emission at fixed gas solar metallicity value.
        The _b and _c tuples for jax are composed of None or 0, depending on whether you don't or do want the
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

        if not os.path.isdir(self.config.filter_folder):
            raise OSError("File {self.config.filter_folder} not found")

        self.wavebands = self.config.wavebands.split(',')
        self.filter_wavelengths = np.array([load_transmission_curve(fn=os.path.join(self.config.filter_folder,
                                                                                    '{}_{}_transmission.h5'
                                                                                    .format(self.config.instrument_name,
                                                                                            waveband))).wave
                                            for waveband in self.wavebands], dtype=object)

        self.filter_transmissions = np.array([load_transmission_curve(fn=os.path.join(self.config.filter_folder,
                                                                                      '{}_{}_transmission.h5'
                                                                                      .format(self.config.
                                                                                              instrument_name,
                                                                                              waveband))).transmission
                                             for waveband in self.wavebands], dtype=object)

    def sample(self, seed: int = None,
               input_data=os.path.join(default_files_folder, 'model_DSPSPopulationSedModeler.hdf5'),
               Om0=DEFAULT_COSMOLOGY.Om0, w0=DEFAULT_COSMOLOGY.w0, wa=DEFAULT_COSMOLOGY.wa,
               h=DEFAULT_COSMOLOGY.h, **kwargs):
        r"""
        Creates observed and absolute magnitudes for the population of galaxy rest-frame SEDs and stores them into
        an Hdf5Handle.

        Parameters
        ----------
        seed: int
            The random seed to control sampling
        input_data: str
            Filepath to the hdf5 table containing the galaxy rest-frame SEDs.
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
        output: Hdf5Handle
            Hdf5Handle storing the absolute and apparent magnitudes.

        Notes
        -----
        This method puts  `seed` into the stage configuration data, which makes them available to other methods.
        It then calls the `run` method. Finally, the `Hdf5Handle` associated to the `output` tag is returned.

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
        self.config["seed"] = seed
        self.config.update(**kwargs)
        self.set_data('model', input_data)
        self.run()
        self.finalize()
        output = self.get_handle("output")
        return output

    def _compute_rest_frame_absolute_magnitudes(self, ssp_data, rest_frame_seds, filter_wavelengths,
                                                filter_transmissions):
        """
        Computes the absolute magnitudes of the input rest-frame SEDs using the DSPS _calc_rest_mag_vmap function.

        Parameters
        ----------
        ssp_data
            SSP templates created with FSPS
        rest_frame_seds
            Rest-frame SEDs in units of Lsun/Hz generated using DSPS v3.* (Hearin+21).
        filter_wavelengths
            Array of wavelengths for each filter band.
        filter_transmissions
            Array of transmissions for each filter band.

        Returns
        -------
        restframe_abs_mags
            Array of absolute magnitudes with size (n_galaxies, n_filter_bands)

        """
        # consider the whole chunk

        self._b = [None, 0, 0, 0]
        self._calc_rest_mag_vmap = jjit(vmap(calc_rest_mag, in_axes=self._b))
        restframe_abs_mags = np.zeros((len(rest_frame_seds), len(self.wavebands)))

        for j in range(len(self.wavebands)):

            args_abs_mags = (ssp_data.ssp_wave, rest_frame_seds, np.array(list(filter_wavelengths[:, j]),
                                                                          dtype='float'),
                             np.array(list(filter_transmissions[:, j]), dtype='float'))

            restframe_abs_mags[:, j] = self._calc_rest_mag_vmap(*args_abs_mags)

        return restframe_abs_mags

    def _compute_apparent_magnitudes(self, ssp_data, rest_frame_seds, redshifts, filter_wavelengths,
                                     filter_transmissions):
        """
        Computes the apparent magnitudes of the input rest-frame SEDs using the DSPS _calc_app_mag_vmap function for
        a user-defined cosmology.

        Parameters
        ----------
        ssp_data
            SSP templates created with FSPS
        rest_frame_seds
            Rest-frame SEDs in units of Lsun/Hz generated using DSPS v3.* (Hearin+21).
        redshifts
            Array of redshifts for each galaxy
        filter_wavelengths
            Array of wavelengths for each filter band.
        filter_transmissions
            Array of transmissions for each filter band.

        Returns
        -------
        apparent_mags
            Array of apparent magnitudes with size (n_galaxies, n_filter_bands)

        """
        # consider the whole chunk

        self._c = [None, 0, 0, 0, 0, None, None, None, None]
        self._calc_app_mag_vmap = jjit(vmap(calc_obs_mag, in_axes=self._c))

        apparent_mags = np.zeros((len(rest_frame_seds), len(self.wavebands)))
        for j in range(len(self.wavebands)):

            args_app_mags = (ssp_data.ssp_wave, rest_frame_seds, np.array(list(filter_wavelengths[:, j]),
                                                                          dtype='float'),
                             np.array(list(filter_transmissions[:, j]), dtype='float'), redshifts,
                             self.config.Om0, self.config.w0, self.config.wa, self.config.h)

            apparent_mags[:, j] = self._calc_app_mag_vmap(*args_app_mags)

        return apparent_mags

    def run(self):
        """
        This function computes rest-frame absolute magnitudes in the provided wavebands for all the galaxies
        in the population by calling `_calc_rest_mag_vmap` from DSPS. It does the same for the observed
        magnitudes in the AB system by calling `_calc_obs_mag_vmap` from DSPS.
        It then stores both kind of magnitudes and the galaxy indices into an Hdf5Handle.

        Returns
        -------

        """

        self.model = self.get_data('model')

        redshifts = self.model[self.config.redshift_key][()]
        rest_frame_seds = self.model[self.config.restframe_sed_key][()]

        ssp_data = load_ssp_templates(fn=self.config.ssp_templates_file)
        filter_wavelengths = np.stack((self.filter_wavelengths,) * len(redshifts), axis=0)
        filter_transmissions = np.stack((self.filter_transmissions,) * len(redshifts), axis=0)

        rest_frame_absolute_mags = self._compute_rest_frame_absolute_magnitudes(ssp_data, rest_frame_seds,
                                                                                filter_wavelengths,
                                                                                filter_transmissions)
        apparent_mags = self._compute_apparent_magnitudes(ssp_data, rest_frame_seds, redshifts, filter_wavelengths,
                                                          filter_transmissions)

        idxs = np.arange(1, len(redshifts) + 1, 1, dtype=int)
        output_mags = {'id': idxs, self.config.absolute_mags_key: rest_frame_absolute_mags,
                       self.config.apparent_mags_key: apparent_mags}
        self.add_data('output', output_mags)
